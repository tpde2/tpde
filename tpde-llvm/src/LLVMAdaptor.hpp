// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <ranges>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include "base.hpp"
#include "tpde/RegisterFile.hpp"
#include "tpde/ValLocalIdx.hpp"
#include "tpde/base.hpp"
#include "tpde/util/SmallVector.hpp"
#include "tpde/util/misc.hpp"

namespace tpde_llvm {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"

// very hacky
inline u32 &val_idx_for_inst(llvm::Instruction *inst) {
  return *reinterpret_cast<u32 *>(reinterpret_cast<u8 *>(inst) +
                                  offsetof(llvm::Instruction, DebugMarker) - 4);
  // static_assert(sizeof(llvm::Instruction) == 64);
}

inline u32 val_idx_for_inst(const llvm::Instruction *inst) {
  return *reinterpret_cast<const u32 *>(
      reinterpret_cast<const u8 *>(inst) +
      offsetof(llvm::Instruction, DebugMarker) - 4);
  // static_assert(sizeof(llvm::Instruction) == 64);
}

#if LLVM_VERSION_MAJOR < 20
// LLVM 20+ has BasicBlock::getNumber()
inline u32 &block_embedded_idx(llvm::BasicBlock *block) {
  return *reinterpret_cast<u32 *>(
      reinterpret_cast<u8 *>(block) +
      offsetof(llvm::BasicBlock, IsNewDbgInfoFormat) + 4);
}

inline u32 block_embedded_idx(const llvm::BasicBlock *block) {
  return block_embedded_idx(const_cast<llvm::BasicBlock *>(block));
}
#endif

#pragma GCC diagnostic pop

// the basic types we handle, the actual compiler can figure out the parts
enum class LLVMBasicValType : u8 {
  invalid,
  none,
  i1, ///< Pseudo-type for i1, not used for values, only as vector element type.
  i8,
  i16,
  i32,
  i64,
  ptr,
  i128,
  f32,
  f64,
  f128,
  f80, ///< x86_fp80

  v8i8,
  v16i8,
  v4i16,
  v8i16,
  v2i32,
  v4i32,
  v2i64,
  v2f32,
  v4f32,
  v2f64,

  // i1 vectors are special. We always represent them in their bit-compact form.
  v8i1,  ///< <N x i1> for 0 < N <= 8; stored like an i8
  v16i1, ///< <N x i1> for 8 < N <= 16; stored like an i16
  v32i1, ///< <N x i1> for 16 < N <= 32; stored like an i32
  v64i1, ///< <N x i1> for 32 < N <= 64; stored like an i64

  complex, ///< Complex escape type

  max
};

/// Decompose vector type into element count and element type.
static constexpr std::pair<unsigned, LLVMBasicValType>
    basic_ty_vector_info(LLVMBasicValType bvt) {
  switch (bvt) {
    using enum LLVMBasicValType;
  case v8i8: return {8, i8};
  case v16i8: return {16, i8};
  case v4i16: return {4, i16};
  case v8i16: return {8, i16};
  case v2i32: return {2, i32};
  case v4i32: return {4, i32};
  case v2i64: return {2, i64};
  case v2f32: return {2, f32};
  case v4f32: return {4, f32};
  case v2f64: return {2, f64};
  case v8i1: return {8, i1};
  case v16i1: return {16, i1};
  case v32i1: return {32, i1};
  case v64i1: return {64, i1};
  default: TPDE_UNREACHABLE("cannot get vector element count of scalar type");
  }
}

union LLVMComplexPart {
  static constexpr u16 MaxLength = UINT16_MAX;

  struct {
    /// Type of the part.
    LLVMBasicValType type;
    /// In-memory size in bytes, e.g. 3 for i24.
    u8 size;
    /// Padding after the part for the in-memory layout.
    u8 pad_after : 7;
    /// Whether the part begins a new LLVM value. This is not the case for,
    /// e.g., the second part of i128.
    u8 ends_value : 1;
    /// Nesting depth increase before the part.
    u8 nest_inc : 4;
    /// Nesting depth decrease after the part.
    u8 nest_dec : 4;
  } part;

  struct {
    /// Number of parts following.
    u16 num_parts;

    /// Indicates that our type layout is incompatible with LLVM's layout.
    /// Example: we scalarize <1 x i8>, but LLVM-AArch64 widens to <8 x i8>.
    bool incompatible_layout : 1;

    /// At least one part has an invalid type.
    bool invalid : 1;
  } desc;

  LLVMComplexPart()
      : desc{.num_parts = 0, .incompatible_layout = false, .invalid = false} {}

  LLVMComplexPart(LLVMBasicValType type, u8 size, bool ends_value = true)
      : part{.type = type,
             .size = size,
             .pad_after = 0,
             .ends_value = ends_value,
             .nest_inc = 0,
             .nest_dec = 0} {}
};

static_assert(sizeof(LLVMComplexPart) == 4);

struct LLVMAdaptor {
  using IRValueRef = const llvm::Value *;
  using IRInstRef = const llvm::Instruction *;
  using IRBlockRef = u32;
  using IRFuncRef = llvm::Function *;

  static constexpr IRValueRef INVALID_VALUE_REF = nullptr;
  static constexpr IRBlockRef INVALID_BLOCK_REF = static_cast<IRBlockRef>(~0u);
  static constexpr IRFuncRef INVALID_FUNC_REF =
      nullptr; // NOLINT(*-misplaced-const)

  /// Threshold when PHI node operands are sorted. This allows O(log n) access
  /// to the incoming value for a given block, as opposed to O(n). This
  /// threshold is quite large, given that sorting llvm::Use-s is expensive.
  static constexpr unsigned PHINodeSortThreshold = 1024;

  const llvm::DataLayout data_layout;
  llvm::LLVMContext *context = nullptr;
  llvm::Module *mod = nullptr;

  struct ValInfo {
    LLVMBasicValType type;
    bool fused;
    u32 complex_part_tys_idx;
  };

  struct BlockAux {
    u32 aux1;
    u32 aux2;
    llvm::BasicBlock::iterator phi_end;
  };

  struct BlockInfo {
    llvm::BasicBlock *block;
    BlockAux aux;
  };

  /// Value info. Values are numbered in the following order:
  /// - 0..<arg_idx_end: Arguments
  /// - arg_idx_end..: Instructions
  tpde::util::SmallVector<ValInfo, 128> values;
  /// Map from global value to value index. Globals are the lowest values.
  /// Keep them separate so that we don't have to repeatedly insert them for
  /// every function.
  llvm::DenseMap<const llvm::GlobalValue *, u32> global_lookup;
  /// Inverse of global_lookup.
  llvm::SmallVector<const llvm::GlobalValue *, 0> global_list;
#ifndef NDEBUG
  llvm::DenseMap<const llvm::Value *, u32> value_lookup;
  #if LLVM_VERSION_MAJOR < 20
  llvm::DenseMap<const llvm::BasicBlock *, u32> block_lookup;
  #endif
#endif
  tpde::util::SmallVector<LLVMComplexPart, 32> complex_part_types;
  /// Map from complex type to the lowered type.
  llvm::DenseMap<const llvm::Type *, std::pair<LLVMBasicValType, u32>>
      complex_type_map;

  // helpers for faster lookup
  tpde::util::SmallVector<const llvm::AllocaInst *, 16>
      initial_stack_slot_indices;

  llvm::Function *cur_func = nullptr;
  bool func_unsupported = false;
  bool func_has_dynamic_alloca = false;

  tpde::util::SmallVector<BlockInfo, 128> blocks;
  tpde::util::SmallVector<u32, 256> block_succ_indices;
  tpde::util::SmallVector<std::pair<u32, u32>, 128> block_succ_ranges;

  LLVMAdaptor(llvm::DataLayout dl) : data_layout(dl) {}

  static constexpr bool TPDE_PROVIDES_HIGHEST_VAL_IDX = true;
  static constexpr bool TPDE_LIVENESS_VISIT_ARGS = true;

  [[nodiscard]] u32 func_count() const { return mod->getFunctionList().size(); }

  [[nodiscard]] auto funcs() const {
    return *mod | std::views::filter([](llvm::Function &fn) {
      return !fn.isIntrinsic();
    }) | std::views::transform([](llvm::Function &fn) { return &fn; });
  }

  [[nodiscard]] auto funcs_to_compile() const { return funcs(); }

  [[nodiscard]] static std::string_view func_link_name(const IRFuncRef func) {
    return func->getName();
  }

  [[nodiscard]] static bool func_extern(const IRFuncRef func) {
    return func->isDeclarationForLinker();
  }

  [[nodiscard]] static bool func_only_local(const IRFuncRef func) {
    if (!func->hasName() && !func->hasLocalLinkage()) [[unlikely]] {
      TPDE_LOG_WARN("unnamed functions converted to internal linkage");
      return true;
    }
    return func->hasLocalLinkage();
  }

  [[nodiscard]] static bool func_has_weak_linkage(const IRFuncRef func) {
    return func->isWeakForLinker();
  }

  [[nodiscard]] bool cur_needs_unwind_info() const {
    return cur_func->needsUnwindTableEntry();
  }

  [[nodiscard]] bool cur_is_vararg() const { return cur_func->isVarArg(); }

  [[nodiscard]] u32 cur_highest_val_idx() const {
    return values.size() + global_list.size();
  }

  [[nodiscard]] auto cur_args() const {
    return cur_func->args() |
           std::views::transform([](llvm::Argument &arg) { return &arg; });
  }

  [[nodiscard]] const auto &cur_static_allocas() const {
    return initial_stack_slot_indices;
  }

  [[nodiscard]] bool cur_has_dynamic_alloca() const {
    return func_has_dynamic_alloca;
  }

  [[nodiscard]] static IRBlockRef cur_entry_block() { return 0; }

  auto cur_blocks() const { return std::views::iota(size_t{0}, blocks.size()); }

  [[nodiscard]] IRBlockRef
      block_lookup_idx(const llvm::BasicBlock *block) const {
#if LLVM_VERSION_MAJOR >= 20
    return block->getNumber();
#else
    auto idx = block_embedded_idx(block);
  #ifndef NDEBUG
    auto it = block_lookup.find(block);
    assert(it != block_lookup.end() && it->second == idx);
  #endif
    return idx;
#endif
  }

  [[nodiscard]] auto block_succs(const IRBlockRef block) const {
    struct BlockRange {
      const IRBlockRef *block_start, *block_end;

      [[nodiscard]] const IRBlockRef *begin() const { return block_start; }

      [[nodiscard]] const IRBlockRef *end() const { return block_end; }
    };

    auto &[start, end] = block_succ_ranges[block];
    return BlockRange{block_succ_indices.data() + start,
                      block_succ_indices.data() + end};
  }

  [[nodiscard]] auto block_insts(const IRBlockRef block) const {
    const auto &aux = blocks[block].aux;
    return std::ranges::subrange(aux.phi_end, blocks[block].block->end()) |
           std::views::transform(
               [](llvm::Instruction &instr) { return &instr; });
  }

  [[nodiscard]] auto block_phis(const IRBlockRef block) const {
    const auto &aux = blocks[block].aux;
    return std::ranges::subrange(blocks[block].block->begin(), aux.phi_end) |
           std::views::transform(
               [](llvm::Instruction &instr) { return &instr; });
  }

  [[nodiscard]] u32 block_info(const IRBlockRef block) const {
    return blocks[block].aux.aux1;
  }

  void block_set_info(const IRBlockRef block, const u32 aux) {
    blocks[block].aux.aux1 = aux;
  }

  [[nodiscard]] u32 block_info2(const IRBlockRef block) const {
    return blocks[block].aux.aux2;
  }

  void block_set_info2(const IRBlockRef block, const u32 aux) {
    blocks[block].aux.aux2 = aux;
  }

  [[nodiscard]] std::string block_fmt_ref(const IRBlockRef block) const {
    std::string buf;
    llvm::raw_string_ostream os{buf};
    blocks[block].block->printAsOperand(os);
    return buf;
  }

  [[nodiscard]] std::string value_fmt_ref(const IRValueRef value) const {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    value->printAsOperand(os, /*PrintType=*/true, mod);
    return buf;
  }

  [[nodiscard]] std::string inst_fmt_ref(const IRInstRef inst) const {
    std::string buf;
    llvm::raw_string_ostream(buf) << *inst;
    return buf;
  }


  tpde::ValLocalIdx val_local_idx(const IRValueRef v) const {
    // Globals are handled together with constants; so only instructions and
    // arguments have local indices.
    if (auto *arg = llvm::dyn_cast<llvm::Argument>(v)) [[unlikely]] {
      return tpde::ValLocalIdx(arg_lookup_idx(arg));
    }
    return tpde::ValLocalIdx(inst_lookup_idx(llvm::cast<llvm::Instruction>(v)));
  }

  [[nodiscard]] auto inst_operands(const IRInstRef inst) const {
    return inst->operands() | std::views::transform([](const llvm::Use &use) {
             return use.get();
           });
  }

  [[nodiscard]] auto inst_results(const IRInstRef inst) const {
    bool is_void = inst->getType()->isVoidTy();
    return std::views::single(inst) | std::views::drop(is_void ? 1 : 0);
  }

  [[nodiscard]] bool
      val_ignore_in_liveness_analysis(const IRValueRef value) const {
    return !llvm::isa<llvm::Instruction, llvm::Argument>(value);
  }

  bool val_is_phi(IRValueRef value) const {
    return llvm::isa<llvm::PHINode>(value);
  }

  [[nodiscard]] auto val_as_phi(const IRValueRef value) const {
    struct PHIRef {
      const llvm::PHINode *phi;
      const LLVMAdaptor *self;

      [[nodiscard]] u32 incoming_count() const {
        return phi->getNumIncomingValues();
      }

      [[nodiscard]] IRValueRef incoming_val_for_slot(const u32 slot) const {
        return phi->getIncomingValue(slot);
      }

      [[nodiscard]] IRBlockRef incoming_block_for_slot(const u32 slot) const {
        return self->block_lookup_idx(phi->getIncomingBlock(slot));
      }

      [[nodiscard]] IRValueRef
          incoming_val_for_block(const IRBlockRef block) const {
        llvm::BasicBlock *bb = self->blocks[block].block;
        u32 idx;
        if (incoming_count() < PHINodeSortThreshold) [[likely]] {
          idx = phi->getBasicBlockIndex(bb); // linear search
        } else {
          idx = llvm::lower_bound(phi->blocks(), bb) - phi->block_begin();
          // NB: indices might differ, a PHI node might have the same incoming
          // pair multiple times for multi-edges.
          assert(phi->getIncomingValue(idx) ==
                 phi->getIncomingValueForBlock(bb));
        }
        return phi->getIncomingValue(idx);
      }
    };

    return PHIRef{
        .phi = llvm::cast<llvm::PHINode>(value),
        .self = this,
    };
  }

private:
  static bool is_static_alloca(const llvm::AllocaInst *alloca) {
    // Larger allocas need dynamic stack alignment. In future, we might
    // realign the stack at the beginning, but for now, treat them like
    // dynamic allocas.
    // TODO: properly support over-aligned static allocas.
    return alloca->isStaticAlloca();
  }

public:
  [[nodiscard]] u32 val_alloca_size(const IRValueRef value) const {
    const auto *alloca = llvm::cast<llvm::AllocaInst>(value);
    assert(alloca->isStaticAlloca());
    const u64 size = *alloca->getAllocationSize(mod->getDataLayout());
    assert(size <= std::numeric_limits<u32>::max());
    return size;
  }

  [[nodiscard]] u32 val_alloca_align(const IRValueRef value) const {
    const auto *alloca = llvm::cast<llvm::AllocaInst>(value);
    assert(alloca->isStaticAlloca());
    return alloca->getAlign().value();
  }

  bool cur_arg_is_byval(const u32 idx) const {
    return cur_func->hasParamAttribute(idx, llvm::Attribute::AttrKind::ByVal);
  }

  u32 cur_arg_byval_align(const u32 idx) const {
    if (auto param_align = cur_func->getParamStackAlign(idx)) {
      return param_align->value();
    }
    if (auto param_align = cur_func->getParamAlign(idx)) {
      return param_align->value();
    }
    return mod->getDataLayout()
        .getABITypeAlign(cur_func->getParamByValType(idx))
        .value();
  }

  u32 cur_arg_byval_size(const u32 idx) const {
    return mod->getDataLayout().getTypeAllocSize(
        cur_func->getParamByValType(idx));
  }

  bool cur_arg_is_sret(const u32 idx) const {
    return cur_func->hasParamAttribute(idx,
                                       llvm::Attribute::AttrKind::StructRet);
  }

  static void start_compile() {}

  static void end_compile() {}

private:
  /// Replace constant expressions with instructions. Returns pair of replaced
  /// value and first inserted instruction.
  std::pair<llvm::Value *, llvm::Instruction *>
      fixup_constant(llvm::Constant *cst,
                     llvm::BasicBlock::iterator ins_before);

  /// Handle instruction during switch_func.
  /// retval = restart from instruction, or nullptr to continue
  llvm::Instruction *handle_inst_in_block(llvm::Instruction *inst);

public:
  bool switch_func(const IRFuncRef function);

  bool switch_module(llvm::Module &mod);

  void reset();

  struct ValueParts {
    LLVMBasicValType bvt;
    const LLVMComplexPart *complex;

    u32 count() const {
      if (bvt != LLVMBasicValType::complex) {
        return LLVMAdaptor::basic_ty_part_count(bvt);
      }
      return complex->desc.num_parts;
    }

    LLVMBasicValType type(u32 n) const {
      return bvt != LLVMBasicValType::complex ? bvt : complex[n + 1].part.type;
    }

    u32 size_bytes(u32 n) const {
      return LLVMAdaptor::basic_ty_part_size(type(n));
    }

    tpde::RegBank reg_bank(u32 n) const { return basic_ty_part_bank(type(n)); }
  };

  ValueParts val_parts(const IRValueRef value) {
    if (llvm::isa<llvm::Constant>(value)) {
      auto [ty, ty_idx] = lower_type(value->getType());
      if (ty == LLVMBasicValType::complex) {
        return ValueParts{ty, &complex_part_types[ty_idx]};
      }
      return ValueParts{ty, nullptr};
    }
    return val_parts(val_local_idx(value));
  }

  ValueParts val_parts(tpde::ValLocalIdx local_idx) {
    return val_parts(values[u32(local_idx)]);
  }

  ValueParts val_parts(const ValInfo &info) const {
    if (info.type == LLVMBasicValType::complex) {
      unsigned ty_idx = info.complex_part_tys_idx;
      return ValueParts{info.type, &complex_part_types[ty_idx]};
    }
    return ValueParts{info.type, nullptr};
  }

  u32 type_part_count(LLVMBasicValType bvt, u32 complex_part_tys_idx) {
    if (bvt != LLVMBasicValType::complex) [[likely]] {
      return basic_ty_part_count(bvt);
    }
    return this->complex_part_types[complex_part_tys_idx].desc.num_parts;
  }

  [[nodiscard]] bool inst_fused(const IRInstRef inst) const {
    return val_info(inst).fused;
  }

  void inst_set_fused(const IRInstRef value, const bool fused) {
    values[inst_lookup_idx(value)].fused = fused;
  }

  const ValInfo &val_info(const llvm::Instruction *inst) const {
    return values[inst_lookup_idx(inst)];
  }

  u32 arg_lookup_idx(const llvm::Argument *arg) const {
    return arg->getArgNo();
  }

  [[nodiscard]] u32 inst_lookup_idx(const llvm::Instruction *inst) const {
    const auto idx = val_idx_for_inst(inst);
#ifndef NDEBUG
    assert(value_lookup.find(inst) != value_lookup.end() &&
           value_lookup.find(inst)->second == idx);
#endif
    return idx;
  }

  // internal helpers
  static unsigned basic_ty_part_size(const LLVMBasicValType ty) {
    switch (ty) {
      using enum LLVMBasicValType;
    case i1:
    case i8:
    case v8i1: return 1;
    case i16:
    case v16i1: return 2;
    case i32:
    case v32i1: return 4;
    case i64:
    case v64i1:
    case ptr:
    case i128: return 8;
    case f32: return 4;
    case f64: return 8;
    case f128: return 16;
    case f80: return 16;
    case v8i8: return 8;
    case v16i8: return 16;
    case v4i16: return 8;
    case v8i16: return 16;
    case v2i32: return 8;
    case v4i32: return 16;
    case v2i64: return 16;
    case v2f32: return 8;
    case v4f32: return 16;
    case v2f64: return 16;
    case complex:
    case invalid:
    case none:
    default: TPDE_UNREACHABLE("invalid basic type");
    }
  }

  static tpde::RegBank basic_ty_part_bank(const LLVMBasicValType ty) {
    switch (ty) {
      using enum LLVMBasicValType;
    case i1:
    case i8:
    case i16:
    case i32:
    case i64:
    case i128:
    case ptr:
    case v8i1:
    case v16i1:
    case v32i1:
    case v64i1: return tpde::RegBank{0};
    case f32:
    case f64:
    case f128:
    case f80: // Register moves: SSE; otherwise spilled.
    case v8i8:
    case v16i8:
    case v4i16:
    case v8i16:
    case v2i32:
    case v4i32:
    case v2i64:
    case v2f32:
    case v4f32:
    case v2f64: return tpde::RegBank{1};
    case none:
    case invalid:
    case complex:
    default: TPDE_UNREACHABLE("invalid basic type");
    }
  }

private:
  static unsigned basic_ty_part_count(const LLVMBasicValType ty) {
    return ty == LLVMBasicValType::i128 ? 2 : 1;
  }

public:
  void check_type_compatibility(llvm::Type *type) {
    const auto [bvt, complex_part_idx] = lower_type(type);
    check_type_compatibility(type, bvt, complex_part_idx);
  }

  void check_type_compatibility(llvm::Type *type,
                                LLVMBasicValType bvt,
                                u32 ty_idx) {
    switch (bvt) {
      using enum LLVMBasicValType;
    case complex:
      if (!complex_part_types[ty_idx].desc.incompatible_layout) {
        break;
      }
      [[fallthrough]];
    case v8i1:
    case v16i1:
    case v32i1:
    case v64i1:
      // i1 vectors are incompatible: we use a dense layout in general-purpose
      // registers while LLVM uses a target-specific promoted and possibly
      // widened type in vector registers. Also, we never scalarize i1 vectors,
      // they are always compact (hence, we support at most 64 elements).
      [[fallthrough]];
    case invalid:
      // invalid types are unsupported and hence incompatible
      [[unlikely]];
      report_incompatible_type(type);
      break;
    default: break;
    }
  }

private:
  [[gnu::cold]] void report_incompatible_type(llvm::Type *type);

  [[gnu::cold]] void report_unsupported_type(llvm::Type *type);

  /// Append basic types of specified type to complex_part_types. desc_idx is
  /// the index in complex_part_types containing the descriptor of the
  /// outermost type. Returns the allocation size in bytes and the alignment.
  std::pair<unsigned, unsigned> complex_types_append(llvm::Type *type,
                                                     size_t desc_idx);

  static LLVMBasicValType lower_simple_type(const llvm::Type *);

  std::pair<LLVMBasicValType, unsigned long> lower_complex_type(llvm::Type *);

public:
  std::pair<LLVMBasicValType, u32> lower_type(llvm::Type *type) {
    if (auto ty = lower_simple_type(type); ty != LLVMBasicValType::invalid)
        [[likely]] {
      return std::make_pair(ty, ~0ul);
    }
    auto [ty, num] = lower_complex_type(type);
    return std::make_pair(ty, u32(num));
  }

  std::pair<LLVMBasicValType, u32> lower_type(llvm::Value *value) {
    if (!llvm::isa<llvm::Instruction, llvm::Argument>(value)) {
      return lower_type(value->getType());
    }
    const ValInfo &info = values[u32(val_local_idx(value))];
    return {info.type, info.complex_part_tys_idx};
  }

  /// Map insertvalue/extractvalue indices to parts. Returns (first part,
  /// last part (inclusive)).
  std::pair<unsigned, unsigned>
      complex_part_for_index(IRValueRef val_idx,
                             llvm::ArrayRef<unsigned> search);
};

} // namespace tpde_llvm
