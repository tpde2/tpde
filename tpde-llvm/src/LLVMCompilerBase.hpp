// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Comdat.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalIFunc.h>
#include <llvm/IR/GlobalObject.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/AtomicOrdering.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/Support/raw_ostream.h>

#include "tpde/Assembler.hpp"
#include "tpde/CompilerBase.hpp"
#include "tpde/ValLocalIdx.hpp"
#include "tpde/ValueAssignment.hpp"
#include "tpde/base.hpp"
#include "tpde/util/BumpAllocator.hpp"
#include "tpde/util/SmallVector.hpp"
#include "tpde/util/misc.hpp"

#include "JITMapper.hpp"
#include "LLVMAdaptor.hpp"
#include "tpde-llvm/LLVMCompiler.hpp"

namespace tpde_llvm {

template <typename Adaptor, typename Derived, typename Config>
struct LLVMCompilerBase : public LLVMCompiler,
                          tpde::CompilerBase<LLVMAdaptor, Derived, Config> {
  // TODO
  using Base = tpde::CompilerBase<LLVMAdaptor, Derived, Config>;

  using IRValueRef = typename Base::IRValueRef;
  using IRBlockRef = typename Base::IRBlockRef;
  using IRFuncRef = typename Base::IRFuncRef;
  using ScratchReg = typename Base::ScratchReg;
  using ValuePartRef = typename Base::ValuePartRef;
  using ValuePart = typename Base::ValuePart;
  using ValueRef = typename Base::ValueRef;
  using GenericValuePart = typename Base::GenericValuePart;
  using InstRange = typename Base::InstRange;

  using SecRef = typename tpde::SecRef;
  using SymRef = typename tpde::SymRef;

  using AsmReg = typename Base::AsmReg;

  using ValInfo = typename Adaptor::ValInfo;

  struct ValRefSpecial {
    uint8_t mode = 4;
    IRValueRef value;
  };

  struct RelocInfo {
    enum RELOC_TYPE : uint8_t {
      RELOC_ABS,
      RELOC_PC32,
    };

    uint32_t off;
    int32_t addend;
    SymRef sym;
    RELOC_TYPE type = RELOC_ABS;
  };

  struct ResolvedGEP {
    std::variant<ValuePartRef, ScratchReg> base;
    std::optional<std::variant<ValuePartRef, ScratchReg>> index;
    u64 scale;
    u32 idx_size_bits;
    i64 displacement;
  };

  struct IntBinaryOp {
  private:
    static constexpr u32 index_mask = (1 << 4) - 1;
    static constexpr u32 bit_symm = 1 << 4;
    static constexpr u32 bit_signed = 1 << 5;
    static constexpr u32 bit_ext_lhs = 1 << 6;
    static constexpr u32 bit_ext_rhs = 1 << 7;
    static constexpr u32 bit_div = 1 << 8;
    static constexpr u32 bit_rem = 1 << 9;
    static constexpr u32 bit_shift = 1 << 10;

  public:
    enum Value : u32 {
      add = 0 | bit_symm,
      sub = 1,
      mul = 2 | bit_symm,
      udiv = 3 | bit_ext_lhs | bit_ext_rhs | bit_div,
      sdiv = 4 | bit_signed | bit_ext_lhs | bit_ext_rhs | bit_div,
      urem = 5 | bit_ext_lhs | bit_ext_rhs | bit_rem,
      srem = 6 | bit_signed | bit_ext_lhs | bit_ext_rhs | bit_rem,
      land = 7 | bit_symm,
      lor = 8 | bit_symm,
      lxor = 9 | bit_symm,
      shl = 10 | bit_shift,
      shr = 11 | bit_ext_lhs | bit_shift,
      ashr = 12 | bit_signed | bit_ext_lhs | bit_shift,
      num_ops = 13
    };

    Value op;

    constexpr IntBinaryOp(Value op) noexcept : op(op) {}

    /// Whether the operation is symmetric.
    constexpr bool is_symmetric() const noexcept { return op & bit_symm; }
    /// Whether the operation is signed and therefore needs sign-extension.
    constexpr bool is_signed() const noexcept { return op & bit_signed; }
    /// Whether the operation needs the first operand extended.
    constexpr bool needs_lhs_ext() const noexcept { return op & bit_ext_lhs; }
    /// Whether the operation needs the second operand extended.
    constexpr bool needs_rhs_ext() const noexcept { return op & bit_ext_rhs; }
    /// Whether the operation is a div
    constexpr bool is_div() const noexcept { return op & bit_div; }
    /// Whether the operation is a rem
    constexpr bool is_rem() const noexcept { return op & bit_rem; }
    /// Whether the operation is a shift
    constexpr bool is_shift() const noexcept { return op & bit_shift; }

    constexpr unsigned index() const noexcept { return op & index_mask; }

    bool operator==(const IntBinaryOp &o) const noexcept { return op == o.op; }
  };

  struct FloatBinaryOp {
    enum {
      add,
      sub,
      mul,
      div,
      rem
    };
  };

  enum class OverflowOp {
    uadd,
    sadd,
    usub,
    ssub,
    umul,
    smul
  };

  tpde::util::BumpAllocator<> const_allocator;

  /// Set of all symbols referenced by llvm.used.
  llvm::SmallPtrSet<const llvm::GlobalObject *, 2> used_globals;

  llvm::DenseMap<const llvm::GlobalValue *, SymRef> global_syms;
  /// Map from LLVM Comdat to the corresponding group section.
  llvm::DenseMap<const llvm::Comdat *, SecRef> group_secs;

  tpde::util::SmallVector<std::pair<IRValueRef, SymRef>, 16> type_info_syms;

  enum class LibFunc {
    divti3,
    udivti3,
    modti3,
    umodti3,
    fmod,
    fmodf,
    floorf,
    floor,
    ceilf,
    ceil,
    roundf,
    round,
    rintf,
    rint,
    memcpy,
    memset,
    memmove,
    resume,
    powisf2,
    powidf2,
    trunc,
    truncf,
    pow,
    powf,
    sin,
    sinf,
    cos,
    cosf,
    log,
    logf,
    log10,
    log10f,
    exp,
    expf,
    trunctfsf2,
    trunctfdf2,
    extendsftf2,
    extenddftf2,
    eqtf2,
    netf2,
    gttf2,
    getf2,
    lttf2,
    letf2,
    unordtf2,
    floatsitf,
    floatditf,
    floatunditf,
    floatunsitf,
    fixtfdi,
    fixunstfdi,
    addtf3,
    subtf3,
    multf3,
    divtf3,
    MAX
  };
  std::array<SymRef, static_cast<size_t>(LibFunc::MAX)> libfunc_syms;

  llvm::TimeTraceProfilerEntry *time_entry;

  LLVMCompilerBase(LLVMAdaptor *adaptor) : Base{adaptor} {
    static_assert(tpde::Compiler<Derived, Config>);
    static_assert(std::is_same_v<Adaptor, LLVMAdaptor>);
    libfunc_syms.fill({});
  }

  Derived *derived() noexcept { return static_cast<Derived *>(this); }

  const Derived *derived() const noexcept {
    return static_cast<Derived *>(this);
  }

  // TODO(ts): check if it helps to check this
  static bool cur_func_may_emit_calls() noexcept { return true; }

  SymRef cur_personality_func() const noexcept;

  static bool try_force_fixed_assignment(IRValueRef) noexcept { return false; }

  void analysis_start() noexcept;
  void analysis_end() noexcept;

  LLVMAdaptor::ValueParts val_parts(IRValueRef val) const noexcept {
    return this->adaptor->val_parts(val);
  }

  ValuePart val_ref_constant(const llvm::Constant *, u32 part) noexcept;

  std::optional<ValRefSpecial> val_ref_special(IRValueRef value) noexcept {
    if (llvm::isa<llvm::Constant>(value)) {
      return ValRefSpecial{.value = value};
    }
    return std::nullopt;
  }

  ValuePart val_part_ref_special(ValRefSpecial &vrs, u32 part) noexcept {
    return val_ref_constant(llvm::cast<llvm::Constant>(vrs.value), part);
  }

  ValueRef result_ref(const llvm::Value *v) noexcept {
    assert((llvm::isa<llvm::Argument, llvm::PHINode>(v)));
    // For arguments, phis nodes
    return Base::result_ref(v);
  }

  /// Specialized for llvm::Instruction to avoid type check in val_local_idx.
  ValueRef result_ref(const llvm::Instruction *i) noexcept {
    const auto local_idx =
        static_cast<tpde::ValLocalIdx>(this->adaptor->inst_lookup_idx(i));
    if (this->val_assignment(local_idx) == nullptr) {
      this->init_assignment(i, local_idx);
    }
    return ValueRef{this, local_idx};
  }

  std::pair<ValueRef, ValuePartRef>
      result_ref_single(const llvm::Value *v) noexcept {
    assert(llvm::isa<llvm::Argument>(v));
    // For byval arguments
    return Base::result_ref_single(v);
  }

  /// Specialized for llvm::Instruction to avoid type check in val_local_idx.
  std::pair<ValueRef, ValuePartRef>
      result_ref_single(const llvm::Instruction *i) noexcept {
    std::pair<ValueRef, ValuePartRef> res{result_ref(i), this};
    res.second = res.first.part(0);
    return res;
  }

  void prologue_assign_arg(tpde::CCAssigner *cc_assigner,
                           u32 arg_idx,
                           IRValueRef arg) noexcept {
    u32 align = arg->getType()->isIntegerTy(128) ? 16 : 1;
    bool allow_split = derived()->arg_allow_split_reg_stack_passing(arg);
    Base::prologue_assign_arg(cc_assigner, arg_idx, arg, align, allow_split);
  }

private:
  static tpde::Assembler::SymBinding
      convert_linkage(llvm::GlobalValue::LinkageTypes linkage) noexcept {
    if (llvm::GlobalValue::isLocalLinkage(linkage)) {
      return tpde::Assembler::SymBinding::LOCAL;
    } else if (llvm::GlobalValue::isWeakForLinker(linkage)) {
      return tpde::Assembler::SymBinding::WEAK;
    }
    return tpde::Assembler::SymBinding::GLOBAL;
  }

  static tpde::elf::AssemblerElf::SymVisibility
      convert_visibility(const llvm::GlobalValue *gv) noexcept {
    switch (gv->getVisibility()) {
    case llvm::GlobalValue::DefaultVisibility:
      return tpde::elf::AssemblerElf::SymVisibility::DEFAULT;
    case llvm::GlobalValue::HiddenVisibility:
      return tpde::elf::AssemblerElf::SymVisibility::HIDDEN;
    case llvm::GlobalValue::ProtectedVisibility:
      return tpde::elf::AssemblerElf::SymVisibility::PROTECTED;
    default: TPDE_UNREACHABLE("invalid global visibility");
    }
  }

public:
  /// Whether to use a DSO-local access instead of going through the GOT.
  static bool use_local_access(const llvm::GlobalValue *gv) noexcept {
    // If the symbol is preemptible, don't generate a local access.
    if (!gv->isDSOLocal()) {
      return false;
    }

    // Symbol be undefined, hence cannot use relative addressing.
    if (gv->hasExternalWeakLinkage()) {
      return false;
    }

    // If the symbol would need a local alias symbol (LLVM generates an extra
    // .L<sym>$local symbol), we would actually be able to generate a local
    // access through a private symbol (i.e., a section-relative relocation in
    // the object file). We don't support this right now, as it would require
    // fixing up symbols and converting them into relocations if required.
    // TODO: support local aliases for default-visibility dso_local definitions.
    if (gv->canBenefitFromLocalAlias()) {
      return false;
    }

    return true;
  }

  void define_func_idx(IRFuncRef func, const u32 idx) noexcept;

  /// Get comdat section group. sym_hint, if present, is the symbol associated
  /// with go to avoid an extra lookup.
  SecRef get_group_section(const llvm::GlobalObject *go,
                           SymRef sym_hint = {}) noexcept;

  /// Select section for a global. (and create if needed)
  SecRef select_section(SymRef sym,
                        const llvm::GlobalObject *go,
                        bool needs_relocs) noexcept;

  bool hook_post_func_sym_init() noexcept;
  [[nodiscard]] bool
      global_init_to_data(const llvm::Value *reloc_base,
                          tpde::util::SmallVector<u8, 64> &data,
                          tpde::util::SmallVector<RelocInfo, 8> &relocs,
                          const llvm::DataLayout &layout,
                          const llvm::Constant *constant,
                          u32 off) noexcept;

  SymRef get_libfunc_sym(LibFunc func) noexcept;

  SymRef global_sym(const llvm::GlobalValue *global) const noexcept {
    SymRef res = global_syms.lookup(global);
    assert(res.valid());
    return res;
  }

  void setup_var_ref_assignments() noexcept {}

  bool compile_func(IRFuncRef func, u32 idx) noexcept {
    time_entry = nullptr;

    // Reuse/release memory for stored constants from previous function
    const_allocator.reset();

    SecRef sec = this->select_section(this->func_syms[idx], func, true);
    if (!sec.valid()) [[unlikely]] {
      TPDE_LOG_ERR("unable to determine section for function {}",
                   std::string_view(func->getName()));
      return false;
    }
    if (this->text_writer.get_sec_ref() != sec) {
      this->text_writer.flush();
      this->text_writer.switch_section(this->assembler.get_section(sec));
    }

    // We might encounter types that are unsupported during compilation, which
    // cause the flag in the adaptor to be set. In such cases, return false.
    const bool res =
        (Base::compile_func(func, idx) && !this->adaptor->func_unsupported);

    // end the TPDE_CodeGen time trace entry
    if (time_entry) {
      llvm::timeTraceProfilerEnd(time_entry);
      time_entry = nullptr;
    }
    return res;
  }

  bool compile(llvm::Module &mod) noexcept;

  bool compile_unknown(const llvm::Instruction *,
                       const ValInfo &,
                       u64) noexcept {
    return false;
  }

  bool compile_inst(const llvm::Instruction *, InstRange) noexcept;

  bool compile_unreachable(const llvm::Instruction *,
                           const ValInfo &,
                           u64) noexcept;
  bool compile_ret(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_load_generic(const llvm::LoadInst *,
                            GenericValuePart &&) noexcept;
  bool compile_load(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_store_generic(const llvm::StoreInst *,
                             GenericValuePart &&) noexcept;
  bool compile_store(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_int_binary_op_i128(const llvm::Instruction *,
                                  const ValInfo &,
                                  IntBinaryOp op) noexcept;
  bool compile_int_binary_op(const llvm::Instruction *,
                             const ValInfo &,
                             u64) noexcept;
  bool compile_float_binary_op(const llvm::Instruction *,
                               const ValInfo &,
                               u64) noexcept;
  bool compile_fneg(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_float_ext_trunc(const llvm::Instruction *,
                               const ValInfo &,
                               u64) noexcept;
  bool compile_float_to_int(const llvm::Instruction *,
                            const ValInfo &,
                            u64) noexcept;
  bool compile_int_to_float(const llvm::Instruction *,
                            const ValInfo &,
                            u64) noexcept;
  bool compile_int_trunc(const llvm::Instruction *,
                         const ValInfo &,
                         u64) noexcept;
  bool
      compile_int_ext(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_ptr_to_int(const llvm::Instruction *,
                          const ValInfo &,
                          u64) noexcept;
  bool compile_int_to_ptr(const llvm::Instruction *,
                          const ValInfo &,
                          u64) noexcept;
  bool
      compile_bitcast(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_extract_value(const llvm::Instruction *,
                             const ValInfo &,
                             u64) noexcept;
  bool compile_insert_value(const llvm::Instruction *,
                            const ValInfo &,
                            u64) noexcept;

  void extract_element(ValueRef &vec_vr,
                       unsigned idx,
                       LLVMBasicValType ty,
                       ValuePart &out) noexcept;
  void insert_element(ValueRef &vec_vr,
                      unsigned idx,
                      LLVMBasicValType ty,
                      GenericValuePart &&el) noexcept;
  bool compile_extract_element(const llvm::Instruction *,
                               const ValInfo &,
                               u64) noexcept;
  bool compile_insert_element(const llvm::Instruction *,
                              const ValInfo &,
                              u64) noexcept;
  bool compile_shuffle_vector(const llvm::Instruction *,
                              const ValInfo &,
                              u64) noexcept;
  bool compile_icmp_vector(const llvm::Instruction *,
                           const ValInfo &,
                           u64) noexcept;

  bool
      compile_cmpxchg(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_atomicrmw(const llvm::Instruction *,
                         const ValInfo &,
                         u64) noexcept;
  bool compile_fence(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_freeze(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_call(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_select(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_alloca(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_gep(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_fcmp(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_switch(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_invoke(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_landing_pad(const llvm::Instruction *,
                           const ValInfo &,
                           u64) noexcept;
  bool compile_resume(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  SymRef lookup_type_info_sym(const llvm::GlobalValue *value) noexcept;
  bool compile_intrin(const llvm::IntrinsicInst *, const ValInfo &) noexcept;
  bool compile_is_fpclass(const llvm::IntrinsicInst *) noexcept;
  bool compile_overflow_intrin(const llvm::IntrinsicInst *,
                               OverflowOp) noexcept;
  bool compile_saturating_intrin(const llvm::IntrinsicInst *,
                                 OverflowOp) noexcept;
  bool compile_vector_reduce(const llvm::IntrinsicInst *,
                             const ValInfo &) noexcept;


  bool compile_br(const llvm::Instruction *, const ValInfo &, u64) noexcept {
    return false;
  }

  bool compile_inline_asm(const llvm::CallBase *) { return false; }

  bool handle_intrin(const llvm::IntrinsicInst *) noexcept { return false; }

  bool compile_to_elf(llvm::Module &mod,
                      std::vector<uint8_t> &buf) noexcept override;

  JITMapper compile_and_map(
      llvm::Module &mod,
      std::function<void *(std::string_view)> resolver) noexcept override;
};

template <typename Adaptor, typename Derived, typename Config>
typename LLVMCompilerBase<Adaptor, Derived, Config>::SymRef
    LLVMCompilerBase<Adaptor, Derived, Config>::cur_personality_func()
        const noexcept {
  if (!this->adaptor->cur_func->hasPersonalityFn()) {
    return SymRef();
  }

  llvm::Constant *p = this->adaptor->cur_func->getPersonalityFn();
  if (auto *gv = llvm::dyn_cast<llvm::GlobalValue>(p)) [[likely]] {
    assert(global_syms.contains(gv));
    return global_syms.lookup(gv);
  }

  TPDE_LOG_ERR("non-GlobalValue personality function unsupported");
  this->adaptor->func_unsupported = true;
  return SymRef();
}

template <typename Adaptor, typename Derived, typename Config>
void LLVMCompilerBase<Adaptor, Derived, Config>::analysis_start() noexcept {
  if (llvm::timeTraceProfilerEnabled()) {
    time_entry = llvm::timeTraceProfilerBegin("TPDE_Analysis", "");
  }
}

template <typename Adaptor, typename Derived, typename Config>
void LLVMCompilerBase<Adaptor, Derived, Config>::analysis_end() noexcept {
  if (time_entry) {
    llvm::timeTraceProfilerEnd(time_entry);
    time_entry = llvm::timeTraceProfilerBegin("TPDE_CodeGen", "");
  }
}

template <typename Adaptor, typename Derived, typename Config>
typename LLVMCompilerBase<Adaptor, Derived, Config>::ValuePart
    LLVMCompilerBase<Adaptor, Derived, Config>::val_ref_constant(
        const llvm::Constant *const_val, u32 part) noexcept {
  auto [ty, ty_idx] = this->adaptor->lower_type(const_val->getType());
  unsigned sub_part = part;

  if (ty == LLVMBasicValType::complex) {
    LLVMComplexPart *part_descs =
        &this->adaptor->complex_part_types[ty_idx + 1];

    // Iterate over complex data type to find the struct/array indices that
    // belong to part.
    sub_part = 0;
    tpde::util::SmallVector<unsigned, 16> indices;
    for (unsigned i = 0; i < part; i++) {
      indices.resize(indices.size() + part_descs[i].part.nest_inc -
                     part_descs[i].part.nest_dec);
      if (part_descs[i].part.ends_value) {
        sub_part = 0;
        if (!indices.empty()) {
          indices.back()++;
        }
      } else {
        sub_part++;
      }
    }
    indices.resize(indices.size() + part_descs[part].part.nest_inc);

    for (unsigned idx : indices) {
      if (!const_val) {
        break;
      }
      if (auto *cda = llvm::dyn_cast<llvm::ConstantDataArray>(const_val)) {
        const_val = cda->getElementAsConstant(idx);
        break;
      }
      auto *agg = llvm::dyn_cast<llvm::ConstantAggregate>(const_val);
      if (!agg) {
        break;
      }
      const_val = llvm::cast<llvm::Constant>(agg->getOperand(idx));
    }

    ty = part_descs[part].part.type;
    assert(ty != LLVMBasicValType::invalid);
  } else if (ty == LLVMBasicValType::invalid) [[unlikely]] {
    // Invalid types were already reported. Just try to keep going.
    return ValuePart(u64{0}, 1, Config::GP_BANK);
  }

  // At this point, ty is the basic type of the element and sub_part the part
  // inside the basic type.

  if (auto *gv = llvm::dyn_cast<llvm::GlobalValue>(const_val)) {
    assert(ty == LLVMBasicValType::ptr && sub_part == 0);
    u32 gv_id = this->adaptor->global_lookup.lookup(gv);
    auto local_idx = tpde::ValLocalIdx(this->adaptor->values.size() + gv_id);
    auto *assignment = this->val_assignment(local_idx);
    if (!assignment) {
      this->init_variable_ref(local_idx, gv_id);
      assignment = this->val_assignment(local_idx);
    }
    return ValuePart{local_idx, assignment, 0, /*owned=*/false};
  }

  u32 size = this->adaptor->basic_ty_part_size(ty);
  tpde::RegBank bank = this->adaptor->basic_ty_part_bank(ty);

  if (llvm::isa<llvm::PoisonValue>(const_val) ||
      llvm::isa<llvm::UndefValue>(const_val) ||
      llvm::isa<llvm::ConstantPointerNull>(const_val) ||
      llvm::isa<llvm::ConstantAggregateZero>(const_val)) {
    static const std::array<u64, 8> zero{};
    assert(size <= zero.size() * sizeof(u64));
    return ValuePart(zero.data(), size, bank);
  }

  if (auto *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(const_val)) {
    llvm::StringRef data = cdv->getRawDataValues();
    assert((sub_part + 1) * size <= data.size());
    // TODO: use data.data() to avoid copying if possible?
    u64 *copy = new (const_allocator) u64[(size + 7) / 8];
    if (size < 8) {
      *copy = 0; // zero-initialize
    }
    std::memcpy(copy, data.data() + sub_part * size, size);
    return ValuePart(copy, size, bank);
  }

  if (llvm::isa<llvm::ConstantVector>(const_val)) {
    if (const_val->getType()->getScalarType()->isIntegerTy(1)) {
      assert(const_val->getNumOperands() <= 64 &&
             "i1-vector with more than 64 elements should not be legal");
      u64 val = 0;
      for (auto it : llvm::enumerate(const_val->operands())) {
        if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(it.value())) {
          val |= u64{ci->isOne()} << it.index();
        } else {
          // All other constant types should've been converted to insertelement.
          assert((llvm::isa<llvm::UndefValue, llvm::PoisonValue>(it.value())));
        }
      }
      return ValuePart(val, size, bank);
    }

    // TODO(ts): check how to handle this
    TPDE_FATAL("non-sequential vector constants should not be legal");
  }

  if (const auto *const_int = llvm::dyn_cast<llvm::ConstantInt>(const_val)) {
    assert(sub_part < (ty == LLVMBasicValType::i128 ? 2 : 1));
    assert(size <= 8 && "multi-word integer as single part?");
    const u64 *data = const_int->getValue().getRawData();
    return ValuePart(data[sub_part], size, bank);
  }

  if (const auto *const_fp = llvm::dyn_cast<llvm::ConstantFP>(const_val);
      const_fp != nullptr) {
    // APFloat has no bitwise storage of the floating-point number and
    // bitcastToAPInt constructs a new APInt, so we need to copy the value.
    llvm::APInt int_val = const_fp->getValue().bitcastToAPInt();
    u64 *data = new (const_allocator) u64[int_val.getNumWords()];
    std::memcpy(
        data, int_val.getRawData(), int_val.getNumWords() * sizeof(u64));

    assert(size <= int_val.getNumWords() * sizeof(u64));
    return ValuePart(data, size, bank);
  }

  std::string const_str;
  llvm::raw_string_ostream(const_str) << *const_val;
  TPDE_LOG_ERR("unhandled constant in operand: {}", const_str);
  this->adaptor->func_unsupported = true;

  // Try to keep going with a null constant.
  static const std::array<u64, 8> zero{};
  assert(size <= zero.size() * sizeof(u64));
  return ValuePart(zero.data(), size, bank);
}

template <typename Adaptor, typename Derived, typename Config>
void LLVMCompilerBase<Adaptor, Derived, Config>::define_func_idx(
    IRFuncRef func, const u32 idx) noexcept {
  SymRef fn_sym = this->func_syms[idx];
  global_syms[func] = fn_sym;
  if (!func->hasDefaultVisibility()) {
    this->assembler.sym_set_visibility(fn_sym, convert_visibility(func));
  }
}

template <typename Adaptor, typename Derived, typename Config>
LLVMCompilerBase<Adaptor, Derived, Config>::SecRef
    LLVMCompilerBase<Adaptor, Derived, Config>::get_group_section(
        const llvm::GlobalObject *go, SymRef sym_hint) noexcept {
  const llvm::Comdat *comdat = go->getComdat();
  if (!comdat) {
    return SecRef();
  }

  bool is_comdat;
  switch (comdat->getSelectionKind()) {
  case llvm::Comdat::Any: is_comdat = true; break;
  case llvm::Comdat::NoDeduplicate: is_comdat = false; break;
  default:
    // ELF only support any/nodeduplicate.
    return SecRef();
  }

  auto [it, inserted] = this->group_secs.try_emplace(comdat);
  if (inserted) {
    // We need to find or create the group signature symbol. Typically, this
    // is the same as the name of the global.
    SymRef group_sym;
    bool define_group_sym = false;
    if (llvm::StringRef cn = comdat->getName();
        sym_hint.valid() && go->getName() == cn) {
      group_sym = sym_hint;
    } else if (auto *cgv = this->adaptor->mod->getNamedValue(cn)) {
      // In this case, we need to search for or create a symbol with the
      // comdat name. As we don't have a symbol string map, we do this
      // through the Module's map to find the matching global and map this to
      // the symbol.
      // TODO: name mangling might make this impossible: the names of globals
      // are mangled, but comdat names are not.
      group_sym = global_sym(cgv);
    } else {
      // Create a new symbol if no equally named global, thus symbol, exists.
      // The symbol will be STB_LOCAL, STT_NOTYPE, section=group.
      group_sym =
          this->assembler.sym_add_undef(cn, tpde::Assembler::SymBinding::LOCAL);
      define_group_sym = true;
    }
    it->second = this->assembler.create_group_section(group_sym, is_comdat);
    if (define_group_sym) {
      this->assembler.sym_def(group_sym, it->second, 0, 0);
    }
  }

  return it->second;
}

template <typename Adaptor, typename Derived, typename Config>
LLVMCompilerBase<Adaptor, Derived, Config>::SecRef
    LLVMCompilerBase<Adaptor, Derived, Config>::select_section(
        SymRef sym, const llvm::GlobalObject *go, bool needs_relocs) noexcept {
  // TODO: factor this out into platform-specific code.

  // TODO: support ifuncs
  if (llvm::isa<llvm::GlobalIFunc>(go)) {
    return SecRef();
  }

  // I'm certain this simplified section assignment code is buggy...
  using tpde::SectionKind;
  SectionKind kind;
  if (llvm::isa<llvm::Function>(go)) {
    kind = SectionKind::Text;
  } else {
    auto gv = llvm::cast<llvm::GlobalVariable>(go);
    bool init_zero = gv->getInitializer()->isNullValue();
    bool read_only = gv->isConstant();
    if (gv->isThreadLocal()) {
      kind = init_zero ? SectionKind::ThreadBSS : SectionKind::ThreadData;
    } else if (!read_only && init_zero) {
      assert(!needs_relocs && "BSS section must not have relocations");
      kind = SectionKind::BSS;
    } else if (read_only) {
      kind = needs_relocs ? SectionKind::DataRelRO : SectionKind::ReadOnly;
    } else {
      kind = SectionKind::Data;
    }
  }

  bool retain = used_globals.contains(go);
  llvm::StringRef sec_name = go->getSection();
  const llvm::Comdat *comdat = go->getComdat();

  // If the section name is empty, use the default section.
  if (!retain && !comdat && sec_name.empty()) [[likely]] {
    return this->assembler.get_default_section(kind);
  }

  // Group section must be created before the group contents.
  SecRef group_sec = get_group_section(go, sym);

  // TODO: is it *required* that we merge sections here? For now, don't.
  SecRef sec = this->assembler.create_section(kind);
  if (!sec_name.empty()) {
    this->assembler.rename_section(sec, sec_name);
  }

  if (group_sec.valid()) {
    // TODO: ELF only
    this->assembler.add_to_group(group_sec, sec);
  }

  if (retain) {
    // TODO: ELF only
    this->assembler.get_section(sec).flags |= tpde::elf::SHF_GNU_RETAIN;
  }

  return sec;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::
    hook_post_func_sym_init() noexcept {
  llvm::TimeTraceScope time_scope("TPDE_GlobalGen");

  // create global symbols and their definitions
  const auto &llvm_mod = *this->adaptor->mod;
  auto &data_layout = llvm_mod.getDataLayout();

  global_syms.reserve(2 * llvm_mod.global_size());

  auto declare_global = [&, this](const llvm::GlobalValue &gv) {
    llvm::GlobalValue::LinkageTypes linkage = gv.getLinkage();
    if (!gv.hasName() && !gv.hasLocalLinkage()) [[unlikely]] {
      // This is unspecified by LangRef at least up to LLVM 21. LLVM mangles
      // the name into "__unnamed_<number>".
      TPDE_LOG_WARN("unnamed global converted to internal linkage");
      linkage = llvm::GlobalValue::InternalLinkage;
    }

    // TODO: name mangling
    llvm::StringRef name = gv.getName();

    if (linkage == llvm::GlobalValue::AppendingLinkage) [[unlikely]] {
      if (name == "llvm.used") {
        auto init = llvm::cast<llvm::GlobalVariable>(gv).getInitializer();
        if (auto used_array = llvm::cast_or_null<llvm::ConstantArray>(init)) {
          for (const auto &op : used_array->operands()) {
            if (const auto *go = llvm::dyn_cast<llvm::GlobalObject>(op)) {
              used_globals.insert(go);
            }
          }
        }
      }

      if (name != "llvm.global_ctors" && name != "llvm.global_dtors" &&
          name != "llvm.used" && name != "llvm.compiler.used") {
        TPDE_LOG_ERR("Unknown global with appending linkage: {}\n",
                     static_cast<std::string_view>(name));
        return false;
      }
      return true;
    }

    auto binding = convert_linkage(linkage);
    SymRef sym;
    if (gv.isThreadLocal()) {
      sym = this->assembler.sym_predef_tls(name, binding);
    } else if (!gv.isDeclarationForLinker()) {
      sym = this->assembler.sym_predef_data(name, binding);
    } else {
      sym = this->assembler.sym_add_undef(name, binding);
    }
    global_syms[&gv] = sym;
    if (!gv.hasDefaultVisibility()) {
      this->assembler.sym_set_visibility(sym, convert_visibility(&gv));
    }
    return true;
  };

  // create the symbols first so that later relocations don't try to look up
  // non-existent symbols
  for (const llvm::GlobalVariable &gv : llvm_mod.globals()) {
    if (!declare_global(gv)) {
      return false;
    }
  }

  for (const llvm::GlobalAlias &ga : llvm_mod.aliases()) {
    if (!declare_global(ga)) {
      return false;
    }
  }

  if (!llvm_mod.ifunc_empty()) {
    TPDE_LOG_ERR("ifuncs are not supported");
    return false;
  }

  // since the adaptor exposes all functions in the module to TPDE,
  // all function symbols are already added

  // now we can initialize the global data
  tpde::util::SmallVector<u8, 64> data;
  tpde::util::SmallVector<RelocInfo, 8> relocs;
  for (auto it = llvm_mod.global_begin(); it != llvm_mod.global_end(); ++it) {
    auto *gv = &*it;
    if (gv->isDeclarationForLinker()) {
      continue;
    }

    if (gv->getMetadata(llvm::LLVMContext::MD_associated)) {
      // Rarely needed, only supported on ELF. The language reference also
      // mentions that linker support is "spotty".
      TPDE_LOG_ERR("!associated is not implemented");
      return false;
    }

    auto *init = gv->getInitializer();
    if (gv->hasAppendingLinkage()) [[unlikely]] {
      llvm::StringRef name = gv->getName();
      if (name == "llvm.used" || name == "llvm.compiler.used") {
        // llvm.used is collected above and handled by select_section.
        // llvm.compiler.used needs no special handling.
        continue;
      }
      assert(name == "llvm.global_ctors" || name == "llvm.global_dtors");
      if (llvm::isa<llvm::ConstantAggregateZero>(init)) {
        continue;
      }

      struct Structor {
        SymRef func;
        SecRef group;
        unsigned priority;

        bool operator<(const Structor &rhs) const noexcept {
          return std::pair(group.id(), priority) <
                 std::pair(rhs.group.id(), rhs.priority);
        }
      };
      tpde::util::SmallVector<Structor, 16> structors;

      // see
      // https://llvm.org/docs/LangRef.html#the-llvm-global-ctors-global-variable
      for (auto &entry : llvm::cast<llvm::ConstantArray>(init)->operands()) {
        const auto *str = llvm::cast<llvm::ConstantStruct>(entry);
        auto *prio = llvm::cast<llvm::ConstantInt>(str->getOperand(0));
        auto *ptr = llvm::cast<llvm::GlobalValue>(str->getOperand(1));
        SecRef group = SecRef();
        if (auto *comdat = str->getOperand(2); !comdat->isNullValue()) {
          comdat = comdat->stripPointerCasts();
          if (auto *comdat_gv = llvm::dyn_cast<llvm::GlobalObject>(comdat)) {
            if (comdat_gv->isDeclarationForLinker()) {
              // Cf. AsmPrinter::emitXXStructorList
              continue;
            }
            group = get_group_section(comdat_gv);
          } else {
            TPDE_LOG_ERR("non-GlobalObject ctor/dtor comdat not implemented");
            return false;
          }
        }
        unsigned prio_val = prio->getLimitedValue(65535);
        if (prio_val != 65535) {
          TPDE_LOG_ERR("ctor/dtor priorities not implemented");
          return false;
        }
        structors.emplace_back(global_sym(ptr), group, prio_val);
      }

      const auto is_ctor = (name == "llvm.global_ctors");

      // We need to create one array section per comdat group per priority.
      // Therefore, sort so that structors for the same section are together.
      std::sort(structors.begin(), structors.end());

      SecRef secref = SecRef();
      for (size_t i = 0; i < structors.size(); ++i) {
        const auto &s = structors[i];
        if (i == 0 || structors[i - 1] < s) {
          secref = this->assembler.create_structor_section(is_ctor, s.group);
        }
        auto &sec = this->assembler.get_section(secref);
        sec.data.resize(sec.data.size() + 8);
        this->assembler.reloc_abs(secref, s.func, sec.data.size() - 8, 0);
      }
      continue;
    }

    auto size = data_layout.getTypeAllocSize(init->getType());
    auto align = gv->getAlign().valueOrOne().value();
    bool is_zero = init->isNullValue();
    auto sym = global_sym(gv);

    data.clear();
    relocs.clear();
    if (!is_zero) {
      data.resize(size);
      if (!global_init_to_data(gv, data, relocs, data_layout, init, 0)) {
        return false;
      }
    }

    SecRef sec = this->select_section(sym, gv, !relocs.empty());
    if (!sec.valid()) [[unlikely]] {
      std::string global_str;
      llvm::raw_string_ostream(global_str) << *gv;
      TPDE_LOG_ERR("unable to determine section for global {}", global_str);
      return false;
    }

    if (is_zero) {
      this->assembler.sym_def_predef_zero(sec, sym, size, align);
      continue;
    }

    u32 off;
    this->assembler.sym_def_predef_data(sec, sym, data, align, &off);
    for (auto &[inner_off, addend, target, type] : relocs) {
      if (type == RelocInfo::RELOC_ABS) {
        this->assembler.reloc_abs(sec, target, off + inner_off, addend);
      } else {
        assert(type == RelocInfo::RELOC_PC32);
        this->assembler.reloc_pc32(sec, target, off + inner_off, addend);
      }
    }
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::global_init_to_data(
    const llvm::Value *reloc_base,
    tpde::util::SmallVector<u8, 64> &data,
    tpde::util::SmallVector<RelocInfo, 8> &relocs,
    const llvm::DataLayout &layout,
    const llvm::Constant *constant,
    u32 off) noexcept {
  // Handle all-zero values quickly.
  if (constant->isNullValue() || llvm::isa<llvm::UndefValue>(constant)) {
    return true;
  }

  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(constant); CI) {
    // TODO: endianness?
    unsigned store_size = (CI->getValue().getBitWidth() + 7) / 8;
    llvm::StoreIntToMemory(CI->getValue(), data.data() + off, store_size);
    return true;
  }
  if (auto *CF = llvm::dyn_cast<llvm::ConstantFP>(constant); CF) {
    // TODO: endianness?
    llvm::APInt int_val = CF->getValue().bitcastToAPInt();
    unsigned store_size = (int_val.getBitWidth() + 7) / 8;
    llvm::StoreIntToMemory(int_val, data.data() + off, store_size);
    return true;
  }
  if (auto *CDS = llvm::dyn_cast<llvm::ConstantDataSequential>(constant); CDS) {
    llvm::copy(CDS->getRawDataValues(), data.begin() + off);
    return true;
  }
  if (auto *CA = llvm::dyn_cast<llvm::ConstantArray>(constant); CA) {
    auto elem_sz = layout.getTypeAllocSize(CA->getType()->getElementType());
    bool success = true;
    for (const llvm::Use &v : CA->operands()) {
      auto *cv = llvm::cast<llvm::Constant>(v);
      success &= global_init_to_data(reloc_base, data, relocs, layout, cv, off);
      off += elem_sz;
    }
    return success;
  }
  if (auto *cs = llvm::dyn_cast<llvm::ConstantStruct>(constant); cs) {
    const auto *struct_layout = layout.getStructLayout(cs->getType());
    llvm::ArrayRef<llvm::TypeSize> offsets = struct_layout->getMemberOffsets();
    bool success = true;
    for (auto [moff, v] : llvm::zip_equal(offsets, cs->operands())) {
      auto *cv = llvm::cast<llvm::Constant>(v);
      success &=
          global_init_to_data(reloc_base, data, relocs, layout, cv, off + moff);
    }
    return success;
  }
  if (auto *GV = llvm::dyn_cast<llvm::GlobalValue>(constant); GV) {
    relocs.push_back({off, 0, global_sym(GV)});
    return true;
  }

  if (auto *expr = llvm::dyn_cast<llvm::ConstantExpr>(constant)) {
    // idk about this design, currently just hardcoding stuff i see
    // in theory i think this needs a new data buffer so we can recursively call
    // parseConstIntoByteArray
    switch (expr->getOpcode()) {
    case llvm::Instruction::IntToPtr:
      if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(expr->getOperand(0))) {
        unsigned store_size = layout.getTypeStoreSize(expr->getType());
        unsigned int_size = (CI->getValue().getBitWidth() + 7) / 8;
        unsigned copy_size = std::min(store_size, int_size);
        // TODO: endianness?
        llvm::StoreIntToMemory(CI->getValue(), data.data() + off, copy_size);
        // If store_size > int_size, zero-fill. data is pre-initialized with
        // zeroes, so do nothing.
        return true;
      }
      break;
    case llvm::Instruction::PtrToInt:
      if (auto *gv = llvm::dyn_cast<llvm::GlobalValue>(expr->getOperand(0))) {
        if (expr->getType()->isIntegerTy(64)) {
          relocs.push_back({off, 0, global_sym(gv)});
          return true;
        }
      }
      break;
    case llvm::Instruction::GetElementPtr: {
      auto *gep = llvm::cast<llvm::GEPOperator>(expr);
      auto *ptr = gep->getPointerOperand();
      if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(ptr); GV) {
        auto indices = tpde::util::SmallVector<llvm::Value *, 8>{};
        for (auto &idx : gep->indices()) {
          indices.push_back(idx.get());
        }

        const auto ty_off = layout.getIndexedOffsetInType(
            gep->getSourceElementType(),
            llvm::ArrayRef{indices.data(), indices.size()});
        relocs.push_back({off, static_cast<int32_t>(ty_off), global_sym(GV)});

        return true;
      }
      break;
    }
    case llvm::Instruction::Trunc:
      // recognize a truncation pattern where we need to emit PC32 relocations
      // i32 trunc (i64 sub (i64 ptrtoint (ptr <someglobal> to i64), i64
      // ptrtoint (ptr <relocBase> to i64)))
      if (expr->getType()->isIntegerTy(32)) {
        if (auto *sub = llvm::dyn_cast<llvm::ConstantExpr>(expr->getOperand(0));
            sub && sub->getOpcode() == llvm::Instruction::Sub &&
            sub->getType()->isIntegerTy(64)) {
          auto *lhs = llvm::dyn_cast<llvm::ConstantExpr>(sub->getOperand(0));
          auto *rhs = llvm::dyn_cast<llvm::ConstantExpr>(sub->getOperand(1));
          if (lhs && rhs && lhs->getOpcode() == llvm::Instruction::PtrToInt &&
              rhs->getOpcode() == llvm::Instruction::PtrToInt) {
            if (rhs->getOperand(0) == reloc_base &&
                llvm::isa<llvm::GlobalVariable>(lhs->getOperand(0))) {
              auto ptr_sym =
                  global_sym(llvm::cast<llvm::GlobalValue>(lhs->getOperand(0)));

              relocs.push_back({off,
                                static_cast<int32_t>(off),
                                ptr_sym,
                                RelocInfo::RELOC_PC32});
              return true;
            }
          }
        }
      }
      break;
    case llvm::Instruction::BitCast: {
      if (expr->getType()->isPointerTy()) {
        auto *op = expr->getOperand(0);
        if (llvm::isa<llvm::GlobalValue>(op)) {
          auto ptr_sym = global_sym(llvm::cast<llvm::GlobalValue>(op));
          // emit absolute relocation
          relocs.push_back({off, 0, ptr_sym, RelocInfo::RELOC_ABS});
          return true;
        }
      }
    } break;
    default: break;
    }
  }

  // It's not a simple constant that we can handle, probably some ConstantExpr.
  // Try constant folding to increase the change that we can handle it. Some
  // front-ends like flang like to generate trivially foldable expressions.
  if (auto *fc = llvm::ConstantFoldConstant(constant, layout); constant != fc) {
    // We folded the constant, so try again.
    return global_init_to_data(reloc_base, data, relocs, layout, fc, off);
  }

  std::string const_str;
  llvm::raw_string_ostream(const_str) << *constant;
  TPDE_LOG_ERR("unhandled constant in global initializer: {}", const_str);
  return false;
}

template <typename Adaptor, typename Derived, typename Config>
typename LLVMCompilerBase<Adaptor, Derived, Config>::SymRef
    LLVMCompilerBase<Adaptor, Derived, Config>::get_libfunc_sym(
        LibFunc func) noexcept {
  assert(func < LibFunc::MAX);
  SymRef &sym = libfunc_syms[static_cast<size_t>(func)];
  if (sym.valid()) [[likely]] {
    return sym;
  }

  std::string_view name = "???";
  switch (func) {
  case LibFunc::divti3: name = "__divti3"; break;
  case LibFunc::udivti3: name = "__udivti3"; break;
  case LibFunc::modti3: name = "__modti3"; break;
  case LibFunc::umodti3: name = "__umodti3"; break;
  case LibFunc::fmod: name = "fmod"; break;
  case LibFunc::fmodf: name = "fmodf"; break;
  case LibFunc::floorf: name = "floorf"; break;
  case LibFunc::floor: name = "floor"; break;
  case LibFunc::ceilf: name = "ceilf"; break;
  case LibFunc::ceil: name = "ceil"; break;
  case LibFunc::roundf: name = "roundf"; break;
  case LibFunc::round: name = "round"; break;
  case LibFunc::rintf: name = "rintf"; break;
  case LibFunc::rint: name = "rint"; break;
  case LibFunc::memcpy: name = "memcpy"; break;
  case LibFunc::memset: name = "memset"; break;
  case LibFunc::memmove: name = "memmove"; break;
  case LibFunc::resume: name = "_Unwind_Resume"; break;
  case LibFunc::powisf2: name = "__powisf2"; break;
  case LibFunc::powidf2: name = "__powidf2"; break;
  case LibFunc::trunc: name = "trunc"; break;
  case LibFunc::truncf: name = "truncf"; break;
  case LibFunc::pow: name = "pow"; break;
  case LibFunc::powf: name = "powf"; break;
  case LibFunc::sin: name = "sin"; break;
  case LibFunc::sinf: name = "sinf"; break;
  case LibFunc::cos: name = "cos"; break;
  case LibFunc::cosf: name = "cosf"; break;
  case LibFunc::log: name = "log"; break;
  case LibFunc::logf: name = "logf"; break;
  case LibFunc::log10: name = "log10"; break;
  case LibFunc::log10f: name = "log10f"; break;
  case LibFunc::exp: name = "exp"; break;
  case LibFunc::expf: name = "expf"; break;
  case LibFunc::trunctfsf2: name = "__trunctfsf2"; break;
  case LibFunc::trunctfdf2: name = "__trunctfdf2"; break;
  case LibFunc::extendsftf2: name = "__extendsftf2"; break;
  case LibFunc::extenddftf2: name = "__extenddftf2"; break;
  case LibFunc::eqtf2: name = "__eqtf2"; break;
  case LibFunc::netf2: name = "__netf2"; break;
  case LibFunc::gttf2: name = "__gttf2"; break;
  case LibFunc::getf2: name = "__getf2"; break;
  case LibFunc::lttf2: name = "__lttf2"; break;
  case LibFunc::letf2: name = "__letf2"; break;
  case LibFunc::unordtf2: name = "__unordtf2"; break;
  case LibFunc::floatsitf: name = "__floatsitf"; break;
  case LibFunc::floatditf: name = "__floatditf"; break;
  case LibFunc::floatunsitf: name = "__floatunsitf"; break;
  case LibFunc::floatunditf: name = "__floatunditf"; break;
  case LibFunc::fixtfdi: name = "__fixtfdi"; break;
  case LibFunc::fixunstfdi: name = "__fixunstfdi"; break;
  case LibFunc::addtf3: name = "__addtf3"; break;
  case LibFunc::subtf3: name = "__subtf3"; break;
  case LibFunc::multf3: name = "__multf3"; break;
  case LibFunc::divtf3: name = "__divtf3"; break;
  default: TPDE_UNREACHABLE("invalid libfunc");
  }

  sym =
      this->assembler.sym_add_undef(name, tpde::Assembler::SymBinding::GLOBAL);
  return sym;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile(
    llvm::Module &mod) noexcept {
  this->adaptor->switch_module(mod);

  type_info_syms.clear();
  global_syms.clear();
  group_secs.clear();
  libfunc_syms.fill({});

  if (!Base::compile()) {
    return false;
  }

  // copy alias symbol definitions
  for (auto it = this->adaptor->mod->alias_begin();
       it != this->adaptor->mod->alias_end();
       ++it) {
    llvm::GlobalAlias *ga = &*it;
    auto *alias_target = llvm::dyn_cast<llvm::GlobalValue>(ga->getAliasee());
    if (alias_target == nullptr) {
      TPDE_LOG_ERR("alias with non-GlobalValue aliasee is unsupported");
      return false;
    }
    auto dst_sym = global_sym(ga);
    auto from_sym = global_sym(alias_target);

    this->assembler.sym_copy(dst_sym, from_sym);
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_inst(
    const llvm::Instruction *i, InstRange) noexcept {
  TPDE_LOG_TRACE("Compiling inst {}", this->adaptor->inst_fmt_ref(i));
  static constexpr auto fns = []() constexpr {
    // TODO: maybe don't use member-function pointers here, these are twice the
    // size of regular function pointers (hence an entry size is 0x18).
    using CompileFn = bool (Derived::*)(
        const llvm::Instruction *, const ValInfo &, u64) noexcept;
    std::array<std::pair<CompileFn, u64>, llvm::Instruction::OtherOpsEnd> res{};
    res.fill({&Derived::compile_unknown, 0});

    // clang-format off

    // Terminators
    res[llvm::Instruction::Ret] = {&Derived::compile_ret, 0};
    res[llvm::Instruction::Br] = {&Derived::compile_br, 0};
    res[llvm::Instruction::Switch] = {&Derived::compile_switch, 0};
    // TODO: IndirectBr
    res[llvm::Instruction::Invoke] = {&Derived::compile_invoke, 0};
    res[llvm::Instruction::Resume] = {&Derived::compile_resume, 0};
    res[llvm::Instruction::Unreachable] = {&Derived::compile_unreachable, 0};

    // Standard unary operators
    res[llvm::Instruction::FNeg] = {&Derived::compile_fneg, 0};

    // Standard binary operators
    res[llvm::Instruction::Add] = {&Derived::compile_int_binary_op, IntBinaryOp::add};
    res[llvm::Instruction::FAdd] = {&Derived::compile_float_binary_op, FloatBinaryOp::add};
    res[llvm::Instruction::Sub] = {&Derived::compile_int_binary_op, IntBinaryOp::sub};
    res[llvm::Instruction::FSub] = {&Derived::compile_float_binary_op, FloatBinaryOp::sub};
    res[llvm::Instruction::Mul] = {&Derived::compile_int_binary_op, IntBinaryOp::mul};
    res[llvm::Instruction::FMul] = {&Derived::compile_float_binary_op, FloatBinaryOp::mul};
    res[llvm::Instruction::UDiv] = {&Derived::compile_int_binary_op, IntBinaryOp::udiv};
    res[llvm::Instruction::SDiv] = {&Derived::compile_int_binary_op, IntBinaryOp::sdiv};
    res[llvm::Instruction::FDiv] = {&Derived::compile_float_binary_op, FloatBinaryOp::div};
    res[llvm::Instruction::URem] = {&Derived::compile_int_binary_op, IntBinaryOp::urem};
    res[llvm::Instruction::SRem] = {&Derived::compile_int_binary_op, IntBinaryOp::srem};
    res[llvm::Instruction::FRem] = {&Derived::compile_float_binary_op, FloatBinaryOp::rem};
    res[llvm::Instruction::Shl] = {&Derived::compile_int_binary_op, IntBinaryOp::shl};
    res[llvm::Instruction::LShr] = {&Derived::compile_int_binary_op, IntBinaryOp::shr};
    res[llvm::Instruction::AShr] = {&Derived::compile_int_binary_op, IntBinaryOp::ashr};
    res[llvm::Instruction::And] = {&Derived::compile_int_binary_op, IntBinaryOp::land};
    res[llvm::Instruction::Or] = {&Derived::compile_int_binary_op, IntBinaryOp::lor};
    res[llvm::Instruction::Xor] = {&Derived::compile_int_binary_op, IntBinaryOp::lxor};

    // Memory operators
    res[llvm::Instruction::Alloca] = {&Derived::compile_alloca, 0};
    res[llvm::Instruction::Load] = {&Derived::compile_load, 0};
    res[llvm::Instruction::Store] = {&Derived::compile_store, 0};
    res[llvm::Instruction::GetElementPtr] = {&Derived::compile_gep, 0};
    res[llvm::Instruction::Fence] = {&Derived::compile_fence, 0};
    res[llvm::Instruction::AtomicCmpXchg] = {&Derived::compile_cmpxchg, 0};
    res[llvm::Instruction::AtomicRMW] = {&Derived::compile_atomicrmw, 0};

    // Cast operators
    res[llvm::Instruction::Trunc] = {&Derived::compile_int_trunc, 0};
    res[llvm::Instruction::ZExt] = {&Derived::compile_int_ext, /*sign=*/false};
    res[llvm::Instruction::SExt] = {&Derived::compile_int_ext, /*sign=*/true};
    res[llvm::Instruction::FPToUI] = {&Derived::compile_float_to_int, /*flags=!sign,!sat*/0};
    res[llvm::Instruction::FPToSI] = {&Derived::compile_float_to_int, /*flags=sign,!sat*/1};
    res[llvm::Instruction::UIToFP] = {&Derived::compile_int_to_float, /*sign=*/false};
    res[llvm::Instruction::SIToFP] = {&Derived::compile_int_to_float, /*sign=*/true};
    res[llvm::Instruction::FPTrunc] = {&Derived::compile_float_ext_trunc, 0};
    res[llvm::Instruction::FPExt] = {&Derived::compile_float_ext_trunc, 0};
    res[llvm::Instruction::PtrToInt] = {&Derived::compile_ptr_to_int, 0};
    res[llvm::Instruction::IntToPtr] = {&Derived::compile_int_to_ptr, 0};
    res[llvm::Instruction::BitCast] = {&Derived::compile_bitcast, 0};
    // TODO: AddrSpaceCast

    // Other operators
    res[llvm::Instruction::ICmp] = {&Derived::compile_icmp, 0};
    res[llvm::Instruction::FCmp] = {&Derived::compile_fcmp, 0};
    // PHI will not be called
    res[llvm::Instruction::Call] = {&Derived::compile_call, 0};
    res[llvm::Instruction::Select] = {&Derived::compile_select, 0};
    res[llvm::Instruction::ExtractElement] = {&Derived::compile_extract_element, 0};
    res[llvm::Instruction::InsertElement] = {&Derived::compile_insert_element, 0};
    res[llvm::Instruction::ShuffleVector] = {&Derived::compile_shuffle_vector, 0};
    res[llvm::Instruction::ExtractValue] = {&Derived::compile_extract_value, 0};
    res[llvm::Instruction::InsertValue] = {&Derived::compile_insert_value, 0};
    res[llvm::Instruction::LandingPad] = {&Derived::compile_landing_pad, 0};
    res[llvm::Instruction::Freeze] = {&Derived::compile_freeze, 0};

    // clang-format on
    return res;
  }();

  const ValInfo &val_info = this->adaptor->val_info(i);
  assert(i->getOpcode() < fns.size());
  const auto [compile_fn, arg] = fns[i->getOpcode()];
  return (derived()->*compile_fn)(i, val_info, arg);
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_unreachable(
    const llvm::Instruction *, const ValInfo &, u64) noexcept {
  derived()->encode_trap();
  this->release_regs_after_return();
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_ret(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  typename Derived::RetBuilder rb{*derived(), *derived()->cur_cc_assigner()};
  if (inst->getNumOperands() != 0) {
    llvm::Value *retval = inst->getOperand(0);
    bool handled = false;
    llvm::Type *ret_ty = retval->getType();
    if (ret_ty->isIntegerTy()) {
      if (unsigned width = ret_ty->getIntegerBitWidth(); width % 32 != 0) {
        assert(width < 64 && "non-i128 multi-word int should be illegal");
        unsigned dst_width = width < 32 ? 32 : 64;
        llvm::AttributeList attrs = this->adaptor->cur_func->getAttributes();
        llvm::AttributeSet ret_attrs = attrs.getRetAttrs();
        if (ret_attrs.hasAttribute(llvm::Attribute::ZExt)) {
          auto [vr, vpr] = this->val_ref_single(retval);
          rb.add(std::move(vpr).into_extended(false, width, dst_width), {});
          handled = true;
        } else if (ret_attrs.hasAttribute(llvm::Attribute::SExt)) {
          auto [vr, vpr] = this->val_ref_single(retval);
          rb.add(std::move(vpr).into_extended(true, width, dst_width), {});
          handled = true;
        }
      }
    } else if (ret_ty->isX86_FP80Ty()) {
      if constexpr (requires { &Derived::fp80_push; }) {
        derived()->fp80_push(this->val_ref(retval).part(0));
        handled = true;
      } else {
        return false;
      }
    }

    if (!handled) {
      rb.add(retval);
    }
  }

  rb.ret();
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_load_generic(
    const llvm::LoadInst *load, GenericValuePart &&ptr_op) noexcept {
  // TODO(ts): if the ref-count is <= 1, then skip emitting the load as LLVM
  // does that, too. at least on ARM

  if (load->isAtomic()) {
    u32 width = 64;
    if (load->getType()->isIntegerTy()) {
      width = load->getType()->getIntegerBitWidth();
      if (width != 8 && width != 16 && width != 32 && width != 64) {
        TPDE_LOG_ERR("atomic loads not of i8/i16/i32/i64/ptr not supported");
        return false;
      }
    } else if (!load->getType()->isPointerTy()) {
      TPDE_LOG_ERR("atomic loads not of i8/i16/i32/i64/ptr not supported");
      return false;
    }
    u32 needed_align = 1;
    switch (width) {
    case 16: needed_align = 2; break;
    case 32: needed_align = 4; break;
    case 64: needed_align = 8; break;
    }
    if (load->getAlign().value() < needed_align) {
      TPDE_LOG_ERR(
          "atomic load of width {} has alignment {} which is too small",
          width,
          load->getAlign().value());
      return false;
    }

    const auto order = load->getOrdering();
    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ValuePart &&);
    EncodeFnTy encode_fn = nullptr;
    if (order == llvm::AtomicOrdering::Monotonic) {
      switch (width) {
      case 8: encode_fn = &Derived::encode_atomic_load_u8_mono; break;
      case 16: encode_fn = &Derived::encode_atomic_load_u16_mono; break;
      case 32: encode_fn = &Derived::encode_atomic_load_u32_mono; break;
      case 64: encode_fn = &Derived::encode_atomic_load_u64_mono; break;
      default: TPDE_UNREACHABLE("invalid size");
      }
    } else if (order == llvm::AtomicOrdering::Acquire) {
      switch (width) {
      case 8: encode_fn = &Derived::encode_atomic_load_u8_acq; break;
      case 16: encode_fn = &Derived::encode_atomic_load_u16_acq; break;
      case 32: encode_fn = &Derived::encode_atomic_load_u32_acq; break;
      case 64: encode_fn = &Derived::encode_atomic_load_u64_acq; break;
      default: TPDE_UNREACHABLE("invalid size");
      }
    } else {
      assert(order == llvm::AtomicOrdering::SequentiallyConsistent);
      switch (width) {
      case 8: encode_fn = &Derived::encode_atomic_load_u8_seqcst; break;
      case 16: encode_fn = &Derived::encode_atomic_load_u16_seqcst; break;
      case 32: encode_fn = &Derived::encode_atomic_load_u32_seqcst; break;
      case 64: encode_fn = &Derived::encode_atomic_load_u64_seqcst; break;
      default: TPDE_UNREACHABLE("invalid size");
      }
    }

    ValueRef res = this->result_ref(load);
    (derived()->*encode_fn)(std::move(ptr_op), res.part(0));
    return true;
  }

  unsigned num_bits;
  bool sext = false;
  const llvm::Instruction *target = load;
  switch (this->adaptor->val_info(load).type) {
    using enum LLVMBasicValType;
  case i1:
  case i8:
  case i16:
  case i32:
  case i64: {
    assert(load->getType()->isIntegerTy());
    num_bits = load->getType()->getIntegerBitWidth();
    // Most loads are 8/16/32/64-bit loads, and are frequently zext/sext-ed or
    // trunc-ed (e.g. to i1) immediately afterwards.
    if (load->hasOneUse()) {
      auto *user = llvm::cast<llvm::Instruction>(load->use_begin()->getUser());
      // TODO: evaluate when same block condition is actually required.
      if (user->getParent() != load->getParent()) {
        goto load_single_integer;
      }
      if (llvm::isa<llvm::TruncInst>(user)) {
        // Trunc is a no-op, so simply make the load the result.
        target = user;
        this->adaptor->inst_set_fused(user, true);
      } else if (llvm::isa<llvm::ZExtInst, llvm::SExtInst>(user) &&
                 user->getType()->getIntegerBitWidth() <= 64 && num_bits >= 8 &&
                 (num_bits & (num_bits - 1)) == 0) {
        // 8/16/32/64-bit loads can trivially zext/sext.
        target = user;
        sext = llvm::isa<llvm::SExtInst>(user);
        this->adaptor->inst_set_fused(user, true);
      }
    }

  load_single_integer:
    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ValuePart &&);
    static constexpr auto fns = []() consteval {
      std::array<EncodeFnTy[2], 8> res{};
      res[0][0] = &Derived::encode_loadi8_zext;
      res[0][1] = &Derived::encode_loadi8_sext;
      res[1][0] = &Derived::encode_loadi16_zext;
      res[1][1] = &Derived::encode_loadi16_sext;
      res[2][0] = &Derived::encode_loadi24;
      res[3][0] = &Derived::encode_loadi32_zext;
      res[3][1] = &Derived::encode_loadi32_sext;
      res[4][0] = &Derived::encode_loadi40;
      res[5][0] = &Derived::encode_loadi48;
      res[6][0] = &Derived::encode_loadi56;
      res[7][0] = &Derived::encode_loadi64;
      return res;
    }();
    EncodeFnTy fn = fns[(num_bits - 1) / 8][sext];
    (derived()->*fn)(std::move(ptr_op), this->result_ref(target).part(0));
    break;
  }
  case v8i1:
  case v16i1:
  case v32i1:
  case v64i1:
    assert(load->getType()->getScalarType()->isIntegerTy(1));
    num_bits =
        llvm::cast<llvm::FixedVectorType>(load->getType())->getNumElements();
    goto load_single_integer;

  case ptr: num_bits = 64; goto load_single_integer;

  case i128: {
    ValueRef res = this->result_ref(load);
    derived()->encode_loadi128(std::move(ptr_op), res.part(0), res.part(1));
    break;
  }
  case f32:
    derived()->encode_loadf32(std::move(ptr_op),
                              this->result_ref(load).part(0));
    break;
  case v8i8:
  case v4i16:
  case v2i32:
  case v2f32:
  case f64:
    derived()->encode_loadf64(std::move(ptr_op),
                              this->result_ref(load).part(0));
    break;
  case v16i8:
  case v8i16:
  case v4i32:
  case v2i64:
  case v4f32:
  case v2f64:
  case f128:
    derived()->encode_loadv128(std::move(ptr_op),
                               this->result_ref(load).part(0));
    break;
  case f80:
    if constexpr (requires { &Derived::fp80_load; }) {
      derived()->fp80_load(std::move(ptr_op), this->result_ref(load).part(0));
      break;
    }
    return false;
  case complex: {
    auto ty_idx = this->adaptor->val_info(load).complex_part_tys_idx;
    const LLVMComplexPart *part_descs =
        &this->adaptor->complex_part_types[ty_idx + 1];
    unsigned part_count = part_descs[-1].desc.num_parts;

    // TODO: fuse expr; not easy, because we lose the GVP
    AsmReg ptr_reg = this->gval_as_reg(ptr_op);

    ValueRef res = this->result_ref(target);
    unsigned off = 0;
    for (unsigned i = 0; i < part_count; i++) {
      auto part_addr =
          typename GenericValuePart::Expr{ptr_reg, static_cast<tpde::i32>(off)};
      auto part_ty = part_descs[i].part.type;
      switch (part_ty) {
      case i1:
      case i8:
      case v8i1:
        derived()->encode_loadi8_zext(std::move(part_addr), res.part(i));
        break;
      case i16:
      case v16i1:
        derived()->encode_loadi16_zext(std::move(part_addr), res.part(i));
        break;
      case i32:
      case v32i1:
        derived()->encode_loadi32_zext(std::move(part_addr), res.part(i));
        break;
      case i64:
      case v64i1:
      case ptr:
        derived()->encode_loadi64(std::move(part_addr), res.part(i));
        break;
      case i128:
        derived()->encode_loadi64(std::move(part_addr), res.part(i));
        break;
      case f32:
        derived()->encode_loadf32(std::move(part_addr), res.part(i));
        break;
      case v8i8:
      case v4i16:
      case v2i32:
      case v2f32:
      case f64:
        derived()->encode_loadf64(std::move(part_addr), res.part(i));
        break;
      case v16i8:
      case v8i16:
      case v4i32:
      case v2i64:
      case v4f32:
      case v2f64:
      case f128:
        derived()->encode_loadv128(std::move(part_addr), res.part(i));
        break;
      case f80:
        if constexpr (requires { &Derived::fp80_load; }) {
          derived()->fp80_load(std::move(part_addr), res.part(i));
          break;
        }
        return false;
      default: return false;
      }

      off += part_descs[i].part.size + part_descs[i].part.pad_after;
    }
    return true;
  }
  default: return false;
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_load(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *load = llvm::cast<llvm::LoadInst>(inst);
  auto [_, ptr_ref] = this->val_ref_single(load->getPointerOperand());
  if (ptr_ref.has_assignment() && ptr_ref.assignment().is_stack_variable()) {
    GenericValuePart addr =
        derived()->create_addr_for_alloca(ptr_ref.assignment());
    return compile_load_generic(load, std::move(addr));
  }

  return compile_load_generic(load, std::move(ptr_ref));
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_store_generic(
    const llvm::StoreInst *store, GenericValuePart &&ptr_op) noexcept {
  const auto *op_val = store->getValueOperand();
  auto op_ref = this->val_ref(op_val);

  if (store->isAtomic()) {
    u32 width = 64;
    if (op_val->getType()->isIntegerTy()) {
      width = op_val->getType()->getIntegerBitWidth();
      if (width != 8 && width != 16 && width != 32 && width != 64) {
        TPDE_LOG_ERR("atomic loads not of i8/i16/i32/i64/ptr not supported");
        return false;
      }
    } else if (!op_val->getType()->isPointerTy()) {
      TPDE_LOG_ERR("atomic loads not of i8/i16/i32/i64/ptr not supported");
      return false;
    }

    if (auto align = store->getAlign().value(); align * 8 < width) {
      TPDE_LOG_ERR("unaligned store ({}) not implemented", align);
      return false;
    }

    const auto order = store->getOrdering();
    using EncodeFnTy =
        bool (Derived::*)(GenericValuePart &&, GenericValuePart &&);
    EncodeFnTy encode_fn = nullptr;
    if (order == llvm::AtomicOrdering::Monotonic) {
      switch (width) {
      case 8: encode_fn = &Derived::encode_atomic_store_u8_mono; break;
      case 16: encode_fn = &Derived::encode_atomic_store_u16_mono; break;
      case 32: encode_fn = &Derived::encode_atomic_store_u32_mono; break;
      case 64: encode_fn = &Derived::encode_atomic_store_u64_mono; break;
      default: TPDE_UNREACHABLE("invalid size");
      }
    } else if (order == llvm::AtomicOrdering::Release) {
      switch (width) {
      case 8: encode_fn = &Derived::encode_atomic_store_u8_rel; break;
      case 16: encode_fn = &Derived::encode_atomic_store_u16_rel; break;
      case 32: encode_fn = &Derived::encode_atomic_store_u32_rel; break;
      case 64: encode_fn = &Derived::encode_atomic_store_u64_rel; break;
      default: TPDE_UNREACHABLE("invalid size");
      }
    } else {
      assert(order == llvm::AtomicOrdering::SequentiallyConsistent);
      switch (width) {
      case 8: encode_fn = &Derived::encode_atomic_store_u8_seqcst; break;
      case 16: encode_fn = &Derived::encode_atomic_store_u16_seqcst; break;
      case 32: encode_fn = &Derived::encode_atomic_store_u32_seqcst; break;
      case 64: encode_fn = &Derived::encode_atomic_store_u64_seqcst; break;
      default: TPDE_UNREACHABLE("invalid size");
      }
    }

    if (!(derived()->*encode_fn)(std::move(ptr_op), op_ref.part(0))) {
      TPDE_LOG_ERR("fooooo");
      return false;
    }
    return true;
  }

  // TODO: don't recompute this, this is currently computed for every val part
  auto [ty, ty_idx] = this->adaptor->lower_type(op_val->getType());

  unsigned num_bits;
  switch (ty) {
    using enum LLVMBasicValType;
  case i1:
  case i8:
  case i16:
  case i32:
  case i64: {
    assert(op_val->getType()->isIntegerTy());
    num_bits = op_val->getType()->getIntegerBitWidth();
  store_single_integer:
    switch (tpde::util::align_up(num_bits, 8)) {
    case 1:
    case 8: derived()->encode_storei8(std::move(ptr_op), op_ref.part(0)); break;
    case 16:
      derived()->encode_storei16(std::move(ptr_op), op_ref.part(0));
      break;
    case 24:
      derived()->encode_storei24(std::move(ptr_op), op_ref.part(0));
      break;
    case 32:
      derived()->encode_storei32(std::move(ptr_op), op_ref.part(0));
      break;
    case 40:
      derived()->encode_storei40(std::move(ptr_op), op_ref.part(0));
      break;
    case 48:
      derived()->encode_storei48(std::move(ptr_op), op_ref.part(0));
      break;
    case 56:
      derived()->encode_storei56(std::move(ptr_op), op_ref.part(0));
      break;
    case 64:
      derived()->encode_storei64(std::move(ptr_op), op_ref.part(0));
      break;
    default: return false;
    }
    break;
  }
  case v8i1:
  case v16i1:
  case v32i1:
  case v64i1:
    assert(op_val->getType()->getScalarType()->isIntegerTy(1));
    num_bits =
        llvm::cast<llvm::FixedVectorType>(op_val->getType())->getNumElements();
    goto store_single_integer;

  case ptr:
    derived()->encode_storei64(std::move(ptr_op), op_ref.part(0));
    break;
  case i128:
    derived()->encode_storei128(
        std::move(ptr_op), op_ref.part(0), op_ref.part(1));
    break;
  case f32:
    derived()->encode_storef32(std::move(ptr_op), op_ref.part(0));
    break;
  case v8i8:
  case v4i16:
  case v2i32:
  case v2f32:
  case f64:
    derived()->encode_storef64(std::move(ptr_op), op_ref.part(0));
    break;
  case v16i8:
  case v8i16:
  case v4i32:
  case v2i64:
  case v4f32:
  case v2f64:
  case f128:
    derived()->encode_storev128(std::move(ptr_op), op_ref.part(0));
    break;
  case f80:
    if constexpr (requires { &Derived::fp80_store; }) {
      derived()->fp80_store(std::move(ptr_op), op_ref.part(0));
      break;
    }
    return false;
  case complex: {
    const LLVMComplexPart *part_descs =
        &this->adaptor->complex_part_types[ty_idx + 1];
    unsigned part_count = part_descs[-1].desc.num_parts;

    // TODO: fuse expr; not easy, because we lose the GVP
    AsmReg ptr_reg = this->gval_as_reg(ptr_op);

    unsigned off = 0;
    for (unsigned i = 0; i < part_count; i++) {
      auto part_ref = op_ref.part(i);
      auto part_addr =
          typename GenericValuePart::Expr{ptr_reg, static_cast<tpde::i32>(off)};
      // Note: val_ref might call val_ref_special, which calls val_parts, which
      // calls lower_type, which will invalidate part_descs.
      // TODO: don't recompute value parts for every constant part
      const LLVMComplexPart *part_descs =
          &this->adaptor->complex_part_types[ty_idx + 1];
      auto part_ty = part_descs[i].part.type;
      switch (part_ty) {
      case i1:
      case i8:
      case v8i1:
        derived()->encode_storei8(std::move(part_addr), std::move(part_ref));
        break;
      case i16:
      case v16i1:
        derived()->encode_storei16(std::move(part_addr), std::move(part_ref));
        break;
      case i32:
      case v32i1:
        derived()->encode_storei32(std::move(part_addr), std::move(part_ref));
        break;
      case i64:
      case v64i1:
      case ptr:
        derived()->encode_storei64(std::move(part_addr), std::move(part_ref));
        break;
      case i128:
        derived()->encode_storei64(std::move(part_addr), std::move(part_ref));
        break;
      case f32:
        derived()->encode_storef32(std::move(part_addr), std::move(part_ref));
        break;
      case v8i8:
      case v4i16:
      case v2i32:
      case v2f32:
      case f64:
        derived()->encode_storef64(std::move(part_addr), std::move(part_ref));
        break;
      case v16i8:
      case v8i16:
      case v4i32:
      case v2i64:
      case v4f32:
      case v2f64:
      case f128:
        derived()->encode_storev128(std::move(part_addr), std::move(part_ref));
        break;
      case f80:
        if constexpr (requires { &Derived::fp80_store; }) {
          derived()->fp80_store(std::move(part_addr), std::move(part_ref));
          break;
        }
        return false;
      default: return false;
      }

      off += part_descs[i].part.size + part_descs[i].part.pad_after;
    }
    return true;
  }
  default: return false;
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_store(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *store = llvm::cast<llvm::StoreInst>(inst);
  auto [_, ptr_ref] = this->val_ref_single(store->getPointerOperand());
  if (ptr_ref.has_assignment() && ptr_ref.assignment().is_stack_variable()) {
    GenericValuePart addr =
        derived()->create_addr_for_alloca(ptr_ref.assignment());
    return compile_store_generic(store, std::move(addr));
  }

  return compile_store_generic(store, std::move(ptr_ref));
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_int_binary_op_i128(
    const llvm::Instruction *inst, const ValInfo &, IntBinaryOp op) noexcept {
  assert(inst->getType()->getIntegerBitWidth() == 128 &&
         "non-i128 multi-word integer should not be legal");
  llvm::Value *lhs_op = inst->getOperand(0);
  llvm::Value *rhs_op = inst->getOperand(1);

  auto res = this->result_ref(inst);

  if (op.is_div() || op.is_rem()) {
    LibFunc lf;
    if (op.is_div()) {
      lf = op.is_signed() ? LibFunc::divti3 : LibFunc::udivti3;
    } else {
      lf = op.is_signed() ? LibFunc::modti3 : LibFunc::umodti3;
    }

    std::array<IRValueRef, 2> args{lhs_op, rhs_op};
    derived()->create_helper_call(args, &res, get_libfunc_sym(lf));
    return true;
  }

  auto lhs = this->val_ref(lhs_op);
  auto rhs = this->val_ref(rhs_op);

  // Use has_assignment as proxy for not being a constant.
  if (op.is_symmetric() && !lhs.has_assignment() && rhs.has_assignment()) {
    // TODO(ts): this is a hack since the encoder can currently not do
    // commutable operations so we reorder immediates manually here
    std::swap(lhs, rhs);
  }

  if (op.is_shift()) {
    ValuePartRef shift_amt = rhs.part(0);
    if (shift_amt.is_const()) {
      u64 imm1 = shift_amt.const_data()[0] & 0b111'1111; // amt
      if (imm1 < 64) {
        u64 imm2 = (64 - imm1) & 0b11'1111; // iamt
        if (op == IntBinaryOp::shl) {
          derived()->encode_shli128_lt64(
              lhs.part(0),
              lhs.part(1),
              ValuePartRef(this, imm1, 1, Config::GP_BANK),
              ValuePartRef(this, imm2, 1, Config::GP_BANK),
              res.part(0),
              res.part(1));
        } else if (op == IntBinaryOp::shr) {
          derived()->encode_shri128_lt64(
              lhs.part(0),
              lhs.part(1),
              ValuePartRef(this, imm1, 1, Config::GP_BANK),
              ValuePartRef(this, imm2, 1, Config::GP_BANK),
              res.part(0),
              res.part(1));
        } else {
          assert(op == IntBinaryOp::ashr);
          derived()->encode_ashri128_lt64(
              lhs.part(0),
              lhs.part(1),
              ValuePartRef(this, imm1, 1, Config::GP_BANK),
              ValuePartRef(this, imm2, 1, Config::GP_BANK),
              res.part(0),
              res.part(1));
        }
      } else {
        imm1 -= 64;
        if (op == IntBinaryOp::shl) {
          derived()->encode_shli128_ge64(
              lhs.part(0),
              ValuePartRef(this, imm1, 1, Config::GP_BANK),
              res.part(0),
              res.part(1));
        } else if (op == IntBinaryOp::shr) {
          derived()->encode_shri128_ge64(
              lhs.part(1),
              ValuePartRef(this, imm1, 1, Config::GP_BANK),
              res.part(0),
              res.part(1));
        } else {
          assert(op == IntBinaryOp::ashr);
          derived()->encode_ashri128_ge64(
              lhs.part(1),
              ValuePartRef(this, imm1, 1, Config::GP_BANK),
              res.part(0),
              res.part(1));
        }
      }
    } else {
      if (op == IntBinaryOp::shl) {
        derived()->encode_shli128(lhs.part(0),
                                  lhs.part(1),
                                  std::move(shift_amt),
                                  res.part(0),
                                  res.part(1));
      } else if (op == IntBinaryOp::shr) {
        derived()->encode_shri128(lhs.part(0),
                                  lhs.part(1),
                                  std::move(shift_amt),
                                  res.part(0),
                                  res.part(1));
      } else {
        assert(op == IntBinaryOp::ashr);
        derived()->encode_ashri128(lhs.part(0),
                                   lhs.part(1),
                                   std::move(shift_amt),
                                   res.part(0),
                                   res.part(1));
      }
    }
  } else {
    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&,
                                         GenericValuePart &&,
                                         GenericValuePart &&,
                                         GenericValuePart &&,
                                         ValuePart &&,
                                         ValuePart &&);
    static const std::array<EncodeFnTy, 10> encode_ptrs = {
        {
         &Derived::encode_addi128,
         &Derived::encode_subi128,
         &Derived::encode_muli128,
         nullptr, // division/remainder is a libcall
            nullptr, // division/remainder is a libcall
            nullptr, // division/remainder is a libcall
            nullptr, // division/remainder is a libcall
            &Derived::encode_landi128,
         &Derived::encode_lori128,
         &Derived::encode_lxori128,
         }
    };

    (derived()->*(encode_ptrs[op.index()]))(lhs.part(0),
                                            lhs.part(1),
                                            rhs.part(0),
                                            rhs.part(1),
                                            res.part(0),
                                            res.part(1));
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_int_binary_op(
    const llvm::Instruction *inst, const ValInfo &info, u64 op_val) noexcept {
  IntBinaryOp op = typename IntBinaryOp::Value(op_val);
  auto parts = this->adaptor->val_parts(info);
  if (info.type == LLVMBasicValType::i128) [[unlikely]] {
    return compile_int_binary_op_i128(inst, info, op);
  }

  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ValuePart &);
  // fns[op.index()][idx]
  static constexpr auto fns = []() constexpr {
    std::array<EncodeFnTy[12], IntBinaryOp::num_ops> res{};
    auto entry = [&res](IntBinaryOp op) { return res[op.index()]; };

    // TODO: more consistent naming of encode functions
#define FN_ENTRY_INT(op, fn)                                                   \
  entry(op)[1] = &Derived::encode_##fn##i##32;                                 \
  entry(op)[2] = &Derived::encode_##fn##i##64;
#define FN_ENTRY_VEC(op, fn, sign)                                             \
  entry(op)[3] = &Derived::encode_##fn##v8##sign##8;                           \
  entry(op)[4] = &Derived::encode_##fn##v4##sign##16;                          \
  entry(op)[5] = &Derived::encode_##fn##v2##sign##32;                          \
  entry(op)[6] = &Derived::encode_##fn##v16##sign##8;                          \
  entry(op)[7] = &Derived::encode_##fn##v8##sign##16;                          \
  entry(op)[8] = &Derived::encode_##fn##v4##sign##32;                          \
  entry(op)[9] = &Derived::encode_##fn##v2##sign##64;
#define FN_ENTRY(op, fn, sign) FN_ENTRY_INT(op, fn) FN_ENTRY_VEC(op, fn, sign)

    FN_ENTRY(IntBinaryOp::add, add, u)
    FN_ENTRY(IntBinaryOp::sub, sub, u)
    FN_ENTRY(IntBinaryOp::mul, mul, u)
    FN_ENTRY_INT(IntBinaryOp::udiv, udiv)
    FN_ENTRY_INT(IntBinaryOp::sdiv, sdiv)
    FN_ENTRY_INT(IntBinaryOp::urem, urem)
    FN_ENTRY_INT(IntBinaryOp::srem, srem)
    FN_ENTRY(IntBinaryOp::land, land, u)
    FN_ENTRY(IntBinaryOp::lxor, lxor, u)
    FN_ENTRY(IntBinaryOp::lor, lor, u)
    FN_ENTRY(IntBinaryOp::shl, shl, u)
    FN_ENTRY(IntBinaryOp::shr, shr, u)
    FN_ENTRY(IntBinaryOp::ashr, ashr, i)
#undef FN_ENTRY
#undef FN_ENTRY_VEC
#undef FN_ENTRY_INT

    // i1 is special.
    entry(IntBinaryOp::add)[10] = &Derived::encode_lxori32;
    entry(IntBinaryOp::add)[11] = &Derived::encode_lxori64;
    entry(IntBinaryOp::sub)[10] = &Derived::encode_lxori32;
    entry(IntBinaryOp::sub)[11] = &Derived::encode_lxori64;
    entry(IntBinaryOp::mul)[10] = &Derived::encode_landi32;
    entry(IntBinaryOp::mul)[11] = &Derived::encode_landi64;
    // udiv: x/1 = x; x/0 = UB => and is equivalent
    entry(IntBinaryOp::udiv)[10] = &Derived::encode_landi32;
    entry(IntBinaryOp::udiv)[11] = &Derived::encode_landi64;
    // sdiv: 0/-1 = 0; -1/-1 = UB; x/0 = UB => and is equivalent
    entry(IntBinaryOp::sdiv)[10] = &Derived::encode_landi32;
    entry(IntBinaryOp::sdiv)[11] = &Derived::encode_landi64;
    // urem/srem are always zero, but we have no encode function to return zero.
    // For now, keep them unassigned.
    entry(IntBinaryOp::land)[10] = &Derived::encode_landi32;
    entry(IntBinaryOp::land)[11] = &Derived::encode_landi64;
    entry(IntBinaryOp::lxor)[10] = &Derived::encode_lxori32;
    entry(IntBinaryOp::lxor)[11] = &Derived::encode_lxori64;
    entry(IntBinaryOp::lor)[10] = &Derived::encode_lori32;
    entry(IntBinaryOp::lor)[11] = &Derived::encode_lori64;
    // shl/lshr/ashr are always poison, so we could use any operation... for
    // now, keep them unassigned.

    return res;
  }();
  auto get_encode_fn =
      [op](LLVMBasicValType bvt) -> std::pair<EncodeFnTy, bool> {
    static constexpr auto bvt_lut = []() consteval {
      std::array<u8, unsigned(LLVMBasicValType::max)> res{};
      res[unsigned(LLVMBasicValType::i8)] = 1;
      res[unsigned(LLVMBasicValType::i16)] = 1;
      res[unsigned(LLVMBasicValType::i32)] = 1;
      res[unsigned(LLVMBasicValType::i64)] = 2;
      res[unsigned(LLVMBasicValType::v8i8)] = 3;
      res[unsigned(LLVMBasicValType::v4i16)] = 4;
      res[unsigned(LLVMBasicValType::v2i32)] = 5;
      res[unsigned(LLVMBasicValType::v16i8)] = 6;
      res[unsigned(LLVMBasicValType::v8i16)] = 7;
      res[unsigned(LLVMBasicValType::v4i32)] = 8;
      res[unsigned(LLVMBasicValType::v2i64)] = 9;
      res[unsigned(LLVMBasicValType::v8i1)] = 10;
      res[unsigned(LLVMBasicValType::v16i1)] = 10;
      res[unsigned(LLVMBasicValType::v32i1)] = 10;
      res[unsigned(LLVMBasicValType::v64i1)] = 11;
      return res;
    }();
    unsigned ty_idx = bvt_lut[unsigned(bvt)];
    return {fns[op.index()][ty_idx], ty_idx < 3};
  };

  unsigned int_width = inst->getType()->getScalarType()->getIntegerBitWidth();
  assert(int_width <= 64);
  const llvm::Use *operands = inst->getOperandList();
  ValueRef lhs = this->val_ref(operands[0]);
  ValueRef rhs = this->val_ref(operands[1]);
  ValueRef res = this->result_ref(inst);

  auto handle_part = [this, int_width, op](EncodeFnTy encode_fn,
                                           bool is_scalar,
                                           ValuePartRef &&lhs_op,
                                           ValuePartRef &&rhs_op,
                                           ValuePartRef &res_op) {
    if (is_scalar) {
      if (op.is_symmetric() && lhs_op.is_const() && !rhs_op.is_const()) {
        // TODO(ts): this is a hack since the encoder can currently not do
        // commutable operations so we reorder immediates manually here
        std::swap(lhs_op, rhs_op);
      }

      // TODO(ts): optimize div/rem by constant to a shift?
      unsigned ext_width = tpde::util::align_up(int_width, 32);
      if (ext_width != int_width) {
        bool sext = op.is_signed();
        if (op.needs_lhs_ext()) {
          lhs_op = std::move(lhs_op).into_extended(sext, int_width, ext_width);
        }
        if (op.needs_rhs_ext()) {
          rhs_op = std::move(rhs_op).into_extended(sext, int_width, ext_width);
        }
      }
    }

    (derived()->*encode_fn)(std::move(lhs_op), std::move(rhs_op), res_op);
  };

  for (u32 i = 0, n = parts.count(); i != n; ++i) {
    LLVMBasicValType ty = parts.type(i);
    ValuePartRef res_part = res.part(i);
    if (auto [encode_fn, is_scalar] = get_encode_fn(ty); encode_fn) [[likely]] {
      handle_part(encode_fn, is_scalar, lhs.part(i), rhs.part(i), res_part);
      continue;
    }

    // This is a legal vector type for which we don't have an encode function.
    // Extract elements individually and use scalar functions.
    if (!inst->getType()->isVectorTy() || int_width == 1) {
      return false;
    }
    auto [elem_cnt, elem_ty] = basic_ty_vector_info(ty);
    auto [encode_fn, is_scalar] = get_encode_fn(elem_ty);
    assert(is_scalar && "vector element must be a scalar type");
    if (!encode_fn) {
      return false;
    }

    tpde::RegBank bank = this->adaptor->basic_ty_part_bank(elem_ty);
    for (u32 j = 0; j != elem_cnt; ++j) {
      u32 elem_idx = i * elem_cnt + j;
      ValuePartRef e_res{this, bank};
      ValuePartRef e_lhs{this, bank};
      ValuePartRef e_rhs{this, bank};
      // TODO: we might pass the last element as owned. But this code is
      // fallback only, so don't bother optimizing.
      ValueRef lhs_unowned = lhs.disowned();
      ValueRef rhs_unowned = rhs.disowned();
      derived()->extract_element(lhs_unowned, elem_idx, elem_ty, e_lhs);
      derived()->extract_element(rhs_unowned, elem_idx, elem_ty, e_rhs);
      handle_part(encode_fn, true, std::move(e_lhs), std::move(e_rhs), e_res);
      // insert_element always treats res as unowned.
      derived()->insert_element(res, elem_idx, elem_ty, std::move(e_res));
    }
  }
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_float_binary_op(
    const llvm::Instruction *inst, const ValInfo &val_info, u64 op) noexcept {
  auto lhs = this->val_ref(inst->getOperand(0));
  auto rhs = this->val_ref(inst->getOperand(1));
  ValueRef res = this->result_ref(inst);

  if (val_info.type == LLVMBasicValType::f128) {
    LibFunc lf;
    switch (op) {
    case FloatBinaryOp::add: lf = LibFunc::addtf3; break;
    case FloatBinaryOp::sub: lf = LibFunc::subtf3; break;
    case FloatBinaryOp::mul: lf = LibFunc::multf3; break;
    case FloatBinaryOp::div: lf = LibFunc::divtf3; break;
    case FloatBinaryOp::rem: return false;
    default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
    }
    auto cb = derived()->create_call_builder();
    cb->add_arg(lhs.part(0), tpde::CCAssignment{});
    cb->add_arg(rhs.part(0), tpde::CCAssignment{});
    cb->call(get_libfunc_sym(lf));
    cb->add_ret(res);
    return true;
  }

  if (op == FloatBinaryOp::rem) {
    LibFunc lf;
    switch (val_info.type) {
    case LLVMBasicValType::f32: lf = LibFunc::fmodf; break;
    case LLVMBasicValType::f64: lf = LibFunc::fmod; break;
    default: return false;
    }

    auto cb = derived()->create_call_builder();
    cb->add_arg(lhs.part(0), tpde::CCAssignment{});
    cb->add_arg(rhs.part(0), tpde::CCAssignment{});
    cb->call(get_libfunc_sym(lf));
    cb->add_ret(res);
    return true;
  }

  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ValuePart &&);
  EncodeFnTy encode_fn = nullptr;

  switch (val_info.type) {
    using enum LLVMBasicValType;
  case f32:
    switch (op) {
    case FloatBinaryOp::add: encode_fn = &Derived::encode_addf32; break;
    case FloatBinaryOp::sub: encode_fn = &Derived::encode_subf32; break;
    case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulf32; break;
    case FloatBinaryOp::div: encode_fn = &Derived::encode_divf32; break;
    default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
    }
    break;
  case f64:
    switch (op) {
    case FloatBinaryOp::add: encode_fn = &Derived::encode_addf64; break;
    case FloatBinaryOp::sub: encode_fn = &Derived::encode_subf64; break;
    case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulf64; break;
    case FloatBinaryOp::div: encode_fn = &Derived::encode_divf64; break;
    default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
    }
    break;
  case v2f32:
    switch (op) {
    case FloatBinaryOp::add: encode_fn = &Derived::encode_addv2f32; break;
    case FloatBinaryOp::sub: encode_fn = &Derived::encode_subv2f32; break;
    case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulv2f32; break;
    case FloatBinaryOp::div: encode_fn = &Derived::encode_divv2f32; break;
    default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
    }
    break;
  case v4f32:
    switch (op) {
    case FloatBinaryOp::add: encode_fn = &Derived::encode_addv4f32; break;
    case FloatBinaryOp::sub: encode_fn = &Derived::encode_subv4f32; break;
    case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulv4f32; break;
    case FloatBinaryOp::div: encode_fn = &Derived::encode_divv4f32; break;
    default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
    }
    break;
  case v2f64:
    switch (op) {
    case FloatBinaryOp::add: encode_fn = &Derived::encode_addv2f64; break;
    case FloatBinaryOp::sub: encode_fn = &Derived::encode_subv2f64; break;
    case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulv2f64; break;
    case FloatBinaryOp::div: encode_fn = &Derived::encode_divv2f64; break;
    default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
    }
    break;
  case f80:
    if constexpr (requires { &Derived::fp80_load; }) {
      switch (op) {
      case FloatBinaryOp::add:
        derived()->fp80_add(lhs.part(0), rhs.part(0), res.part(0));
        return true;
      case FloatBinaryOp::sub:
        derived()->fp80_sub(lhs.part(0), rhs.part(0), res.part(0));
        return true;
      case FloatBinaryOp::mul:
        derived()->fp80_mul(lhs.part(0), rhs.part(0), res.part(0));
        return true;
      case FloatBinaryOp::div:
        derived()->fp80_div(lhs.part(0), rhs.part(0), res.part(0));
        return true;
      default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
      }
    } else {
      return false;
    }
  default: return false;
  }

  return (derived()->*encode_fn)(lhs.part(0), rhs.part(0), res.part(0));
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_fneg(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  ValueRef src = this->val_ref(inst->getOperand(0));
  ValueRef res = this->result_ref(inst);
  switch (val_info.type) {
    using enum LLVMBasicValType;
  case f32: derived()->encode_fnegf32(src.part(0), res.part(0)); break;
  case f64: derived()->encode_fnegf64(src.part(0), res.part(0)); break;
  case f128: derived()->encode_fnegf128(src.part(0), res.part(0)); break;
  case v2f32: derived()->encode_fnegv2f32(src.part(0), res.part(0)); break;
  case v4f32: derived()->encode_fnegv4f32(src.part(0), res.part(0)); break;
  case v2f64: derived()->encode_fnegv2f64(src.part(0), res.part(0)); break;
  case f80:
    if constexpr (requires { &Derived::fp80_neg; }) {
      derived()->fp80_neg(src.part(0), res.part(0));
      return true;
    }
    return false;
  default: return false;
  }
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_float_ext_trunc(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const llvm::Value *src_val = inst->getOperand(0);
  auto *src_ty = src_val->getType();
  auto *dst_ty = inst->getType();

  auto res_vr = this->result_ref(inst);

  SymRef sym;
  if (src_ty->isDoubleTy() && dst_ty->isFloatTy()) {
    auto src_ref = this->val_ref(src_val);
    derived()->encode_f64tof32(src_ref.part(0), res_vr.part(0));
  } else if (src_ty->isFP128Ty() && dst_ty->isFloatTy()) {
    sym = get_libfunc_sym(LibFunc::trunctfsf2);
  } else if (src_ty->isFP128Ty() && dst_ty->isDoubleTy()) {
    sym = get_libfunc_sym(LibFunc::trunctfdf2);
  } else if (src_ty->isFloatTy() && dst_ty->isDoubleTy()) {
    auto src_ref = this->val_ref(src_val);
    derived()->encode_f32tof64(src_ref.part(0), res_vr.part(0));
  } else if (src_ty->isFloatTy() && dst_ty->isFP128Ty()) {
    sym = get_libfunc_sym(LibFunc::extendsftf2);
  } else if (src_ty->isDoubleTy() && dst_ty->isFP128Ty()) {
    sym = get_libfunc_sym(LibFunc::extenddftf2);
  } else if constexpr (requires { &Derived::fp80_load; }) {
    auto src_ref = this->val_ref(src_val);
    if (src_ty->isFloatTy() && dst_ty->isX86_FP80Ty()) {
      derived()->fp80_ext_float(src_ref.part(0), res_vr.part(0));
    } else if (src_ty->isDoubleTy() && dst_ty->isX86_FP80Ty()) {
      derived()->fp80_ext_double(src_ref.part(0), res_vr.part(0));
    } else if (src_ty->isX86_FP80Ty() && dst_ty->isFloatTy()) {
      derived()->fp80_trunc_float(src_ref.part(0), res_vr.part(0));
    } else if (src_ty->isX86_FP80Ty() && dst_ty->isDoubleTy()) {
      derived()->fp80_trunc_double(src_ref.part(0), res_vr.part(0));
    } else {
      return false;
    }
  } else {
    return false;
  }

  if (sym.valid()) {
    derived()->create_helper_call({&src_val, 1}, &res_vr, sym);
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_float_to_int(
    const llvm::Instruction *inst, const ValInfo &, u64 flags) noexcept {
  bool sign = flags & 0b01;
  bool saturate = flags & 0b10;

  const llvm::Value *src_val = inst->getOperand(0);
  auto *src_ty = src_val->getType();
  if (src_ty->isVectorTy()) {
    return false;
  }

  const auto bit_width = inst->getType()->getIntegerBitWidth();

  if (bit_width > 64) {
    return false;
  }

  unsigned ty_idx;
  switch (src_ty->getTypeID()) {
  case llvm::Type::FloatTyID: ty_idx = 0; break;
  case llvm::Type::DoubleTyID: ty_idx = 1; break;
  case llvm::Type::FP128TyID: {
    if (saturate) {
      return false;
    }

    LibFunc lf = sign ? LibFunc::fixtfdi : LibFunc::fixunstfdi;
    SymRef sym = get_libfunc_sym(lf);

    auto res_vr = this->result_ref(inst);
    derived()->create_helper_call({&src_val, 1}, &res_vr, sym);
    return true;
  }
  case llvm::Type::X86_FP80TyID:
    if (saturate) {
      return false;
    }
    if constexpr (requires { &Derived::fp80_to_int; }) {
      ValueRef src = this->val_ref(src_val);
      ValueRef res = this->result_ref(inst);
      derived()->fp80_to_int(sign, bit_width > 32, src.part(0), res.part(0));
      return true;
    }
    return false;
  default: return false;
  }

  using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ValuePart &&);
  static constexpr auto fns = []() {
    // fns[is_double][dst64][sign][sat]
    std::array<EncodeFnTy[2][2][2], 2> fns{};
    fns[0][0][0][0] = &Derived::encode_f32tou32;
    fns[0][0][0][1] = &Derived::encode_f32tou32_sat;
    fns[0][0][1][0] = &Derived::encode_f32toi32;
    fns[0][0][1][1] = &Derived::encode_f32toi32_sat;
    fns[0][1][0][0] = &Derived::encode_f32tou64;
    fns[0][1][0][1] = &Derived::encode_f32tou64_sat;
    fns[0][1][1][0] = &Derived::encode_f32toi64;
    fns[0][1][1][1] = &Derived::encode_f32toi64_sat;
    fns[1][0][0][0] = &Derived::encode_f64tou32;
    fns[1][0][0][1] = &Derived::encode_f64tou32_sat;
    fns[1][0][1][0] = &Derived::encode_f64toi32;
    fns[1][0][1][1] = &Derived::encode_f64toi32_sat;
    fns[1][1][0][0] = &Derived::encode_f64tou64;
    fns[1][1][0][1] = &Derived::encode_f64tou64_sat;
    fns[1][1][1][0] = &Derived::encode_f64toi64;
    fns[1][1][1][1] = &Derived::encode_f64toi64_sat;
    return fns;
  }();
  EncodeFnTy fn = fns[ty_idx][bit_width > 32][sign][saturate];

  if (saturate && bit_width % 32 != 0) {
    // TODO: clamp result to smaller integer bounds
    return false;
  }

  auto src_ref = this->val_ref(src_val);
  auto res_ref = this->result_ref(inst);
  return (derived()->*fn)(src_ref.part(0), res_ref.part(0));
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_int_to_float(
    const llvm::Instruction *inst, const ValInfo &val_info, u64 sign) noexcept {
  const llvm::Value *src_val = inst->getOperand(0);
  auto *dst_ty = inst->getType();
  if (dst_ty->isVectorTy()) {
    return false;
  }

  auto bit_width = src_val->getType()->getIntegerBitWidth();
  if (bit_width > 64) {
    return false;
  }

  if (dst_ty->isFP128Ty()) {
    LibFunc lf;
    if (bit_width == 32) {
      lf = sign ? LibFunc::floatsitf : LibFunc::floatunsitf;
    } else if (bit_width == 64) {
      lf = sign ? LibFunc::floatditf : LibFunc::floatunditf;
    } else {
      // TODO: extend, but create_helper_call currently takes only an IRValueRef
      return false;
    }

    SymRef sym = get_libfunc_sym(lf);

    auto res_vr = this->result_ref(inst);
    derived()->create_helper_call({&src_val, 1}, &res_vr, sym);
    return true;
  }

  ValueRef src_ref = this->val_ref(src_val);
  ValuePartRef src_op = src_ref.part(0);
  ValueRef res = this->result_ref(inst);

  if (bit_width != 32 && bit_width != 64) {
    unsigned ext = tpde::util::align_up(bit_width, 32);
    src_op = std::move(src_op).into_extended(sign, bit_width, ext);
  }

  unsigned ty_idx;
  switch (val_info.type) {
  case LLVMBasicValType::f32: ty_idx = 0; break;
  case LLVMBasicValType::f64: ty_idx = 1; break;
  case LLVMBasicValType::f80:
    if constexpr (requires { &Derived::fp80_from_int; }) {
      derived()->fp80_from_int(
          sign, bit_width > 32, std::move(src_op), res.part(0));
      return true;
    }
    return false;
  default: return false;
  }

  using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ValuePart &&);
  static constexpr auto encode_fns = []() consteval {
    std::array<EncodeFnTy[2][2], 2> res;
    res[0][0][0] = &Derived::encode_i32tof32;
    res[0][0][1] = &Derived::encode_i32tof64;
    res[0][1][0] = &Derived::encode_i64tof32;
    res[0][1][1] = &Derived::encode_i64tof64;
    res[1][0][0] = &Derived::encode_u32tof32;
    res[1][0][1] = &Derived::encode_u32tof64;
    res[1][1][0] = &Derived::encode_u64tof32;
    res[1][1][1] = &Derived::encode_u64tof64;
    return res;
  }();
  EncodeFnTy fn = encode_fns[!sign][bit_width > 32][ty_idx];
  (derived()->*fn)(std::move(src_op), res.part(0));
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_int_trunc(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  llvm::Value *src = inst->getOperand(0);
  ValueRef res_vr = this->result_ref(inst);
  ValueRef src_vr = this->val_ref(src);

  LLVMBasicValType bvt = val_info.type;
  switch (bvt) {
    using enum LLVMBasicValType;
  case i8:
  case i16:
  case i32:
  case i64:
    // no-op, users will extend anyways. When truncating an i128, the first part
    // contains the lowest bits.
    res_vr.part(0).set_value(src_vr.part(0));
    return true;
  case v8i1:
  case v16i1:
  case v32i1:
  case v64i1: {
    // Cast to i1 vector.
    // TODO: support illegal vector types. This is not trivial and requires bit
    // packing of the individual comparison results.
    auto [ty, ty_idx] = this->adaptor->lower_type(src);

    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ValuePart &&);
    EncodeFnTy encode_fn;
    switch (ty) {
    case v8i8: encode_fn = &Derived::encode_trunc_v8i8_1; break;
    case v4i16: encode_fn = &Derived::encode_trunc_v4i16_1; break;
    case v2i32: encode_fn = &Derived::encode_trunc_v2i32_1; break;
    case v16i8: encode_fn = &Derived::encode_trunc_v16i8_1; break;
    case v8i16: encode_fn = &Derived::encode_trunc_v8i16_1; break;
    case v4i32: encode_fn = &Derived::encode_trunc_v4i32_1; break;
    case v2i64: encode_fn = &Derived::encode_trunc_v2i64_1; break;
    default: return false;
    }
    return (derived()->*encode_fn)(src_vr.part(0), res_vr.part(0));
  }
  case v8i8:
  case v4i16:
  case v2i32: {
    auto dst_width = inst->getType()->getScalarType()->getIntegerBitWidth();
    auto src_width = src->getType()->getScalarType()->getIntegerBitWidth();
    // With the currently legal vector types, we only support halving vectors.
    if (dst_width * 2 != src_width) {
      return false;
    }

    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ValuePart &&);
    EncodeFnTy encode_fn = nullptr;
    switch (dst_width) {
    case 8: encode_fn = &Derived::encode_trunc_v8i16_8; break;
    case 16: encode_fn = &Derived::encode_trunc_v4i32_16; break;
    case 32: encode_fn = &Derived::encode_trunc_v2i64_32; break;
    default: return false;
    }
    return (derived()->*encode_fn)(src_vr.part(0), res_vr.part(0));
  }
  default: return false;
  }
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_int_ext(
    const llvm::Instruction *inst, const ValInfo &, u64 sign) noexcept {
  if (!inst->getType()->isIntegerTy()) {
    return false;
  }

  auto *src_val = inst->getOperand(0);

  unsigned src_width = src_val->getType()->getIntegerBitWidth();
  unsigned dst_width = inst->getType()->getIntegerBitWidth();
  assert(dst_width >= src_width);

  auto src_ref = this->val_ref(src_val);

  ValuePartRef low = src_ref.part(0);
  if (src_width < 64) {
    unsigned ext_width = dst_width <= 64 ? dst_width : 64;
    low = std::move(low).into_extended(sign, src_width, ext_width);
  } else if (src_width > 64) {
    return false;
  }

  auto res = this->result_ref(inst);

  if (dst_width == 128) {
    auto res_ref_high = res.part(1);

    if (sign) {
      if (!low.has_reg()) {
        low.load_to_reg();
      }
      derived()->encode_fill_with_sign64(low.get_unowned_ref(), res_ref_high);
    } else {
      res_ref_high.set_value(ValuePart{u64{0}, 8, res_ref_high.bank()});
    }
  }

  res.part(0).set_value(std::move(low));
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_ptr_to_int(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  ValueRef src = this->val_ref(inst->getOperand(0));
  ValueRef res = this->result_ref(inst);

  llvm::Type *dst_ty = inst->getType();
  if (dst_ty->isIntegerTy() || dst_ty->getScalarType()->isIntegerTy(64)) {
    // Just copy all parts, this is a no-op. For integers, the upper bits are
    // undefined, so no actual truncation needs to be done.
    auto part_count = this->adaptor->val_parts(val_info).count();
    for (u32 i = 0; i != part_count; ++i) {
      res.part(i).set_value(src.part(i));
    }
    return true;
  }

  // TODO: implement vector ptrtoint with truncation.
  // Might simply reuse compile_trunc for this.
  return false;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_int_to_ptr(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  if (!inst->getType()->isPointerTy()) {
    return false;
  }

  // zero-extend the value
  auto *src_val = inst->getOperand(0);
  const auto bit_width = src_val->getType()->getIntegerBitWidth();

  auto src_ref = this->val_ref(src_val);
  auto [res_vr, res_ref] = this->result_ref_single(inst);
  if (bit_width == 64) {
    // no-op
    res_ref.set_value(src_ref.part(0));
    return true;
  } else if (bit_width < 64) {
    // zero-extend
    res_ref.set_value(src_ref.part(0).into_extended(false, bit_width, 64));
    return true;
  }

  return false;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_bitcast(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  const auto src = inst->getOperand(0);
  ValueRef src_ref = this->val_ref(src);
  ValueRef res_ref = this->result_ref(inst);

  const auto src_part_count = this->adaptor->val_parts(src).count();
  const auto dst_part_count = this->adaptor->val_parts(val_info).count();
  // bitcast only support scalar and vector types of the same size. Multi-part
  // values must be of homogeneous element type.
  if (src_part_count == dst_part_count) {
    for (u32 i = 0; i != dst_part_count; ++i) {
      ValuePartRef res_vpr = res_ref.part(i);
      ValuePartRef src_vpr = src_ref.part(i);
      assert(src_vpr.part_size() == res_vpr.part_size());
      if (src_vpr.bank() == res_vpr.bank()) {
        res_vpr.set_value(std::move(src_vpr));
      } else {
        AsmReg src_reg = src_vpr.load_to_reg();
        derived()->mov(res_vpr.alloc_reg(), src_reg, res_vpr.part_size());
      }
    }
    return true;
  }

  // In-memory bitcast.
  this->allocate_spill_slot(tpde::AssignmentPartRef{res_ref.assignment(), 0});
  for (u32 i = 0; i != dst_part_count; ++i) {
    tpde::AssignmentPartRef ap{res_ref.assignment(), i};
    ap.set_stack_valid();
  }

  i32 frame_off = tpde::AssignmentPartRef{res_ref.assignment(), 0}.frame_off();
  for (u32 i = 0, n = src_part_count; i != n; ++i) {
    ValuePartRef src_vpr = src_ref.part(i);
    u32 part_size = src_vpr.part_size();
    derived()->spill_reg(src_vpr.load_to_reg(), frame_off, part_size);
    frame_off += part_size;
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_extract_value(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *extract = llvm::cast<llvm::ExtractValueInst>(inst);
  auto src = extract->getAggregateOperand();

  auto [first_part, last_part] =
      this->adaptor->complex_part_for_index(src, extract->getIndices());

  auto src_ref = this->val_ref(src);
  auto res_ref = this->result_ref(extract);
  for (unsigned i = first_part; i <= last_part; i++) {
    res_ref.part(i - first_part).set_value(src_ref.part(i));
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_insert_value(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  const auto *insert = llvm::cast<llvm::InsertValueInst>(inst);
  auto agg = insert->getAggregateOperand();
  auto ins = insert->getInsertedValueOperand();

  unsigned part_count = this->adaptor->val_parts(val_info).count();
  auto [first_part, last_part] =
      this->adaptor->complex_part_for_index(insert, insert->getIndices());

  ValueRef agg_ref = this->val_ref(agg);
  ValueRef ins_ref = this->val_ref(ins);
  if (agg_ref.is_owned()) {
    ValueRef res_ref = this->result_ref_alias(insert, std::move(agg_ref));
    for (unsigned i = first_part; i <= last_part; i++) {
      res_ref.part(i).set_value(ins_ref.part(i - first_part));
    }
    return true;
  }

  ValueRef res_ref = this->result_ref(insert);
  for (unsigned i = 0; i < part_count; i++) {
    ValuePartRef val_ref{this};
    if (i >= first_part && i <= last_part) {
      val_ref = ins_ref.part(i - first_part);
    } else {
      val_ref = agg_ref.part(i);
    }
    res_ref.part(i).set_value(std::move(val_ref));
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
void LLVMCompilerBase<Adaptor, Derived, Config>::extract_element(
    ValueRef &vec_vr,
    unsigned idx,
    LLVMBasicValType ty,
    ValuePart &out) noexcept {
  if (!vec_vr.has_assignment()) {
    // Constant.
    auto *cst = llvm::cast<llvm::Constant>(vec_vr.state.s.value);
    assert(cst->getType()->isVectorTy());
    auto *elem = cst->getAggregateElement(idx);
    out.set_value(this, val_ref_constant(elem, 0));
    return;
  }

  tpde::ValueAssignment *va = vec_vr.assignment();
  u32 elem_sz = this->adaptor->basic_ty_part_size(ty);

  if (elem_sz == va->max_part_size) {
    // Scalarized vector: simply take part idx.
    out.set_value(this, vec_vr.part(idx));
    return;
  }

  // Offset inside whole vector
  u32 vector_off = idx * elem_sz;
  // A vector can consist of multiple, equally sized parts.
  u32 part = vector_off / va->max_part_size;
  u32 off_in_part = vector_off % va->max_part_size;
  assert(part < va->part_count);

  this->spill({va, part});
  GenericValuePart addr = derived()->val_spill_slot({va, part});
  auto &expr = std::get<typename GenericValuePart::Expr>(addr.state);
  expr.disp += off_in_part;

  switch (ty) {
    using enum LLVMBasicValType;
  case i8: derived()->encode_loadi8_zext(std::move(addr), out); break;
  case i16: derived()->encode_loadi16_zext(std::move(addr), out); break;
  case i32: derived()->encode_loadi32_zext(std::move(addr), out); break;
  case i64:
  case ptr: derived()->encode_loadi64(std::move(addr), out); break;
  case f32: derived()->encode_loadf32(std::move(addr), out); break;
  case f64: derived()->encode_loadf64(std::move(addr), out); break;
  default: TPDE_UNREACHABLE("unexpected vector element type");
  }
}

template <typename Adaptor, typename Derived, typename Config>
void LLVMCompilerBase<Adaptor, Derived, Config>::insert_element(
    ValueRef &vec_vr,
    unsigned idx,
    LLVMBasicValType ty,
    GenericValuePart &&el) noexcept {
  tpde::ValueAssignment *va = vec_vr.assignment();
  u32 elem_sz = this->adaptor->basic_ty_part_size(ty);

  if (ty == LLVMBasicValType::i1) {
    assert(vec_vr.assignment()->part_count == 1);
    assert(idx < 64);
    ValuePartRef vec_part = vec_vr.part_unowned(0);
    if (!vec_part.assignment().stack_valid() &&
        !vec_part.assignment().register_valid()) {
      AsmReg dst_reg = vec_part.alloc_reg(); // Uninitialized
      derived()->generate_raw_bfiz(dst_reg, derived()->gval_as_reg(el), idx, 1);
    } else {
      AsmReg dst_reg = vec_part.load_to_reg();
      derived()->generate_raw_bfi(dst_reg, derived()->gval_as_reg(el), idx, 1);
    }
    return;
  }

  // TODO: optimize for scalarized vectors, reuse el as new value part.

  // Offset inside whole vector
  u32 vector_off = idx * elem_sz;
  // A vector can consist of multiple, equally sized parts.
  u32 part = vector_off / va->max_part_size;
  u32 off_in_part = vector_off % va->max_part_size;
  assert(part < va->part_count);

  tpde::AssignmentPartRef ap{va, part};
  if (ap.register_valid()) {
    this->evict(ap);
  } else if (!ap.stack_valid()) {
    // Value part is uninitialized
    this->allocate_spill_slot(ap);
    ap.set_stack_valid();
  }

  GenericValuePart addr = derived()->val_spill_slot(ap);
  auto &expr = std::get<typename GenericValuePart::Expr>(addr.state);
  expr.disp += off_in_part;

  switch (ty) {
    using enum LLVMBasicValType;
  case i8: derived()->encode_storei8(std::move(addr), std::move(el)); break;
  case i16: derived()->encode_storei16(std::move(addr), std::move(el)); break;
  case i32: derived()->encode_storei32(std::move(addr), std::move(el)); break;
  case i64:
  case ptr: derived()->encode_storei64(std::move(addr), std::move(el)); break;
  case f32: derived()->encode_storef32(std::move(addr), std::move(el)); break;
  case f64: derived()->encode_storef64(std::move(addr), std::move(el)); break;
  default: TPDE_UNREACHABLE("unexpected vector element type");
  }
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_extract_element(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  llvm::Value *src = inst->getOperand(0);
  llvm::Value *index = inst->getOperand(1);

  // TODO: support extractelement from constant vectors.
  if (llvm::isa<llvm::Constant>(src)) {
    return false;
  }

  ValueRef vec_vr = this->val_ref(src);
  auto [res_vr, result] = this->result_ref_single(inst);

  if (inst->getType()->isIntegerTy(1)) {
    // i1 vectors are integers, extracting an element is just a shift.
    assert(this->adaptor->val_parts(src).count() == 1);
    // index 0 extract occurs so often that is it worth to have a special case.
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(index);
        ci && ci->isZeroValue()) {
      result.set_value(vec_vr.part(0));
    } else {
      derived()->encode_shri64(
          vec_vr.part(0), this->val_ref(index).part(0), result);
    }
    return true;
  }

  auto *vec_ty = llvm::cast<llvm::FixedVectorType>(src->getType());
  unsigned nelem = vec_ty->getNumElements();
  assert(index->getType()->getIntegerBitWidth() >= 8);
  LLVMBasicValType bvt = val_info.type;

  if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
    unsigned cidx = ci->getZExtValue();
    cidx = cidx < nelem ? cidx : 0;
    derived()->extract_element(vec_vr, cidx, bvt, result);
    return true;
  }

  // TODO: deduplicate with code above somehow?
  // First, copy value into the spill slot.
  for (unsigned i = 0; i < vec_vr.assignment()->part_count; ++i) {
    this->spill(tpde::AssignmentPartRef{vec_vr.assignment(), i});
  }

  // Second, create address. Mask index, out-of-bounds access are just poison.
  ValuePartRef idx_scratch{this, Config::GP_BANK};
  GenericValuePart addr = derived()->val_spill_slot({vec_vr.assignment(), 0});
  auto &expr = std::get<typename GenericValuePart::Expr>(addr.state);
  if ((nelem & (nelem - 1)) == 0) {
    u64 mask = nelem - 1;
    derived()->encode_landi64(this->val_ref(index).part(0),
                              ValuePartRef{this, mask, 8, Config::GP_BANK},
                              idx_scratch);
  } else {
    derived()->encode_uremi64(this->val_ref(index).part(0),
                              ValuePartRef{this, nelem, 8, Config::GP_BANK},
                              idx_scratch);
  }
  assert(expr.scale == 0);
  expr.scale = this->adaptor->basic_ty_part_size(bvt);
  expr.index = std::move(idx_scratch).into_scratch();

  // Third, do the load.
  switch (bvt) {
    using enum LLVMBasicValType;
  case i8: derived()->encode_loadi8_zext(std::move(addr), result); break;
  case i16: derived()->encode_loadi16_zext(std::move(addr), result); break;
  case i32: derived()->encode_loadi32_zext(std::move(addr), result); break;
  case i64:
  case ptr: derived()->encode_loadi64(std::move(addr), result); break;
  case f32: derived()->encode_loadf32(std::move(addr), result); break;
  case f64: derived()->encode_loadf64(std::move(addr), result); break;
  default: TPDE_UNREACHABLE("unexpected vector element type");
  }
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_insert_element(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  llvm::Value *index = inst->getOperand(2);

  auto *vec_ty = llvm::cast<llvm::FixedVectorType>(inst->getType());
  unsigned nelem = vec_ty->getNumElements();
  if (index->getType()->getIntegerBitWidth() < 8) {
    return false;
  }

  auto ins = inst->getOperand(1);
  auto [val_ref, val] = this->val_ref_single(ins);
  ValueRef res_vr{derived()};

  if (ins->getType()->isIntegerTy(1)) {
    res_vr = this->result_ref(inst);
    assert(res_vr.assignment()->part_count == 1);
    ValueRef src_vr = this->val_ref(inst->getOperand(0));
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
      res_vr.part(0).set_value(src_vr.part(0));
      unsigned cidx = ci->getZExtValue() < nelem ? ci->getZExtValue() : 0;
      LLVMBasicValType bvt = LLVMBasicValType::i1;
      derived()->insert_element(res_vr, cidx, bvt, std::move(val));
      return true;
    }

    derived()->encode_insert_vi1(src_vr.part(0),
                                 this->val_ref(index).part(0),
                                 std::move(val),
                                 res_vr.part(0));
    return true;
  }

  auto [bvt, _] = this->adaptor->lower_type(ins->getType());
  assert(bvt != LLVMBasicValType::complex);

  // We do the dynamic insert in the spill slot of result.
  // TODO: reuse spill slot of vec_ref if possible.

  // First, copy value into the result. We must also do this for constant
  // indices, because the value reference must always be initialized.
  {
    ValueRef src_vr = this->val_ref(inst->getOperand(0));
    if (src_vr.is_owned()) {
      res_vr = this->result_ref_alias(inst, std::move(src_vr));
    } else {
      res_vr = this->result_ref(inst);
      for (u32 i = 0; i < res_vr.assignment()->part_count; ++i) {
        // TODO: skip overwritten part in scalarized case.
        res_vr.part(i).set_value(src_vr.part(i));
      }
    }
  }

  if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
    unsigned cidx = ci->getZExtValue();
    cidx = cidx < nelem ? cidx : 0;
    derived()->insert_element(res_vr, cidx, bvt, std::move(val));
    // No need for ref counting: all operands and results were ValuePartRefs.
    return true;
  }

  // Evict, because we will overwrite the value in the stack slot.
  for (unsigned i = 0; i < res_vr.assignment()->part_count; ++i) {
    tpde::AssignmentPartRef ap{res_vr.assignment(), i};
    if (ap.register_valid()) {
      this->evict(ap);
    }
  }

  // Second, create address. Mask index, out-of-bounds access are just poison.
  ValuePartRef idx_scratch{this, Config::GP_BANK};
  GenericValuePart addr = derived()->val_spill_slot({res_vr.assignment(), 0});
  auto &expr = std::get<typename GenericValuePart::Expr>(addr.state);
  if ((nelem & (nelem - 1)) == 0) {
    u64 mask = nelem - 1;
    derived()->encode_landi64(this->val_ref(index).part(0),
                              ValuePartRef{this, mask, 8, Config::GP_BANK},
                              idx_scratch);
  } else {
    derived()->encode_uremi64(this->val_ref(index).part(0),
                              ValuePartRef{this, nelem, 8, Config::GP_BANK},
                              idx_scratch);
  }
  assert(expr.scale == 0);
  expr.scale = val.part_size();
  expr.index = std::move(idx_scratch).into_scratch();

  // Third, do the store.
  switch (bvt) {
    using enum LLVMBasicValType;
  case i8: derived()->encode_storei8(std::move(addr), std::move(val)); break;
  case i16: derived()->encode_storei16(std::move(addr), std::move(val)); break;
  case i32: derived()->encode_storei32(std::move(addr), std::move(val)); break;
  case i64:
  case ptr: derived()->encode_storei64(std::move(addr), std::move(val)); break;
  case f32: derived()->encode_storef32(std::move(addr), std::move(val)); break;
  case f64: derived()->encode_storef64(std::move(addr), std::move(val)); break;
  default: TPDE_UNREACHABLE("unexpected vector element type");
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_shuffle_vector(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *shuffle = llvm::cast<llvm::ShuffleVectorInst>(inst);
  llvm::Value *lhs = shuffle->getOperand(0);
  llvm::Value *rhs = shuffle->getOperand(1);

  auto *dst_ty = llvm::cast<llvm::FixedVectorType>(shuffle->getType());
  auto *src_ty = llvm::cast<llvm::FixedVectorType>(lhs->getType());
  unsigned dst_nelem = dst_ty->getNumElements();
  unsigned src_nelem = src_ty->getNumElements();

  // TODO: deduplicate with adaptor
  LLVMBasicValType bvt;
  assert(dst_ty->getElementType() == src_ty->getElementType());
  auto *elem_ty = dst_ty->getElementType();
  if (elem_ty->isFloatTy()) {
    bvt = LLVMBasicValType::f32;
  } else if (elem_ty->isDoubleTy()) {
    bvt = LLVMBasicValType::f64;
  } else if (elem_ty->isIntegerTy(8)) {
    bvt = LLVMBasicValType::i8;
  } else if (elem_ty->isIntegerTy(16)) {
    bvt = LLVMBasicValType::i16;
  } else if (elem_ty->isIntegerTy(32)) {
    bvt = LLVMBasicValType::i32;
  } else if (elem_ty->isIntegerTy(64) || elem_ty->isPointerTy()) {
    bvt = LLVMBasicValType::i64;
  } else {
    // E.g., i1 vectors.
    return false;
  }

  auto bank = this->adaptor->basic_ty_part_bank(bvt);
  auto size = this->adaptor->basic_ty_part_size(bvt);

  ValueRef lhs_vr = this->val_ref(lhs);
  ValueRef rhs_vr = this->val_ref(rhs);
  ValueRef res_vr = this->result_ref(inst);

  ValuePartRef tmp{this, bank};
  llvm::ArrayRef<int> mask = shuffle->getShuffleMask();
  for (unsigned i = 0; i < dst_nelem; i++) {
    if (mask[i] == llvm::PoisonMaskElem) {
      continue;
    }
    bool src_is_lhs = unsigned(mask[i]) < src_nelem;
    if (auto *cst = llvm::dyn_cast<llvm::Constant>(src_is_lhs ? lhs : rhs)) {
      auto *cst_elem = cst->getAggregateElement(mask[i] % src_nelem);
      u64 const_elem;
      if (llvm::isa<llvm::PoisonValue>(cst_elem)) {
        continue;
      } else if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(cst_elem)) {
        const_elem = ci->getZExtValue();
      } else if (auto *cfp = llvm::dyn_cast<llvm::ConstantFP>(cst_elem)) {
        const_elem = cfp->getValue().bitcastToAPInt().getZExtValue();
      } else if (llvm::isa<llvm::ConstantPointerNull>(cst_elem)) {
        const_elem = 0;
      } else {
        TPDE_UNREACHABLE("invalid constant element type");
      }
      ValuePartRef const_ref{this, &const_elem, size, bank};
      derived()->insert_element(res_vr, i, bvt, std::move(const_ref));
    } else {
      ValueRef src_vr = (src_is_lhs ? lhs_vr : rhs_vr).disowned();
      derived()->extract_element(src_vr, mask[i] % src_nelem, bvt, tmp);
      derived()->insert_element(res_vr, i, bvt, std::move(tmp));
    }
  }

  // Make sure that all parts are initialized.
  // TODO: maybe remove assertion that no value is uninitialized?
  for (u32 i = 0, n = res_vr.assignment()->part_count; i != n; ++i) {
    tpde::AssignmentPartRef ap{res_vr.assignment(), i};
    if (!ap.register_valid() && !ap.stack_valid()) {
      // Value part is uninitialized
      this->allocate_spill_slot(ap);
      ap.set_stack_valid();
    }
  }
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_icmp_vector(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ValuePart &&);
  // fns[pred][type][bitvec=0/mask=1]
  static constexpr auto fns = []() constexpr {
    constexpr unsigned NumPreds = llvm::ICmpInst::LAST_ICMP_PREDICATE -
                                  llvm::ICmpInst::FIRST_ICMP_PREDICATE + 1;
    std::array<EncodeFnTy[9][3], NumPreds> res{};
    auto entry = [&res](llvm::ICmpInst::Predicate pred) {
      return res[pred - llvm::ICmpInst::FIRST_ICMP_PREDICATE];
    };

    // TODO: more consistent naming of encode functions
#define FN_ENTRY(predval, predname, sign)                                      \
  entry(predval)[0][0] = &Derived::encode_icmp_##predname##v8##sign##8;        \
  entry(predval)[0][1] = &Derived::encode_icmpmask_##predname##v8##sign##8;    \
  entry(predval)[0][2] = &Derived::encode_icmpset_##predname##v8##sign##8;     \
  entry(predval)[1][0] = &Derived::encode_icmp_##predname##v4##sign##16;       \
  entry(predval)[1][1] = &Derived::encode_icmpmask_##predname##v4##sign##16;   \
  entry(predval)[1][2] = &Derived::encode_icmpset_##predname##v4##sign##16;    \
  entry(predval)[2][0] = &Derived::encode_icmp_##predname##v2##sign##32;       \
  entry(predval)[2][1] = &Derived::encode_icmpmask_##predname##v2##sign##32;   \
  entry(predval)[2][2] = &Derived::encode_icmpset_##predname##v2##sign##32;    \
  entry(predval)[3][0] = &Derived::encode_icmp_##predname##v16##sign##8;       \
  entry(predval)[3][1] = &Derived::encode_icmpmask_##predname##v16##sign##8;   \
  entry(predval)[3][2] = &Derived::encode_icmpset_##predname##v16##sign##8;    \
  entry(predval)[4][0] = &Derived::encode_icmp_##predname##v8##sign##16;       \
  entry(predval)[4][1] = &Derived::encode_icmpmask_##predname##v8##sign##16;   \
  entry(predval)[4][2] = &Derived::encode_icmpset_##predname##v8##sign##16;    \
  entry(predval)[5][0] = &Derived::encode_icmp_##predname##v4##sign##32;       \
  entry(predval)[5][1] = &Derived::encode_icmpmask_##predname##v4##sign##32;   \
  entry(predval)[5][2] = &Derived::encode_icmpset_##predname##v4##sign##32;    \
  entry(predval)[6][0] = &Derived::encode_icmp_##predname##v2##sign##64;       \
  entry(predval)[6][1] = &Derived::encode_icmpmask_##predname##v2##sign##64;   \
  entry(predval)[6][2] = &Derived::encode_icmpset_##predname##v2##sign##64;    \
  entry(predval)[7][0] = &Derived::encode_icmp_##predname##i32;                \
  entry(predval)[7][1] = &Derived::encode_icmpmask_##predname##i32;            \
  entry(predval)[7][2] = &Derived::encode_icmpset_##predname##i32;             \
  entry(predval)[8][0] = &Derived::encode_icmp_##predname##i64;                \
  entry(predval)[8][1] = &Derived::encode_icmpmask_##predname##i64;            \
  entry(predval)[8][2] = &Derived::encode_icmpset_##predname##i64;

    FN_ENTRY(llvm::ICmpInst::ICMP_EQ, eq, u)
    FN_ENTRY(llvm::ICmpInst::ICMP_NE, ne, u)
    FN_ENTRY(llvm::ICmpInst::ICMP_UGT, ugt, u)
    FN_ENTRY(llvm::ICmpInst::ICMP_UGE, uge, u)
    FN_ENTRY(llvm::ICmpInst::ICMP_ULT, ult, u)
    FN_ENTRY(llvm::ICmpInst::ICMP_ULE, ule, u)
    FN_ENTRY(llvm::ICmpInst::ICMP_SGT, sgt, i)
    FN_ENTRY(llvm::ICmpInst::ICMP_SGE, sge, i)
    FN_ENTRY(llvm::ICmpInst::ICMP_SLT, slt, i)
    FN_ENTRY(llvm::ICmpInst::ICMP_SLE, sle, i)
#undef FN_ENTRY

    return res;
  }();

  const auto *icmp = llvm::cast<llvm::ICmpInst>(inst);
  llvm::Value *lhs = icmp->getOperand(0);
  llvm::Value *rhs = icmp->getOperand(1);

  u32 pred_idx = icmp->getPredicate() - llvm::ICmpInst::FIRST_ICMP_PREDICATE;

  auto [ty, _] = this->adaptor->lower_type(lhs);
  unsigned ty_idx;
  switch (ty) {
  case LLVMBasicValType::v8i8: ty_idx = 0; break;
  case LLVMBasicValType::v4i16: ty_idx = 1; break;
  case LLVMBasicValType::v2i32: ty_idx = 2; break;
  case LLVMBasicValType::v16i8: ty_idx = 3; break;
  case LLVMBasicValType::v8i16: ty_idx = 4; break;
  case LLVMBasicValType::v4i32: ty_idx = 5; break;
  case LLVMBasicValType::v2i64: ty_idx = 6; break;
  case LLVMBasicValType::complex: {
    auto *vec_ty = llvm::cast<llvm::FixedVectorType>(lhs->getType());
    unsigned nelem = vec_ty->getNumElements();
    auto [elem_ty, _] = this->adaptor->lower_type(vec_ty->getElementType());
    switch (elem_ty) {
    case LLVMBasicValType::i8:
    case LLVMBasicValType::i16:
      // TODO: support i1/i8/i16 vector icmp, needs element extension.
      return false;
    case LLVMBasicValType::i32: ty_idx = 7; break;
    case LLVMBasicValType::i64:
    case LLVMBasicValType::ptr: ty_idx = 8; break;
    default: TPDE_UNREACHABLE("unexpected legal icmp vector type");
    }

    EncodeFnTy fn = fns[pred_idx][ty_idx][0];

    ValueRef lhs_vr = this->val_ref(lhs);
    ValueRef rhs_vr = this->val_ref(rhs);
    ValueRef lhs_vr_disowned = lhs_vr.disowned();
    ValueRef rhs_vr_disowned = rhs_vr.disowned();
    ValueRef res = this->result_ref(inst);

    ValuePartRef tmp{this, Config::GP_BANK};
    ValuePartRef e_lhs{this, LLVMAdaptor::basic_ty_part_bank(elem_ty)};
    ValuePartRef e_rhs{this, LLVMAdaptor::basic_ty_part_bank(elem_ty)};
    for (unsigned i = 0; i != nelem; i++) {
      derived()->extract_element(lhs_vr_disowned, i, elem_ty, e_lhs);
      derived()->extract_element(rhs_vr_disowned, i, elem_ty, e_rhs);
      (derived()->*fn)(std::move(e_lhs), std::move(e_rhs), std::move(tmp));
      derived()->insert_element(res, i, LLVMBasicValType::i1, std::move(tmp));
    }
    return true;
  }
  default: return false;
  }

  const llvm::Instruction *fuse_ext = nullptr;
  u32 res_type_idx = 0;
  if (icmp->hasNUses(1) && *icmp->user_begin() == icmp->getNextNode()) {
    auto *fuse_inst = icmp->getNextNode();
    if (llvm::isa<llvm::SExtInst, llvm::ZExtInst>(fuse_inst) &&
        fuse_inst->getType() == lhs->getType()) {
      fuse_ext = fuse_inst;
      res_type_idx = llvm::isa<llvm::ZExtInst>(fuse_ext) ? 2 : 1;
      this->adaptor->inst_set_fused(fuse_ext, true);
    }
  }

  EncodeFnTy encode_fn = fns[pred_idx][ty_idx][res_type_idx];
  auto lhs_vr = this->val_ref(lhs);
  auto rhs_vr = this->val_ref(rhs);
  auto res = this->result_ref(fuse_ext ? fuse_ext : inst);
  return (derived()->*encode_fn)(lhs_vr.part(0), rhs_vr.part(0), res.part(0));
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_cmpxchg(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *cmpxchg = llvm::cast<llvm::AtomicCmpXchgInst>(inst);
  auto *new_val = cmpxchg->getNewValOperand();
  auto *val_ty = new_val->getType();
  unsigned width = 64;
  if (val_ty->isIntegerTy()) {
    width = val_ty->getIntegerBitWidth();
    // LLVM only permits widths that are a power of two and >= 8.
    if (width > 64) {
      return false;
    }
  }

  unsigned width_idx;
  switch (width) {
  case 8: width_idx = 0; break;
  case 16: width_idx = 1; break;
  case 32: width_idx = 2; break;
  case 64: width_idx = 3; break;
  default: return false;
  }

  // ptr, cmp, new_val, old_val, success
  using EncodeFnTy = bool (Derived::*)(GenericValuePart &&,
                                       GenericValuePart &&,
                                       GenericValuePart &&,
                                       ValuePart &&,
                                       ValuePart &&);
  static constexpr auto fns = []() constexpr {
    using enum llvm::AtomicOrdering;
    std::array<EncodeFnTy[size_t(LAST) + 1], 4> res{};
    res[0][u32(Monotonic)] = &Derived::encode_cmpxchg_u8_monotonic;
    res[1][u32(Monotonic)] = &Derived::encode_cmpxchg_u16_monotonic;
    res[2][u32(Monotonic)] = &Derived::encode_cmpxchg_u32_monotonic;
    res[3][u32(Monotonic)] = &Derived::encode_cmpxchg_u64_monotonic;
    res[0][u32(Acquire)] = &Derived::encode_cmpxchg_u8_acquire;
    res[1][u32(Acquire)] = &Derived::encode_cmpxchg_u16_acquire;
    res[2][u32(Acquire)] = &Derived::encode_cmpxchg_u32_acquire;
    res[3][u32(Acquire)] = &Derived::encode_cmpxchg_u64_acquire;
    res[0][u32(Release)] = &Derived::encode_cmpxchg_u8_release;
    res[1][u32(Release)] = &Derived::encode_cmpxchg_u16_release;
    res[2][u32(Release)] = &Derived::encode_cmpxchg_u32_release;
    res[3][u32(Release)] = &Derived::encode_cmpxchg_u64_release;
    res[0][u32(AcquireRelease)] = &Derived::encode_cmpxchg_u8_acqrel;
    res[1][u32(AcquireRelease)] = &Derived::encode_cmpxchg_u16_acqrel;
    res[2][u32(AcquireRelease)] = &Derived::encode_cmpxchg_u32_acqrel;
    res[3][u32(AcquireRelease)] = &Derived::encode_cmpxchg_u64_acqrel;
    res[0][u32(SequentiallyConsistent)] = &Derived::encode_cmpxchg_u8_seqcst;
    res[1][u32(SequentiallyConsistent)] = &Derived::encode_cmpxchg_u16_seqcst;
    res[2][u32(SequentiallyConsistent)] = &Derived::encode_cmpxchg_u32_seqcst;
    res[3][u32(SequentiallyConsistent)] = &Derived::encode_cmpxchg_u64_seqcst;
    return res;
  }();

  auto ptr_ref = this->val_ref(cmpxchg->getPointerOperand());
  auto cmp_ref = this->val_ref(cmpxchg->getCompareOperand());
  auto new_ref = this->val_ref(new_val);
  auto res = this->result_ref(cmpxchg);

  llvm::AtomicOrdering order = cmpxchg->getMergedOrdering();
  EncodeFnTy encode_fn = fns[width_idx][size_t(order)];
  assert(encode_fn && "invalid cmpxchg ordering");
  if (!(derived()->*encode_fn)(ptr_ref.part(0),
                               cmp_ref.part(0),
                               new_ref.part(0),
                               res.part(0),
                               res.part(1))) {
    return false;
  }

  // clang-format off
    // TODO(ts): fusing with subsequent extractvalues + br's
    // e.g. clang generates
    // %4 = cmpxchg ptr %0, i64 %3, i64 1 seq_cst seq_cst, align 8
    // %5 = extractvalue { i64, i1 } %4, 1
    // %6 = extractvalue { i64, i1 } %4, 0
    // br i1 %5, label %7, label %2, !llvm.loop !3
  // clang-format on

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_atomicrmw(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  const auto *rmw = llvm::cast<llvm::AtomicRMWInst>(inst);
  llvm::Type *ty = rmw->getType();
  unsigned size = this->adaptor->mod->getDataLayout().getTypeSizeInBits(ty);
  // This is checked by the IR verifier.
  assert(size >= 8 && (size & (size - 1)) == 0 && "invalid atomicrmw size");
  // Unaligned atomicrmw is very tricky to implement. While x86-64 supports
  // unaligned atomics, this is not always supported (to prevent split locks
  // from locking the bus in shared systems). Other platforms don't support
  // unaligned accesses at all. Therefore, supporting this needs to go through
  // library support from libgcc or compiler-rt with a cmpxchg loop.
  // TODO: implement support for unaligned atomics
  // TODO: do this check without consulting DataLayout
  if (rmw->getAlign().value() < size / 8) {
    TPDE_LOG_ERR("unaligned atomicrmw is not supported (align={} < size={})",
                 rmw->getAlign().value(),
                 size / 8);
    return false;
  }

  auto bvt = val_info.type;

  // TODO: implement non-seq_cst orderings more efficiently
  // TODO: use more efficient implementation when the result is not used. On
  // x86-64, the current implementation gives many cmpxchg loops.
  bool (Derived::*fn)(GenericValuePart &&, GenericValuePart &&, ValuePart &&) =
      nullptr;
  switch (rmw->getOperation()) {
  case llvm::AtomicRMWInst::Xchg:
    // TODO: support f32/f64
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_xchg_u8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_xchg_u16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_xchg_u32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_xchg_u64_seqcst; break;
    case ptr: fn = &Derived::encode_atomic_xchg_u64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::Add:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_add_u8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_add_u16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_add_u32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_add_u64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::Sub:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_sub_u8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_sub_u16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_sub_u32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_sub_u64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::And:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_and_u8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_and_u16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_and_u32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_and_u64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::Nand:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_nand_u8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_nand_u16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_nand_u32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_nand_u64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::Or:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_or_u8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_or_u16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_or_u32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_or_u64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::Xor:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_xor_u8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_xor_u16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_xor_u32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_xor_u64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::Min:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_min_i8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_min_i16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_min_i32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_min_i64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::Max:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_max_i8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_max_i16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_max_i32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_max_i64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::UMin:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_min_u8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_min_u16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_min_u32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_min_u64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::UMax:
    switch (bvt) {
      using enum LLVMBasicValType;
    case i8: fn = &Derived::encode_atomic_max_u8_seqcst; break;
    case i16: fn = &Derived::encode_atomic_max_u16_seqcst; break;
    case i32: fn = &Derived::encode_atomic_max_u32_seqcst; break;
    case i64: fn = &Derived::encode_atomic_max_u64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::FAdd:
    switch (bvt) {
      using enum LLVMBasicValType;
    case f32: fn = &Derived::encode_atomic_add_f32_seqcst; break;
    case f64: fn = &Derived::encode_atomic_add_f64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::FSub:
    switch (bvt) {
      using enum LLVMBasicValType;
    case f32: fn = &Derived::encode_atomic_sub_f32_seqcst; break;
    case f64: fn = &Derived::encode_atomic_sub_f64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::FMin:
    switch (bvt) {
      using enum LLVMBasicValType;
    case f32: fn = &Derived::encode_atomic_min_f32_seqcst; break;
    case f64: fn = &Derived::encode_atomic_min_f64_seqcst; break;
    default: return false;
    }
    break;
  case llvm::AtomicRMWInst::FMax:
    switch (bvt) {
      using enum LLVMBasicValType;
    case f32: fn = &Derived::encode_atomic_max_f32_seqcst; break;
    case f64: fn = &Derived::encode_atomic_max_f64_seqcst; break;
    default: return false;
    }
    break;
  default: return false;
  }

  auto ptr_ref = this->val_ref(rmw->getPointerOperand());
  auto val_ref = this->val_ref(rmw->getValOperand());
  auto res_ref = this->result_ref(rmw);
  return (derived()->*fn)(ptr_ref.part(0), val_ref.part(0), res_ref.part(0));
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_fence(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *fence = llvm::cast<llvm::FenceInst>(inst);
  if (fence->getSyncScopeID() == llvm::SyncScope::SingleThread) {
    // memory barrier only
    return true;
  }

  switch (fence->getOrdering()) {
    using enum llvm::AtomicOrdering;
  case Acquire: derived()->encode_fence_acq(); break;
  case Release: derived()->encode_fence_rel(); break;
  case AcquireRelease: derived()->encode_fence_acqrel(); break;
  case SequentiallyConsistent: derived()->encode_fence_seqcst(); break;
  default: return false;
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_freeze(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  // essentially a no-op
  auto src_ref = this->val_ref(inst->getOperand(0));
  auto res_ref = this->result_ref(inst);
  const auto part_count = res_ref.assignment()->part_count;
  for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
    res_ref.part(part_idx).set_value(src_ref.part(part_idx));
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_call(
    const llvm::Instruction *inst, const ValInfo &info, u64) noexcept {
  const auto *call = llvm::cast<llvm::CallBase>(inst);
  if (auto *intrin = llvm::dyn_cast<llvm::IntrinsicInst>(call)) {
    return compile_intrin(intrin, info);
  }

  if (call->isMustTailCall() || call->hasOperandBundles()) {
    return false;
  }

  if (call->isInlineAsm()) {
    return derived()->compile_inline_asm(call);
  }

  auto cb = derived()->create_call_builder(call);
  if (!cb) {
    return false;
  }

  for (u32 i = 0, num_args = call->arg_size(); i != num_args; ++i) {
    using CallArg = typename Derived::CallArg;

    auto *op = call->getArgOperand(i);
    CallArg arg{op};

    auto [ty, ty_idx] = this->adaptor->lower_type(op);
    this->adaptor->check_type_compatibility(op->getType(), ty, ty_idx);

    // paramHasAttr queries are expensive, so avoid them if possible. LLVM
    // provides no possibility to query the absence of all ABI-related
    // attributes (which is the common case) somewhat efficiently.
    switch (ty) {
    case LLVMBasicValType::ptr:
      if (call->paramHasAttr(i, llvm::Attribute::AttrKind::ByVal)) {
        arg.flag = CallArg::Flag::byval;
        auto &data_layout = this->adaptor->mod->getDataLayout();
        llvm::Type *byval_ty = call->getParamByValType(i);
        arg.byval_size = data_layout.getTypeAllocSize(byval_ty);

        if (auto param_align = call->getParamStackAlign(i)) {
          arg.byval_align = param_align->value();
        } else if (auto param_align = call->getParamAlign(i)) {
          arg.byval_align = param_align->value();
        } else {
          arg.byval_align = data_layout.getABITypeAlign(byval_ty).value();
        }
      } else if (call->paramHasAttr(i, llvm::Attribute::AttrKind::StructRet)) {
        arg.flag = CallArg::Flag::sret;
      }
      break;
    case LLVMBasicValType::i8:
    case LLVMBasicValType::i16:
    case LLVMBasicValType::i32:
    case LLVMBasicValType::i64:
      if (call->paramHasAttr(i, llvm::Attribute::AttrKind::ZExt)) {
        arg.flag = CallArg::Flag::zext;
        arg.ext_bits = op->getType()->getIntegerBitWidth();
      } else if (call->paramHasAttr(i, llvm::Attribute::AttrKind::SExt)) {
        arg.flag = CallArg::Flag::sext;
        arg.ext_bits = op->getType()->getIntegerBitWidth();
      }
      break;
    case LLVMBasicValType::i128: arg.byval_align = 16; break;
    case LLVMBasicValType::f80: {
      auto [vr, vpr] = this->val_ref_single(op);
      tpde::CCAssignment cca{
          .align = 16, .bank = tpde::RegBank(-2), .size = 16};
      cb->add_arg(std::move(vpr), cca);
      continue;
    }
    case LLVMBasicValType::complex:
      if (derived()->arg_allow_split_reg_stack_passing(op)) {
        arg.flag = CallArg::Flag::allow_split;
      }
      break;
    default: break;
    }
    assert(!call->paramHasAttr(i, llvm::Attribute::AttrKind::InAlloca));
    assert(!call->paramHasAttr(i, llvm::Attribute::AttrKind::Preallocated));

    // Explicitly pass part count to avoid duplicate type lowering.
    cb->add_arg(arg, this->adaptor->type_part_count(ty, ty_idx));
  }

  llvm::Value *target = call->getCalledOperand();
  if (auto *global = llvm::dyn_cast<llvm::GlobalValue>(target)) {
    cb->call(global_sym(global));
  } else {
    auto [_, tgt_vp] = this->val_ref_single(target);
    cb->call(std::move(tgt_vp));
  }

  if (!call->getType()->isVoidTy()) {
    ValueRef res = this->result_ref(call);
    if (call->getType()->isX86_FP80Ty()) [[unlikely]] {
      if constexpr (requires { &Derived::fp80_pop; }) {
        ValuePartRef res_vpr = res.part(0);
        derived()->fp80_pop(res_vpr);
      } else {
        return false;
      }
    } else {
      cb->add_ret(res);
    }
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_select(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  if (!inst->getOperand(0)->getType()->isIntegerTy()) {
    return false;
  }

  auto [cond_vr, cond] = this->val_ref_single(inst->getOperand(0));
  auto lhs = this->val_ref(inst->getOperand(1));
  auto rhs = this->val_ref(inst->getOperand(2));

  auto res = this->result_ref(inst);

  switch (val_info.type) {
    using enum LLVMBasicValType;
  case i1:
  case i8:
  case v8i1:
  case i16:
  case v16i1:
  case i32:
  case v32i1:
    derived()->encode_select_i32(
        std::move(cond), lhs.part(0), rhs.part(0), res.part(0));
    break;
  case i64:
  case v64i1:
  case ptr:
    derived()->encode_select_i64(
        std::move(cond), lhs.part(0), rhs.part(0), res.part(0));
    break;
  case f32:
    derived()->encode_select_f32(
        std::move(cond), lhs.part(0), rhs.part(0), res.part(0));
    break;
  case f64:
  case v8i8:
  case v4i16:
  case v2i32:
  case v2f32:
    derived()->encode_select_f64(
        std::move(cond), lhs.part(0), rhs.part(0), res.part(0));
    break;
  case f80: // x86_fp80 is mapped to XMM register, so we can reuse the logic.
  case f128:
  case v16i8:
  case v8i16:
  case v4i32:
  case v2i64:
  case v4f32:
  case v2f64:
    derived()->encode_select_v2u64(
        std::move(cond), lhs.part(0), rhs.part(0), res.part(0));
    break;
  case complex: {
    // Handle case of complex with two i64 as i128, this is extremely hacky...
    // TODO(ts): support full complex types using branches
    const auto parts = this->adaptor->val_parts(val_info);
    if (parts.count() != 2 || parts.reg_bank(0) != Config::GP_BANK ||
        parts.reg_bank(1) != Config::GP_BANK) {
      return false;
    }
  }
    [[fallthrough]];
  case i128: {
    derived()->encode_select_i128(std::move(cond),
                                  lhs.part(0),
                                  lhs.part(1),
                                  rhs.part(0),
                                  rhs.part(1),
                                  res.part(0),
                                  res.part(1));
    break;
  }
  default: TPDE_UNREACHABLE("invalid select basic type"); break;
  }
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_alloca(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *alloca = llvm::cast<llvm::AllocaInst>(inst);

  auto [res_vr, res_ref] = this->result_ref_single(alloca);

  auto align = alloca->getAlign().value();
  auto &layout = this->adaptor->mod->getDataLayout();
  if (auto size = alloca->getAllocationSize(layout)) {
    if (size->isScalable()) {
      return false;
    }
    derived()->alloca_fixed(size->getFixedValue(), align, res_ref);
    return true;
  }

  const llvm::Value *size_val = alloca->getArraySize();
  auto [size_vr, size_ref] = this->val_ref_single(size_val);

  auto elem_size = layout.getTypeAllocSize(alloca->getAllocatedType());
  if (auto width = size_val->getType()->getIntegerBitWidth(); width != 64) {
    if (width > 64) {
      // Don't support alloca array sizes beyond i64...
      return false;
    }
    size_ref = std::move(size_ref).into_extended(false, width, 64);
  }
  derived()->alloca_dynamic(elem_size, std::move(size_ref), align, res_ref);
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_gep(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  auto *gep = llvm::cast<llvm::GetElementPtrInst>(inst);
  if (gep->getType()->isVectorTy()) {
    return false;
  }

  ValueRef index_vr{this};
  ValuePartRef index_vp{this};
  GenericValuePart addr = typename GenericValuePart::Expr{};
  auto &expr = std::get<typename GenericValuePart::Expr>(addr.state);

  // Kept separate, we don't want to fold the displacement whenever we add an
  // index. GEP components are sign-extended, but we must use an unsigned
  // integer here to correctly handle overflows.
  u64 displacement = 0;
  // If set, the base is actually a stack variable reference and expr.base is
  // still uninitialized.
  bool base_is_stack_var = false;

  auto [ptr_ref, base] = this->val_ref_single(gep->getPointerOperand());
  if (base.has_assignment() && base.assignment().is_stack_variable()) {
    base_is_stack_var = true;
  } else {
    expr.base = base.load_to_reg();
    if (base.can_salvage()) {
      expr.base = ScratchReg{this};
      std::get<ScratchReg>(expr.base).alloc_specific(base.salvage());
    }
  }

  auto &data_layout = this->adaptor->mod->getDataLayout();

  // Next single-use val
  const llvm::Instruction *next_val = nullptr;
  do {
    // Handle index
    bool first_idx = true;
    auto *cur_ty = gep->getSourceElementType();
    for (const llvm::Use &idx : gep->indices()) {
      const auto idx_width = idx->getType()->getIntegerBitWidth();
      if (idx_width > 64) {
        return false;
      }

      if (auto *Const = llvm::dyn_cast<llvm::ConstantInt>(idx)) {
        u64 off_disp = 0;
        if (first_idx) {
          // array index
          if (i64 idx_val = Const->getSExtValue(); idx_val != 0) {
            i64 alloc_size = 1;
            // LLVM nowadays canonicalizes getelementptr to i8 as a preparation
            // for a possible ptradd migration. Add a fast path for this.
            if (!cur_ty->isIntegerTy() || cur_ty->getIntegerBitWidth() != 8) {
              alloc_size = data_layout.getTypeAllocSize(cur_ty);
            }
            off_disp = alloc_size * idx_val;
          }
        } else if (auto *struct_ty = llvm::dyn_cast<llvm::StructType>(cur_ty)) {
          u64 field_idx = Const->getZExtValue();
          cur_ty = cur_ty->getStructElementType(field_idx);
          if (field_idx != 0) {
            auto *struct_layout = data_layout.getStructLayout(struct_ty);
            off_disp = struct_layout->getElementOffset(field_idx);
          }
        } else {
          assert(cur_ty->isArrayTy());
          cur_ty = cur_ty->getArrayElementType();
          if (i64 idx_val = Const->getSExtValue(); idx_val != 0) {
            off_disp = data_layout.getTypeAllocSize(cur_ty) * idx_val;
          }
        }
        displacement += off_disp;
      } else {
        // A non-constant GEP. This must either be an offset calculation (for
        // index == 0) or an array traversal
        if (!first_idx) {
          cur_ty = cur_ty->getArrayElementType();
        }

        if (base_is_stack_var) {
          addr = derived()->create_addr_for_alloca(base.assignment());
          assert(addr.is_expr());
          displacement += expr.disp;
          expr.disp = 0;
          base_is_stack_var = false;
        }

        if (expr.scale) {
          derived()->gval_expr_as_reg(addr);
          index_vp.reset();
          index_vr.reset();
          base.reset();
          ptr_ref.reset();

          ScratchReg new_base = std::move(std::get<ScratchReg>(addr.state));
          addr = typename GenericValuePart::Expr{};
          expr.base = std::move(new_base);
        }

        index_vr = this->val_ref(idx);
        if (idx_width != 64) {
          index_vp = index_vr.part(0).into_extended(true, idx_width, 64);
        } else {
          index_vp = index_vr.part(0);
        }
        if (index_vp.can_salvage()) {
          expr.index = ScratchReg{this};
          std::get<ScratchReg>(expr.index).alloc_specific(index_vp.salvage());
        } else {
          expr.index = index_vp.load_to_reg();
        }

        expr.scale = data_layout.getTypeAllocSize(cur_ty);
      }

      first_idx = false;
    }

    if (!gep->hasOneUse()) {
      break;
    }

    // Try to fuse next instruction
    next_val = gep->getNextNode();
    if (gep->use_begin()->getUser() != next_val) {
      next_val = nullptr;
      break;
    }

    auto *next_gep = llvm::dyn_cast<llvm::GetElementPtrInst>(next_val);
    if (!next_gep) {
      break;
    }

    // GEP only takes a single pointer operand, so this must be the use.
    assert(next_gep->getPointerOperand() == gep);
    this->adaptor->inst_set_fused(next_val, true);
    gep = next_gep;
  } while (true);

  if (base_is_stack_var) {
    if (!next_val) {
      // Create a new stack variable reference to avoid materializing this
      // simple addition.
      (void)this->result_ref_stack_slot(gep, base.assignment(), displacement);
      return true;
    }

    addr = derived()->create_addr_for_alloca(base.assignment());
    expr.disp += displacement;
  } else {
    expr.disp = displacement;
  }

  if (auto *store = llvm::dyn_cast_if_present<llvm::StoreInst>(next_val);
      store && store->getPointerOperand() == gep) {
    this->adaptor->inst_set_fused(next_val, true);
    return compile_store_generic(store, std::move(addr));
  }
  if (auto *load = llvm::dyn_cast_if_present<llvm::LoadInst>(next_val)) {
    assert(load->getPointerOperand() == gep);
    this->adaptor->inst_set_fused(next_val, true);
    return compile_load_generic(load, std::move(addr));
  }

  auto [res_vr, res_ref] = this->result_ref_single(gep);

  AsmReg res_reg = derived()->gval_expr_as_reg(addr);
  if (auto *op_reg = std::get_if<ScratchReg>(&addr.state)) {
    res_ref.set_value(std::move(*op_reg));
  } else {
    derived()->mov(res_ref.alloc_reg(), res_reg, 8);
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_fcmp(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *cmp = llvm::cast<llvm::FCmpInst>(inst);
  auto *cmp_ty = cmp->getOperand(0)->getType();
  if (cmp_ty->isVectorTy()) {
    return false;
  }

  const auto pred = cmp->getPredicate();

  if (pred == llvm::CmpInst::FCMP_FALSE || pred == llvm::CmpInst::FCMP_TRUE) {
    u64 val = pred == llvm::CmpInst::FCMP_FALSE ? 0u : 1u;
    (void)this->val_ref(cmp->getOperand(0)); // ref-count
    (void)this->val_ref(cmp->getOperand(1)); // ref-count
    auto const_ref = ValuePartRef{this, val, 1, Config::GP_BANK};
    this->result_ref(cmp).part(0).set_value(std::move(const_ref));
    return true;
  }

  if (cmp_ty->isFP128Ty()) {
    SymRef sym;
    llvm::CmpInst::Predicate cmp_pred = llvm::CmpInst::ICMP_EQ;
    switch (pred) {
    case llvm::CmpInst::FCMP_OEQ:
      sym = get_libfunc_sym(LibFunc::eqtf2);
      cmp_pred = llvm::CmpInst::ICMP_EQ;
      break;
    case llvm::CmpInst::FCMP_UNE:
      sym = get_libfunc_sym(LibFunc::netf2);
      cmp_pred = llvm::CmpInst::ICMP_NE;
      break;
    case llvm::CmpInst::FCMP_OGT:
      sym = get_libfunc_sym(LibFunc::gttf2);
      cmp_pred = llvm::CmpInst::ICMP_SGT;
      break;
    case llvm::CmpInst::FCMP_ULE:
      sym = get_libfunc_sym(LibFunc::gttf2);
      cmp_pred = llvm::CmpInst::ICMP_SLE;
      break;
    case llvm::CmpInst::FCMP_OGE:
      sym = get_libfunc_sym(LibFunc::getf2);
      cmp_pred = llvm::CmpInst::ICMP_SGE;
      break;
    case llvm::CmpInst::FCMP_ULT:
      sym = get_libfunc_sym(LibFunc::getf2);
      cmp_pred = llvm::CmpInst::ICMP_SLT;
      break;
    case llvm::CmpInst::FCMP_OLT:
      sym = get_libfunc_sym(LibFunc::lttf2);
      cmp_pred = llvm::CmpInst::ICMP_SLT;
      break;
    case llvm::CmpInst::FCMP_UGE:
      sym = get_libfunc_sym(LibFunc::lttf2);
      cmp_pred = llvm::CmpInst::ICMP_SGE;
      break;
    case llvm::CmpInst::FCMP_OLE:
      sym = get_libfunc_sym(LibFunc::letf2);
      cmp_pred = llvm::CmpInst::ICMP_SLE;
      break;
    case llvm::CmpInst::FCMP_UGT:
      sym = get_libfunc_sym(LibFunc::letf2);
      cmp_pred = llvm::CmpInst::ICMP_SGT;
      break;
    case llvm::CmpInst::FCMP_ORD:
      sym = get_libfunc_sym(LibFunc::unordtf2);
      cmp_pred = llvm::CmpInst::ICMP_EQ;
      break;
    case llvm::CmpInst::FCMP_UNO:
      sym = get_libfunc_sym(LibFunc::unordtf2);
      cmp_pred = llvm::CmpInst::ICMP_NE;
      break;
    case llvm::CmpInst::FCMP_ONE:
    case llvm::CmpInst::FCMP_UEQ:
      // TODO: implement fp128 fcmp one/ueq
      // ONE __unordtf2 == 0 && __eqtf2 != 0
      // UEQ __unordtf2 != 0 || __eqtf2 == 0
      return false;
    default: TPDE_UNREACHABLE("unexpected fcmp predicate");
    }

    IRValueRef lhs = cmp->getOperand(0);
    IRValueRef rhs = cmp->getOperand(1);
    std::array<IRValueRef, 2> args{lhs, rhs};

    auto res_vr = this->result_ref(cmp);
    derived()->create_helper_call(args, &res_vr, sym);
    derived()->compile_i32_cmp_zero(res_vr.part(0).load_to_reg(), cmp_pred);

    return true;
  }

  ValueRef lhs = this->val_ref(cmp->getOperand(0));
  ValueRef rhs = this->val_ref(cmp->getOperand(1));
  ValueRef res = this->result_ref(cmp);
  if (cmp_ty->isX86_FP80Ty()) {
    if constexpr (requires { &Derived::fp80_cmp; }) {
      derived()->fp80_cmp(pred, lhs.part(0), rhs.part(0), res.part(0));
      return true;
    }
  }

  if (!cmp_ty->isFloatTy() && !cmp_ty->isDoubleTy()) {
    return false;
  }

  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ValuePart &&);
  EncodeFnTy fn = nullptr;

  if (cmp_ty->isFloatTy()) {
    switch (pred) {
      using enum llvm::CmpInst::Predicate;
    case FCMP_OEQ: fn = &Derived::encode_fcmp_oeq_float; break;
    case FCMP_OGT: fn = &Derived::encode_fcmp_ogt_float; break;
    case FCMP_OGE: fn = &Derived::encode_fcmp_oge_float; break;
    case FCMP_OLT: fn = &Derived::encode_fcmp_olt_float; break;
    case FCMP_OLE: fn = &Derived::encode_fcmp_ole_float; break;
    case FCMP_ONE: fn = &Derived::encode_fcmp_one_float; break;
    case FCMP_ORD: fn = &Derived::encode_fcmp_ord_float; break;
    case FCMP_UEQ: fn = &Derived::encode_fcmp_ueq_float; break;
    case FCMP_UGT: fn = &Derived::encode_fcmp_ugt_float; break;
    case FCMP_UGE: fn = &Derived::encode_fcmp_uge_float; break;
    case FCMP_ULT: fn = &Derived::encode_fcmp_ult_float; break;
    case FCMP_ULE: fn = &Derived::encode_fcmp_ule_float; break;
    case FCMP_UNE: fn = &Derived::encode_fcmp_une_float; break;
    case FCMP_UNO: fn = &Derived::encode_fcmp_uno_float; break;
    default: TPDE_UNREACHABLE("invalid fcmp predicate");
    }
  } else {
    switch (pred) {
      using enum llvm::CmpInst::Predicate;
    case FCMP_OEQ: fn = &Derived::encode_fcmp_oeq_double; break;
    case FCMP_OGT: fn = &Derived::encode_fcmp_ogt_double; break;
    case FCMP_OGE: fn = &Derived::encode_fcmp_oge_double; break;
    case FCMP_OLT: fn = &Derived::encode_fcmp_olt_double; break;
    case FCMP_OLE: fn = &Derived::encode_fcmp_ole_double; break;
    case FCMP_ONE: fn = &Derived::encode_fcmp_one_double; break;
    case FCMP_ORD: fn = &Derived::encode_fcmp_ord_double; break;
    case FCMP_UEQ: fn = &Derived::encode_fcmp_ueq_double; break;
    case FCMP_UGT: fn = &Derived::encode_fcmp_ugt_double; break;
    case FCMP_UGE: fn = &Derived::encode_fcmp_uge_double; break;
    case FCMP_ULT: fn = &Derived::encode_fcmp_ult_double; break;
    case FCMP_ULE: fn = &Derived::encode_fcmp_ule_double; break;
    case FCMP_UNE: fn = &Derived::encode_fcmp_une_double; break;
    case FCMP_UNO: fn = &Derived::encode_fcmp_uno_double; break;
    default: TPDE_UNREACHABLE("invalid fcmp predicate");
    }
  }

  return (derived()->*fn)(lhs.part(0), rhs.part(0), res.part(0));
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_switch(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *switch_inst = llvm::cast<llvm::SwitchInst>(inst);
  llvm::Value *cond = switch_inst->getCondition();
  u32 width = cond->getType()->getIntegerBitWidth();
  if (width > 64) {
    return false;
  }

  // Collect cases, their target block and sort them in ascending order.
  tpde::util::SmallVector<std::pair<u64, IRBlockRef>, 64> cases;
  assert(switch_inst->getNumCases() <= 200000);
  cases.reserve(switch_inst->getNumCases());
  for (auto case_val : switch_inst->cases()) {
    cases.push_back(std::make_pair(
        case_val.getCaseValue()->getZExtValue(),
        this->adaptor->block_lookup_idx(case_val.getCaseSuccessor())));
  }
  std::sort(cases.begin(), cases.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  auto def = this->adaptor->block_lookup_idx(switch_inst->getDefaultDest());

  // cond must be ref-counted before generate_switch.
  ScratchReg cond_scratch = this->val_ref(cond).part(0).into_scratch();
  this->generate_switch(std::move(cond_scratch), width, def, cases);
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_invoke(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  const auto *invoke = llvm::cast<llvm::InvokeInst>(inst);

  // we need to spill here since the call might branch off
  // TODO: this will also spill the call arguments even if the call kills them
  // however, spillBeforeCall already does this anyways so probably something
  // for later
  auto spilled = this->spill_before_branch();

  const auto off_before_call = this->text_writer.offset();
  // compile the call
  // TODO: in the case of an exception we need to invalidate the result
  // registers
  // TODO: if the call needs stack space, this must be undone in the unwind
  // block! LLVM emits .cfi_escape 0x2e, <off>, we should do the same?
  // (Current workaround by treating invoke as dynamic alloca.)
  if (!this->compile_call(invoke, val_info, 0)) {
    return false;
  }
  const auto off_after_call = this->text_writer.offset();

  // build the eh table
  auto *unwind_block = invoke->getUnwindDest();
  llvm::LandingPadInst *landing_pad = nullptr;
  auto unwind_block_has_phi = false;

  for (auto it = unwind_block->begin(), end = unwind_block->end(); it != end;
       ++it) {
    llvm::Instruction *inst = &*it;
    if (llvm::isa<llvm::PHINode>(inst)) {
      unwind_block_has_phi = true;
      continue;
    }

    landing_pad = llvm::cast<llvm::LandingPadInst>(inst);
    break;
  }

  const auto unwind_block_ref = this->adaptor->block_lookup_idx(unwind_block);
  const auto normal_block_ref =
      this->adaptor->block_lookup_idx(invoke->getNormalDest());
  auto unwind_label =
      this->block_labels[(u32)this->analyzer.block_idx(unwind_block_ref)];

  // We always spill the call result. Also, generate_call might move values
  // again into registers, which we need to release again.
  // TODO: evaluate when exactly this is required.
  spilled |= this->spill_before_branch(/*force_spill=*/true);

  // if the unwind block has phi-nodes, we need more code to propagate values
  // to it so do the propagation logic
  if (unwind_block_has_phi) {
    // generate the jump to the normal successor but don't allow
    // fall-through
    derived()->generate_branch_to_block(Derived::Jump::jmp,
                                        normal_block_ref,
                                        /* split */ false,
                                        /* last_inst */ false);

    this->release_spilled_regs(spilled);

    unwind_label = this->text_writer.label_create();
    this->label_place(unwind_label);

    // allocate the special registers that are set by the unwinding logic
    // so the phi-propagation does not use them as temporaries
    ScratchReg scratch1{derived()}, scratch2{derived()};
    assert(!this->register_file.is_used(Derived::LANDING_PAD_RES_REGS[0]));
    assert(!this->register_file.is_used(Derived::LANDING_PAD_RES_REGS[1]));
    scratch1.alloc_specific(Derived::LANDING_PAD_RES_REGS[0]);
    scratch2.alloc_specific(Derived::LANDING_PAD_RES_REGS[1]);

    derived()->generate_branch_to_block(Derived::Jump::jmp,
                                        unwind_block_ref,
                                        /* split */ false,
                                        /* last_inst */ false);
  } else {
    // allow fall-through
    derived()->generate_branch_to_block(Derived::Jump::jmp,
                                        normal_block_ref,
                                        /* split */ false,
                                        /* last_inst */ true);

    this->release_spilled_regs(spilled);
  }

  const auto is_cleanup = landing_pad->isCleanup();
  const auto num_clauses = landing_pad->getNumClauses();
  const auto only_cleanup = is_cleanup && num_clauses == 0;

  this->text_writer.except_add_call_site(off_before_call,
                                         off_after_call - off_before_call,
                                         unwind_label,
                                         only_cleanup);

  if (only_cleanup) {
    // no clause so we are done
    return true;
  }

  for (auto i = 0u; i < num_clauses; ++i) {
    if (landing_pad->isCatch(i)) {
      auto *C = landing_pad->getClause(i);
      SymRef sym;
      if (!C->isNullValue()) {
        sym = lookup_type_info_sym(llvm::cast<llvm::GlobalValue>(C));
      }
      this->text_writer.except_add_action(i == 0, sym);
    } else {
      assert(landing_pad->isFilter(i));
      auto *C = landing_pad->getClause(i);
      assert(C->getType()->isArrayTy());
      if (C->getType()->getArrayNumElements() == 0) {
        this->text_writer.except_add_empty_spec_action(i == 0);
      } else {
        TPDE_LOG_ERR("Exception filters with non-zero length arrays "
                     "not supported");
        return false;
      }
    }
  }

  if (is_cleanup) {
    assert(num_clauses != 0);
    this->text_writer.except_add_cleanup_action();
  }

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_landing_pad(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  auto res_ref = this->result_ref(inst);
  res_ref.part(0).set_value_reg(Derived::LANDING_PAD_RES_REGS[0]);
  res_ref.part(1).set_value_reg(Derived::LANDING_PAD_RES_REGS[1]);

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_resume(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  IRValueRef arg = inst->getOperand(0);

  const auto sym = get_libfunc_sym(LibFunc::resume);

  derived()->create_helper_call({&arg, 1}, nullptr, sym);
  return derived()->compile_unreachable(nullptr, val_info, 0);
}

template <typename Adaptor, typename Derived, typename Config>
typename LLVMCompilerBase<Adaptor, Derived, Config>::SymRef
    LLVMCompilerBase<Adaptor, Derived, Config>::lookup_type_info_sym(
        const llvm::GlobalValue *value) noexcept {
  for (const auto &[val, sym] : type_info_syms) {
    if (val == value) {
      return sym;
    }
  }

  const auto sym = global_sym(value);

  u32 off;
  u8 tmp[8] = {};
  auto rodata = this->assembler.get_data_section(true, true);
  const auto addr_sym =
      this->assembler.sym_def_data(rodata,
                                   {},
                                   {tmp, sizeof(tmp)},
                                   8,
                                   tpde::Assembler::SymBinding::LOCAL,
                                   &off);
  this->assembler.reloc_abs(rodata, sym, off, 0);

  type_info_syms.emplace_back(value, addr_sym);
  return addr_sym;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_intrin(
    const llvm::IntrinsicInst *inst, const ValInfo &info) noexcept {
  const auto intrin_id = inst->getIntrinsicID();

  switch (intrin_id) {
  case llvm::Intrinsic::donothing:
  case llvm::Intrinsic::sideeffect:
  case llvm::Intrinsic::experimental_noalias_scope_decl:
  case llvm::Intrinsic::dbg_assign:
  case llvm::Intrinsic::dbg_declare:
  case llvm::Intrinsic::dbg_label: return true;
  case llvm::Intrinsic::dbg_value:
    // reference counting
    this->val_ref_single(inst->getOperand(1));
    return true;
  case llvm::Intrinsic::assume:
  case llvm::Intrinsic::lifetime_start:
  case llvm::Intrinsic::lifetime_end:
  case llvm::Intrinsic::invariant_start:
  case llvm::Intrinsic::invariant_end:
    // reference counting; also include operand bundle uses
    for (llvm::Value *arg : inst->data_ops()) {
      this->val_ref(arg);
    }
    return true;
  case llvm::Intrinsic::expect: {
    // Just copy the first operand.
    (void)this->val_ref(inst->getOperand(1));
    auto src_ref = this->val_ref(inst->getOperand(0));
    auto res_ref = this->result_ref(inst);
    const auto part_count = res_ref.assignment()->part_count;
    for (u32 part_idx = 0; part_idx < part_count; ++part_idx) {
      res_ref.part(part_idx).set_value(src_ref.part(part_idx));
    }
    return true;
  }
  case llvm::Intrinsic::memcpy: {
    const auto dst = inst->getOperand(0);
    const auto src = inst->getOperand(1);
    const auto len = inst->getOperand(2);

    std::array<IRValueRef, 3> args{dst, src, len};

    const auto sym = get_libfunc_sym(LibFunc::memcpy);
    derived()->create_helper_call(args, nullptr, sym);
    return true;
  }
  case llvm::Intrinsic::memset: {
    const auto dst = inst->getOperand(0);
    const auto val = inst->getOperand(1);
    const auto len = inst->getOperand(2);

    std::array<IRValueRef, 3> args{dst, val, len};

    const auto sym = get_libfunc_sym(LibFunc::memset);
    derived()->create_helper_call(args, nullptr, sym);
    return true;
  }
  case llvm::Intrinsic::memmove: {
    const auto dst = inst->getOperand(0);
    const auto src = inst->getOperand(1);
    const auto len = inst->getOperand(2);

    std::array<IRValueRef, 3> args{dst, src, len};

    const auto sym = get_libfunc_sym(LibFunc::memmove);
    derived()->create_helper_call(args, nullptr, sym);
    return true;
  }
  case llvm::Intrinsic::load_relative: {
    if (!inst->getOperand(1)->getType()->isIntegerTy(64)) {
      return false;
    }

    auto ptr = this->val_ref(inst->getOperand(0));
    auto off = this->val_ref(inst->getOperand(1));
    auto [res_vr, res_ref] = this->result_ref_single(inst);
    derived()->encode_loadreli64(ptr.part(0), off.part(0), res_ref);
    return true;
  }
  case llvm::Intrinsic::threadlocal_address: {
    auto gv = llvm::cast<llvm::GlobalValue>(inst->getOperand(0));
    auto [res_vr, res_ref] = this->result_ref_single(inst);
    // TODO: optimize for different TLS access models
    ScratchReg res =
        derived()->tls_get_addr(global_sym(gv), tpde::TLSModel::GlobalDynamic);
    res_ref.set_value(std::move(res));
    return true;
  }
  case llvm::Intrinsic::vaend: {
    // no-op
    this->val_ref_single(inst->getOperand(0));
    return true;
  }
  case llvm::Intrinsic::is_fpclass: {
    return compile_is_fpclass(inst);
  }
  case llvm::Intrinsic::floor:
  case llvm::Intrinsic::ceil:
  case llvm::Intrinsic::round:
  case llvm::Intrinsic::rint:
  case llvm::Intrinsic::trunc:
  case llvm::Intrinsic::pow:
  case llvm::Intrinsic::powi:
  case llvm::Intrinsic::sin:
  case llvm::Intrinsic::cos:
  case llvm::Intrinsic::log:
  case llvm::Intrinsic::log10:
  case llvm::Intrinsic::exp: {
    // Floating-point intrinsics that can be mapped directly to libcalls.
    const auto is_double = inst->getType()->isDoubleTy();
    if (!is_double && !inst->getType()->isFloatTy()) {
      return false;
    }

    LibFunc func;
    switch (intrin_id) {
      using enum llvm::Intrinsic::IndependentIntrinsics;
    case floor: func = is_double ? LibFunc::floor : LibFunc::floorf; break;
    case ceil: func = is_double ? LibFunc::ceil : LibFunc::ceilf; break;
    case round: func = is_double ? LibFunc::round : LibFunc::roundf; break;
    case rint: func = is_double ? LibFunc::rint : LibFunc::rintf; break;
    case trunc: func = is_double ? LibFunc::trunc : LibFunc::truncf; break;
    case pow: func = is_double ? LibFunc::pow : LibFunc::powf; break;
    case powi: func = is_double ? LibFunc::powidf2 : LibFunc::powisf2; break;
    case sin: func = is_double ? LibFunc::sin : LibFunc::sinf; break;
    case cos: func = is_double ? LibFunc::cos : LibFunc::cosf; break;
    case log: func = is_double ? LibFunc::log : LibFunc::logf; break;
    case log10: func = is_double ? LibFunc::log10 : LibFunc::log10f; break;
    case exp: func = is_double ? LibFunc::exp : LibFunc::expf; break;
    default: TPDE_UNREACHABLE("invalid library fp intrinsic");
    }

    llvm::SmallVector<IRValueRef, 2> ops;
    for (auto &op : inst->args()) {
      ops.push_back(op);
    }
    auto res_vr = this->result_ref(inst);
    derived()->create_helper_call(ops, &res_vr, get_libfunc_sym(func));
    return true;
  }
  case llvm::Intrinsic::minnum:
  case llvm::Intrinsic::maxnum:
  case llvm::Intrinsic::copysign: {
    // Floating-point intrinsics with two operands
    const auto is_double = inst->getType()->isDoubleTy();
    if (!is_double && !inst->getType()->isFloatTy()) {
      return false;
    }

    using EncodeFnTy = bool (Derived::*)(
        GenericValuePart &&, GenericValuePart &&, ValuePart &&);
    EncodeFnTy fn;
    if (is_double) {
      switch (intrin_id) {
        using enum llvm::Intrinsic::IndependentIntrinsics;
      case minnum: fn = &Derived::encode_minnumf64; break;
      case maxnum: fn = &Derived::encode_maxnumf64; break;
      case copysign: fn = &Derived::encode_copysignf64; break;
      default: TPDE_UNREACHABLE("invalid binary fp intrinsic");
      }
    } else {
      switch (intrin_id) {
        using enum llvm::Intrinsic::IndependentIntrinsics;
      case minnum: fn = &Derived::encode_minnumf32; break;
      case maxnum: fn = &Derived::encode_maxnumf32; break;
      case copysign: fn = &Derived::encode_copysignf32; break;
      default: TPDE_UNREACHABLE("invalid binary fp intrinsic");
      }
    }

    auto lhs = this->val_ref(inst->getOperand(0));
    auto rhs = this->val_ref(inst->getOperand(1));
    auto res = this->result_ref(inst);
    return (derived()->*fn)(lhs.part(0), rhs.part(0), res.part(0));
  }
  case llvm::Intrinsic::fabs: {
    auto *val = inst->getOperand(0);
    auto *ty = val->getType();

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    if (ty->isDoubleTy()) {
      derived()->encode_fabsf64(this->val_ref(val).part(0), res_ref);
    } else if (ty->isFloatTy()) {
      derived()->encode_fabsf32(this->val_ref(val).part(0), res_ref);
    } else if (ty->isFP128Ty()) {
      derived()->encode_fabsf128(this->val_ref(val).part(0), res_ref);
    } else {
      return false;
    }
    return true;
  }
  case llvm::Intrinsic::sqrt: {
    auto *val = inst->getOperand(0);
    auto *ty = val->getType();

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    if (ty->isDoubleTy()) {
      derived()->encode_sqrtf64(this->val_ref(val).part(0), res_ref);
    } else if (ty->isFloatTy()) {
      derived()->encode_sqrtf32(this->val_ref(val).part(0), res_ref);
    } else {
      return false;
    }
    return true;
  }
  case llvm::Intrinsic::fmuladd: {
    ValueRef op1 = this->val_ref(inst->getOperand(0));
    ValueRef op2 = this->val_ref(inst->getOperand(1));
    ValueRef op3 = this->val_ref(inst->getOperand(2));
    ValueRef res = this->result_ref(inst);

    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&,
                                         GenericValuePart &&,
                                         GenericValuePart &&,
                                         ValuePart &&);
    EncodeFnTy fn = nullptr;
    switch (info.type) {
      using enum LLVMBasicValType;
    case f32: fn = &Derived::encode_fmuladdf32; break;
    case f64: fn = &Derived::encode_fmuladdf64; break;
    case v2f32: fn = &Derived::encode_fmuladdv2f32; break;
    case v4f32: fn = &Derived::encode_fmuladdv4f32; break;
    case v2f64: fn = &Derived::encode_fmuladdv2f64; break;
    case f128: {
      auto cb1 = derived()->create_call_builder();
      cb1->add_arg(op1.part(0), tpde::CCAssignment{});
      cb1->add_arg(op2.part(0), tpde::CCAssignment{});
      cb1->call(get_libfunc_sym(LibFunc::multf3));
      ValuePartRef tmp{this, Config::FP_BANK};
      cb1->add_ret(tmp, tpde::CCAssignment{});

      auto cb2 = derived()->create_call_builder();
      cb2->add_arg(std::move(tmp), tpde::CCAssignment{});
      cb2->add_arg(op3.part(0), tpde::CCAssignment{});
      cb2->call(get_libfunc_sym(LibFunc::addtf3));
      cb2->add_ret(res);
      return true;
    }
    case f80:
      if constexpr (requires { &Derived::fp80_muladd; }) {
        derived()->fp80_muladd(
            op1.part(0), op2.part(0), op3.part(0), res.part(0));
        return true;
      }
      return false;
    default: return false;
    }

    return (derived()->*fn)(op1.part(0), op2.part(0), op3.part(0), res.part(0));
  }
  case llvm::Intrinsic::abs: {
    auto *val = inst->getOperand(0);
    auto *val_ty = val->getType();
    if (!val_ty->isIntegerTy()) {
      return false;
    }
    const auto width = val_ty->getIntegerBitWidth();
    ValueRef val_ref = this->val_ref(val);

    if (width == 128) {
      ValueRef res_ref = this->result_ref(inst);
      derived()->encode_absi128(
          val_ref.part(0), val_ref.part(1), res_ref.part(0), res_ref.part(1));
      return true;
    }
    if (width > 64) {
      return false;
    }

    ValuePartRef op = val_ref.part(0);
    if (width != 32 && width != 64) {
      unsigned dst_width = tpde::util::align_up(width, 32);
      op = std::move(op).into_extended(/*sign=*/true, width, dst_width);
    }

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    if (width <= 32) {
      derived()->encode_absi32(std::move(op), res_ref);
    } else {
      derived()->encode_absi64(std::move(op), res_ref);
    }
    return true;
  }
  case llvm::Intrinsic::ucmp:
  case llvm::Intrinsic::scmp:
    if (!inst->getType()->isIntegerTy() ||
        inst->getType()->getIntegerBitWidth() > 64) {
      return false;
    }
    [[fallthrough]];
  case llvm::Intrinsic::umin:
  case llvm::Intrinsic::umax:
  case llvm::Intrinsic::smin:
  case llvm::Intrinsic::smax: {
    auto *ty = inst->getOperand(0)->getType();
    if (!ty->isIntegerTy()) {
      return false;
    }
    const auto width = ty->getIntegerBitWidth();
    if (width > 64) {
      return false;
    }

    bool sign = intrin_id == llvm::Intrinsic::scmp ||
                intrin_id == llvm::Intrinsic::smin ||
                intrin_id == llvm::Intrinsic::smax;

    ValueRef lhs_ref = this->val_ref(inst->getOperand(0));
    ValueRef rhs_ref = this->val_ref(inst->getOperand(1));
    ValuePartRef lhs = lhs_ref.part(0);
    ValuePartRef rhs = rhs_ref.part(0);
    if (width != 32 && width != 64) {
      unsigned dst_width = tpde::util::align_up(width, 32);
      lhs = std::move(lhs).into_extended(sign, width, dst_width);
      rhs = std::move(rhs).into_extended(sign, width, dst_width);
    }

    ValueRef res = this->result_ref(inst);
    using EncodeFnTy = bool (Derived::*)(
        GenericValuePart &&, GenericValuePart &&, ValuePart &&);
    EncodeFnTy encode_fn = nullptr;
    if (width <= 32) {
      switch (intrin_id) {
      case llvm::Intrinsic::ucmp: encode_fn = &Derived::encode_ucmpi32; break;
      case llvm::Intrinsic::umin: encode_fn = &Derived::encode_umini32; break;
      case llvm::Intrinsic::umax: encode_fn = &Derived::encode_umaxi32; break;
      case llvm::Intrinsic::scmp: encode_fn = &Derived::encode_scmpi32; break;
      case llvm::Intrinsic::smin: encode_fn = &Derived::encode_smini32; break;
      case llvm::Intrinsic::smax: encode_fn = &Derived::encode_smaxi32; break;
      default: TPDE_UNREACHABLE("invalid intrinsic");
      }
    } else {
      switch (intrin_id) {
      case llvm::Intrinsic::ucmp: encode_fn = &Derived::encode_ucmpi64; break;
      case llvm::Intrinsic::umin: encode_fn = &Derived::encode_umini64; break;
      case llvm::Intrinsic::umax: encode_fn = &Derived::encode_umaxi64; break;
      case llvm::Intrinsic::scmp: encode_fn = &Derived::encode_scmpi64; break;
      case llvm::Intrinsic::smin: encode_fn = &Derived::encode_smini64; break;
      case llvm::Intrinsic::smax: encode_fn = &Derived::encode_smaxi64; break;
      default: TPDE_UNREACHABLE("invalid intrinsic");
      }
    }
    return (derived()->*encode_fn)(std::move(lhs), std::move(rhs), res.part(0));
  }
  case llvm::Intrinsic::ptrmask: {
    // ptrmask is just an integer and.
    ValueRef lhs = this->val_ref(inst->getOperand(0));
    ValueRef rhs = this->val_ref(inst->getOperand(1));
    auto [res_vr, res] = this->result_ref_single(inst);
    derived()->encode_landi64(lhs.part(0), rhs.part(0), res);
    return true;
  }
  case llvm::Intrinsic::uadd_with_overflow:
    return compile_overflow_intrin(inst, OverflowOp::uadd);
  case llvm::Intrinsic::sadd_with_overflow:
    return compile_overflow_intrin(inst, OverflowOp::sadd);
  case llvm::Intrinsic::usub_with_overflow:
    return compile_overflow_intrin(inst, OverflowOp::usub);
  case llvm::Intrinsic::ssub_with_overflow:
    return compile_overflow_intrin(inst, OverflowOp::ssub);
  case llvm::Intrinsic::umul_with_overflow:
    return compile_overflow_intrin(inst, OverflowOp::umul);
  case llvm::Intrinsic::smul_with_overflow:
    return compile_overflow_intrin(inst, OverflowOp::smul);
  case llvm::Intrinsic::uadd_sat:
    return compile_saturating_intrin(inst, OverflowOp::uadd);
  case llvm::Intrinsic::sadd_sat:
    return compile_saturating_intrin(inst, OverflowOp::sadd);
  case llvm::Intrinsic::usub_sat:
    return compile_saturating_intrin(inst, OverflowOp::usub);
  case llvm::Intrinsic::ssub_sat:
    return compile_saturating_intrin(inst, OverflowOp::ssub);
  case llvm::Intrinsic::fptoui_sat:
    return compile_float_to_int(inst, info, /*flags=!sign|sat*/ 0b10);
  case llvm::Intrinsic::fptosi_sat:
    return compile_float_to_int(inst, info, /*flags=sign|sat*/ 0b11);
  case llvm::Intrinsic::vector_reduce_add:
  case llvm::Intrinsic::vector_reduce_fadd:
  case llvm::Intrinsic::vector_reduce_mul:
  case llvm::Intrinsic::vector_reduce_fmul:
  case llvm::Intrinsic::vector_reduce_and:
  case llvm::Intrinsic::vector_reduce_or:
  case llvm::Intrinsic::vector_reduce_xor:
  case llvm::Intrinsic::vector_reduce_smax:
  case llvm::Intrinsic::vector_reduce_smin:
  case llvm::Intrinsic::vector_reduce_umax:
  case llvm::Intrinsic::vector_reduce_umin:
    return compile_vector_reduce(inst, info);
  case llvm::Intrinsic::fshl:
  case llvm::Intrinsic::fshr: {
    if (!inst->getType()->isIntegerTy()) {
      return false;
    }
    const auto width = inst->getType()->getIntegerBitWidth();
    // Implementing non-powers-of-two is difficult, would require modulo of
    // shift amount. Doesn't really occur in practice.
    if (width != 8 && width != 16 && width != 32 && width != 64) {
      return false;
    }

    auto [res_vr, res_ref] = this->result_ref_single(inst);

    // TODO: generate better code for constant amounts.
    bool shift_left = intrin_id == llvm::Intrinsic::fshl;
    if (inst->getOperand(0) == inst->getOperand(1)) {
      // Better code for rotate.
      using EncodeFnTy = bool (Derived::*)(
          GenericValuePart &&, GenericValuePart &&, ValuePart &);
      EncodeFnTy fn = nullptr;
      if (shift_left) {
        switch (width) {
        case 8: fn = &Derived::encode_roli8; break;
        case 16: fn = &Derived::encode_roli16; break;
        case 32: fn = &Derived::encode_roli32; break;
        case 64: fn = &Derived::encode_roli64; break;
        default: TPDE_UNREACHABLE("unreachable width");
        }
      } else {
        switch (width) {
        case 8: fn = &Derived::encode_rori8; break;
        case 16: fn = &Derived::encode_rori16; break;
        case 32: fn = &Derived::encode_rori32; break;
        case 64: fn = &Derived::encode_rori64; break;
        default: TPDE_UNREACHABLE("unreachable width");
        }
      }

      // ref-count; do this first so that lhs might see ref_count == 1
      (void)this->val_ref(inst->getOperand(1));
      auto lhs = this->val_ref(inst->getOperand(0));
      auto amt = this->val_ref(inst->getOperand(2));
      if (!(derived()->*fn)(lhs.part(0), amt.part(0), res_ref)) {
        return false;
      }
    } else {
      using EncodeFnTy = bool (Derived::*)(GenericValuePart &&,
                                           GenericValuePart &&,
                                           GenericValuePart &&,
                                           ValuePart &);
      EncodeFnTy fn = nullptr;
      if (shift_left) {
        switch (width) {
        case 8: fn = &Derived::encode_fshli8; break;
        case 16: fn = &Derived::encode_fshli16; break;
        case 32: fn = &Derived::encode_fshli32; break;
        case 64: fn = &Derived::encode_fshli64; break;
        default: TPDE_UNREACHABLE("unreachable width");
        }
      } else {
        switch (width) {
        case 8: fn = &Derived::encode_fshri8; break;
        case 16: fn = &Derived::encode_fshri16; break;
        case 32: fn = &Derived::encode_fshri32; break;
        case 64: fn = &Derived::encode_fshri64; break;
        default: TPDE_UNREACHABLE("unreachable width");
        }
      }

      auto lhs = this->val_ref(inst->getOperand(0));
      auto rhs = this->val_ref(inst->getOperand(1));
      auto amt = this->val_ref(inst->getOperand(2));
      if (!(derived()->*fn)(lhs.part(0), rhs.part(0), amt.part(0), res_ref)) {
        return false;
      }
    }

    return true;
  }
  case llvm::Intrinsic::bswap: {
    auto *val = inst->getOperand(0);
    if (!val->getType()->isIntegerTy()) {
      return false;
    }
    const auto width = val->getType()->getIntegerBitWidth();
    if (width == 128) {
      ValueRef src = this->val_ref(val);
      ValueRef dst = this->result_ref(inst);
      return derived()->encode_bswapi128(
          src.part(0), src.part(1), dst.part(0), dst.part(1));
    }
    if (width % 16 || width > 64) {
      return false;
    }

    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ValuePart &&);
    static constexpr std::array<EncodeFnTy, 4> encode_fns = {
        &Derived::encode_bswapi16,
        &Derived::encode_bswapi32,
        &Derived::encode_bswapi48,
        &Derived::encode_bswapi64,
    };
    EncodeFnTy encode_fn = encode_fns[width / 16 - 1];

    return (derived()->*encode_fn)(this->val_ref(val).part(0),
                                   this->result_ref(inst).part(0));
  }
  case llvm::Intrinsic::ctpop: {
    auto *val = inst->getOperand(0);
    if (!val->getType()->isIntegerTy()) {
      return false;
    }
    const auto width = val->getType()->getIntegerBitWidth();
    if (width > 64) {
      return false;
    }

    ValueRef val_ref = this->val_ref(val);
    ValuePartRef op = val_ref.part(0);
    if (width % 32) {
      unsigned tgt_width = tpde::util::align_up(width, 32);
      op = std::move(op).into_extended(/*sign=*/false, width, tgt_width);
    }

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    if (width <= 32) {
      derived()->encode_ctpopi32(std::move(op), res_ref);
    } else {
      derived()->encode_ctpopi64(std::move(op), res_ref);
    }
    return true;
  }
  case llvm::Intrinsic::ctlz:
  case llvm::Intrinsic::cttz: {
    auto *val = inst->getOperand(0);
    if (!val->getType()->isIntegerTy()) {
      return false;
    }
    u32 width_idx = 0;
    switch (val->getType()->getIntegerBitWidth()) {
    case 8: width_idx = 0; break;
    case 16: width_idx = 1; break;
    case 32: width_idx = 2; break;
    case 64: width_idx = 3; break;
    default: return false;
    }

    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ValuePart &&);
    static constexpr EncodeFnTy encode_fns[4][2][2] = {
#define F(n, op, suffix) &Derived::encode_##op##i##n##suffix
        {  {F(8, ctlz, ), F(8, ctlz, _zp)},  {F(8, cttz, ), F(32, cttz, _zp)}},
        {{F(16, ctlz, ), F(16, ctlz, _zp)}, {F(16, cttz, ), F(32, cttz, _zp)}},
        {{F(32, ctlz, ), F(32, ctlz, _zp)}, {F(32, cttz, ), F(32, cttz, _zp)}},
        {{F(64, ctlz, ), F(64, ctlz, _zp)}, {F(64, cttz, ), F(64, cttz, _zp)}},
#undef F
    };
    bool zero_is_poison =
        !llvm::cast<llvm::ConstantInt>(inst->getOperand(1))->isZero();
    bool is_cttz = intrin_id == llvm::Intrinsic::cttz;
    EncodeFnTy fn = encode_fns[width_idx][is_cttz][zero_is_poison];
    return (derived()->*fn)(this->val_ref(val).part(0),
                            this->result_ref(inst).part(0));
  }
  case llvm::Intrinsic::bitreverse: {
    auto *val = inst->getOperand(0);
    if (!val->getType()->isIntegerTy()) {
      return false;
    }
    const auto width = val->getType()->getIntegerBitWidth();
    if (width > 64) {
      return false;
    }

    ValueRef val_ref = this->val_ref(val);
    GenericValuePart op = val_ref.part(0);
    if (width % 32) {
      u64 amt = (width < 32 ? 32 : 64) - width;
      ValuePartRef amt_val{this, &amt, 8, Config::GP_BANK};
      ValuePartRef shifted{this, Config::GP_BANK};
      if (width < 32) {
        derived()->encode_shli32(std::move(op), std::move(amt_val), shifted);
      } else {
        derived()->encode_shli64(std::move(op), std::move(amt_val), shifted);
      }
      op = std::move(shifted);
    }

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    if (width <= 32) {
      derived()->encode_bitreversei32(std::move(op), res_ref);
    } else {
      derived()->encode_bitreversei64(std::move(op), res_ref);
    }
    return true;
  }
  case llvm::Intrinsic::trap: return derived()->encode_trap();
  case llvm::Intrinsic::debugtrap: return derived()->encode_debugtrap();
  case llvm::Intrinsic::prefetch: {
    auto ptr_ref = this->val_ref(inst->getOperand(0));

    const auto rw =
        llvm::cast<llvm::ConstantInt>(inst->getOperand(1))->getZExtValue();
    const auto locality =
        llvm::cast<llvm::ConstantInt>(inst->getOperand(2))->getZExtValue();
    // for now, ignore instruction/data distinction

    if (rw == 0) {
      // read
      switch (locality) {
      case 0: derived()->encode_prefetch_rl0(ptr_ref.part(0)); break;
      case 1: derived()->encode_prefetch_rl1(ptr_ref.part(0)); break;
      case 2: derived()->encode_prefetch_rl2(ptr_ref.part(0)); break;
      case 3: derived()->encode_prefetch_rl3(ptr_ref.part(0)); break;
      default: TPDE_UNREACHABLE("invalid prefetch locality");
      }
    } else {
      assert(rw == 1);
      // write
      switch (locality) {
      case 0: derived()->encode_prefetch_wl0(ptr_ref.part(0)); break;
      case 1: derived()->encode_prefetch_wl1(ptr_ref.part(0)); break;
      case 2: derived()->encode_prefetch_wl2(ptr_ref.part(0)); break;
      case 3: derived()->encode_prefetch_wl3(ptr_ref.part(0)); break;
      default: TPDE_UNREACHABLE("invalid prefetch locality");
      }
    }
    return true;
  }
  case llvm::Intrinsic::eh_typeid_for: {
    auto *type = llvm::cast<llvm::GlobalValue>(inst->getOperand(0));

    // not the most efficient but it's OK
    const auto type_info_sym = lookup_type_info_sym(type);
    const u64 idx = this->text_writer.except_type_idx_for_sym(type_info_sym);

    auto const_ref = ValuePartRef{this, idx, 4, Config::GP_BANK};
    this->result_ref(inst).part(0).set_value(std::move(const_ref));
    return true;
  }
  case llvm::Intrinsic::is_constant: {
    // > On the other hand, if constant folding is not run, it will never
    // evaluate to true, even in simple cases. example in
    // 641.leela_s:UCTNode.cpp

    // ref-count the argument
    this->val_ref(inst->getOperand(0)).part(0);
    auto const_ref = ValuePartRef{this, 0, 1, Config::GP_BANK};
    this->result_ref(inst).part(0).set_value(std::move(const_ref));
    return true;
  }
  default: {
    return derived()->handle_intrin(inst);
  }
  }
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_is_fpclass(
    const llvm::IntrinsicInst *inst) noexcept {
  auto *op = inst->getOperand(0);
  auto *op_ty = op->getType();

  if (!op_ty->isFloatTy() && !op_ty->isDoubleTy()) {
    return false;
  }
  const auto is_double = op_ty->isDoubleTy();
  const auto test =
      llvm::dyn_cast<llvm::ConstantInt>(inst->getOperand(1))->getZExtValue();

  enum {
    SIGNALING_NAN = 1 << 0,
    QUIET_NAN = 1 << 1,
    NEG_INF = 1 << 2,
    NEG_NORM = 1 << 3,
    NEG_SUBNORM = 1 << 4,
    NEG_ZERO = 1 << 5,
    POS_ZERO = 1 << 6,
    POS_SUBNORM = 1 << 7,
    POS_NORM = 1 << 8,
    POS_INF = 1 << 9,

    IS_NAN = SIGNALING_NAN | QUIET_NAN,
    IS_INF = NEG_INF | POS_INF,
    IS_NORM = NEG_NORM | POS_NORM,
    IS_FINITE =
        NEG_NORM | NEG_SUBNORM | NEG_ZERO | POS_ZERO | POS_SUBNORM | POS_NORM,
  };

  auto [res_vr, res_ref] = this->result_ref_single(inst);
  auto [op_vr, op_ref] = this->val_ref_single(op);

  auto zero_ref = ValuePartRef{this, 0, 4, Config::GP_BANK};

  // handle common case
#define TEST(cond, name)                                                       \
  if (test == cond) {                                                          \
    if (is_double) {                                                           \
      derived()->encode_is_fpclass_##name##_double(                            \
          std::move(zero_ref), std::move(op_ref), res_ref);                    \
    } else {                                                                   \
      derived()->encode_is_fpclass_##name##_float(                             \
          std::move(zero_ref), std::move(op_ref), res_ref);                    \
    }                                                                          \
    return true;                                                               \
  }

  TEST(IS_NAN, nan)
  TEST(IS_INF, inf)
  TEST(IS_NORM, norm)
#undef TEST

  // we OR' together the results from each test so initialize the result with
  // zero
  ValuePartRef res_scratch{derived(), Config::GP_BANK};
  zero_ref.reload_into_specific_fixed(res_scratch.alloc_reg());

  op_ref.load_to_reg();

  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ValuePart &);
  static constexpr auto encode_fns = []() {
    return std::array<EncodeFnTy[2], 10>{{
#define TEST(name)                                                             \
  {&Derived::encode_is_fpclass_##name##_float,                                 \
   &Derived::encode_is_fpclass_##name##_double},
        TEST(snan) TEST(qnan) TEST(ninf) TEST(nnorm) TEST(nsnorm) TEST(nzero)
            TEST(pzero) TEST(psnorm) TEST(pnorm) TEST(pinf)
#undef TEST
    }};
  }();

  for (unsigned i = 0; i < encode_fns.size(); i++) {
    if (test & (1 << i)) {
      // note that the std::move(res_scratch) here creates a new ValuePart that
      // manages the register inside the GenericValuePart and res_scratch
      // becomes invalid by the time the encode function is entered
      (derived()->*encode_fns[i][is_double])(
          std::move(res_scratch), op_ref.get_unowned_ref(), res_scratch);
    }
  }

  res_ref.set_value(std::move(res_scratch));
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_overflow_intrin(
    const llvm::IntrinsicInst *inst, OverflowOp op) noexcept {
  ValueRef lhs = this->val_ref(inst->getOperand(0));
  ValueRef rhs = this->val_ref(inst->getOperand(1));
  ValueRef res = this->result_ref(inst);

  auto *ty = inst->getOperand(0)->getType();
  if (!ty->isIntegerTy()) {
    return false;
  }
  const auto width = ty->getIntegerBitWidth();

  if (width == 128) {
    if (!derived()->handle_overflow_intrin_128(op,
                                               lhs.part(0),
                                               lhs.part(1),
                                               rhs.part(0),
                                               rhs.part(1),
                                               res.part(0),
                                               res.part(1),
                                               res.part(2))) {
      return false;
    }
    return true;
  }

  u32 width_idx = 0;
  switch (width) {
  case 8: width_idx = 0; break;
  case 16: width_idx = 1; break;
  case 32: width_idx = 2; break;
  case 64: width_idx = 3; break;
  default: return false;
  }

  using EncodeFnTy = bool (Derived::*)(
      GenericValuePart &&, GenericValuePart &&, ValuePart &&, ValuePart &&);
  std::array<std::array<EncodeFnTy, 4>, 6> encode_fns = {
      {
       {&Derived::encode_of_add_u8,
           &Derived::encode_of_add_u16,
           &Derived::encode_of_add_u32,
           &Derived::encode_of_add_u64},
       {&Derived::encode_of_add_i8,
           &Derived::encode_of_add_i16,
           &Derived::encode_of_add_i32,
           &Derived::encode_of_add_i64},
       {&Derived::encode_of_sub_u8,
           &Derived::encode_of_sub_u16,
           &Derived::encode_of_sub_u32,
           &Derived::encode_of_sub_u64},
       {&Derived::encode_of_sub_i8,
           &Derived::encode_of_sub_i16,
           &Derived::encode_of_sub_i32,
           &Derived::encode_of_sub_i64},
       {&Derived::encode_of_mul_u8,
           &Derived::encode_of_mul_u16,
           &Derived::encode_of_mul_u32,
           &Derived::encode_of_mul_u64},
       {&Derived::encode_of_mul_i8,
           &Derived::encode_of_mul_i16,
           &Derived::encode_of_mul_i32,
           &Derived::encode_of_mul_i64},
       }
  };

  EncodeFnTy encode_fn = encode_fns[static_cast<u32>(op)][width_idx];
  (derived()->*encode_fn)(lhs.part(0), rhs.part(0), res.part(0), res.part(1));
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_saturating_intrin(
    const llvm::IntrinsicInst *inst, OverflowOp op) noexcept {
  auto *ty = inst->getType();
  if (!ty->isIntegerTy()) {
    return false;
  }

  const auto width = ty->getIntegerBitWidth();
  u32 width_idx = 0;
  switch (width) {
  case 8: width_idx = 0; break;
  case 16: width_idx = 1; break;
  case 32: width_idx = 2; break;
  case 64: width_idx = 3; break;
  default: return false;
  }

  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ValuePart &&);
  std::array<std::array<EncodeFnTy, 4>, 4> encode_fns{
      {
       {&Derived::encode_sat_add_u8,
           &Derived::encode_sat_add_u16,
           &Derived::encode_sat_add_u32,
           &Derived::encode_sat_add_u64},
       {&Derived::encode_sat_add_i8,
           &Derived::encode_sat_add_i16,
           &Derived::encode_sat_add_i32,
           &Derived::encode_sat_add_i64},
       {&Derived::encode_sat_sub_u8,
           &Derived::encode_sat_sub_u16,
           &Derived::encode_sat_sub_u32,
           &Derived::encode_sat_sub_u64},
       {&Derived::encode_sat_sub_i8,
           &Derived::encode_sat_sub_i16,
           &Derived::encode_sat_sub_i32,
           &Derived::encode_sat_sub_i64},
       }
  };

  EncodeFnTy encode_fn = encode_fns[static_cast<u32>(op)][width_idx];

  ValueRef lhs = this->val_ref(inst->getOperand(0));
  ValueRef rhs = this->val_ref(inst->getOperand(1));
  ValueRef res = this->result_ref(inst);
  return (derived()->*encode_fn)(lhs.part(0), rhs.part(0), res.part(0));
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_vector_reduce(
    const llvm::IntrinsicInst *inst, const ValInfo &info) noexcept {
  if (inst->getType()->isIntegerTy(1)) {
    // i1 needs special handling
    // and/mul/umin/smax = all bits one
    // or/umax/smin = any bit one
    // xor/add = parity
    return false;
  }

  LLVMBasicValType elem_ty = info.type;

  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ValuePart &);
  EncodeFnTy fn;
  bool has_start_elem = false;
  switch (inst->getIntrinsicID()) {
  case llvm::Intrinsic::vector_reduce_add:
    if (elem_ty == LLVMBasicValType::i64) {
      fn = &Derived::encode_addi64;
    } else {
      fn = &Derived::encode_addi32;
    }
    break;
  case llvm::Intrinsic::vector_reduce_fadd:
    if (elem_ty == LLVMBasicValType::f64) {
      fn = &Derived::encode_addf64;
    } else {
      fn = &Derived::encode_addf32;
    }
    has_start_elem = true;
    break;
  case llvm::Intrinsic::vector_reduce_mul:
    if (elem_ty == LLVMBasicValType::i64) {
      fn = &Derived::encode_muli64;
    } else {
      fn = &Derived::encode_muli32;
    }
    break;
  case llvm::Intrinsic::vector_reduce_fmul:
    if (elem_ty == LLVMBasicValType::f64) {
      fn = &Derived::encode_mulf64;
    } else {
      fn = &Derived::encode_mulf32;
    }
    has_start_elem = true;
    break;
  case llvm::Intrinsic::vector_reduce_and:
    if (elem_ty == LLVMBasicValType::i64) {
      fn = &Derived::encode_landi64;
    } else {
      fn = &Derived::encode_landi32;
    }
    break;
  case llvm::Intrinsic::vector_reduce_or:
    if (elem_ty == LLVMBasicValType::i64) {
      fn = &Derived::encode_lori64;
    } else {
      fn = &Derived::encode_lori32;
    }
    break;
  case llvm::Intrinsic::vector_reduce_xor:
    if (elem_ty == LLVMBasicValType::i64) {
      fn = &Derived::encode_lxori64;
    } else {
      fn = &Derived::encode_lxori32;
    }
    break;
  default:
    // Still missing: smin/smax/umin/umix/fmin/fmax/fminimum/fmaximum
    return false;
  }

  llvm::Value *src_op = inst->getOperand(has_start_elem ? 1 : 0);
  ValueRef src_ref = this->val_ref(src_op);
  ValueRef src_ref_disowned = src_ref.disowned();

  ValuePartRef elem{this, LLVMAdaptor::basic_ty_part_bank(elem_ty)};
  ValuePartRef acc{this, LLVMAdaptor::basic_ty_part_bank(elem_ty)};

  if (has_start_elem) {
    acc.set_value(this->val_ref(inst->getOperand(0)).part(0));
  } else {
    derived()->extract_element(src_ref_disowned, 0, elem_ty, acc);
  }

  auto *vec_ty = llvm::cast<llvm::FixedVectorType>(src_op->getType());
  unsigned nelem = vec_ty->getNumElements();
  for (unsigned i = has_start_elem ? 0 : 1; i != nelem; i++) {
    derived()->extract_element(src_ref_disowned, i, elem_ty, elem);
    (derived()->*fn)(std::move(acc), std::move(elem), acc);
  }

  this->result_ref(inst).part(0).set_value(std::move(acc));

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_to_elf(
    llvm::Module &mod, std::vector<uint8_t> &buf) noexcept {
  if (this->adaptor->mod) {
    derived()->reset();
  }
  if (!compile(mod)) {
    return false;
  }

  llvm::TimeTraceScope time_scope("TPDE_EmitObj");
  buf = this->assembler.build_object_file();
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
JITMapper LLVMCompilerBase<Adaptor, Derived, Config>::compile_and_map(
    llvm::Module &mod,
    std::function<void *(std::string_view)> resolver) noexcept {
  if (this->adaptor->mod) {
    derived()->reset();
  }
  if (!compile(mod)) {
    return JITMapper{nullptr};
  }

  auto res = std::make_unique<JITMapperImpl>(std::move(global_syms));
  if (!res->map(this->assembler, resolver)) {
    return JITMapper{nullptr};
  }

  return JITMapper{std::move(res)};
}

} // namespace tpde_llvm
