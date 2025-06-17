// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <elf.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/IR/Comdat.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalIFunc.h>
#include <llvm/IR/GlobalObject.h>
#include <llvm/IR/GlobalValue.h>
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


#include "tpde/CompilerBase.hpp"
#include "tpde/ValLocalIdx.hpp"
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

  using Assembler = typename Base::Assembler;
  using SecRef = typename Assembler::SecRef;
  using SymRef = typename Assembler::SymRef;

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

  std::optional<ValuePartRef> val_ref_constant(IRValueRef val_idx,
                                               u32 part) noexcept;

  std::optional<ValRefSpecial> val_ref_special(IRValueRef value) noexcept {
    if (llvm::isa<llvm::Constant>(value)) {
      return val_ref_constant(value);
    }
    return std::nullopt;
  }

  ValuePartRef val_part_ref_special(ValRefSpecial &vrs, u32 part) noexcept {
    return *val_ref_constant(vrs.value, part);
  }

  ValRefSpecial val_ref_constant(IRValueRef value) noexcept {
    return ValRefSpecial{.value = value};
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

private:
  static typename Assembler::SymBinding
      convert_linkage(const llvm::GlobalValue *gv) noexcept {
    if (gv->hasLocalLinkage()) {
      return Assembler::SymBinding::LOCAL;
    } else if (gv->isWeakForLinker()) {
      return Assembler::SymBinding::WEAK;
    }
    return Assembler::SymBinding::GLOBAL;
  }

  static typename Assembler::SymVisibility
      convert_visibility(const llvm::GlobalValue *gv) noexcept {
    switch (gv->getVisibility()) {
    case llvm::GlobalValue::DefaultVisibility:
      return Assembler::SymVisibility::DEFAULT;
    case llvm::GlobalValue::HiddenVisibility:
      return Assembler::SymVisibility::HIDDEN;
    case llvm::GlobalValue::ProtectedVisibility:
      return Assembler::SymVisibility::PROTECTED;
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
    // Reuse/release memory for stored constants from previous function
    const_allocator.reset();

    SecRef sec = this->select_section(this->func_syms[idx], func, true);
    if (sec == Assembler::INVALID_SEC_REF) {
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
    return Base::compile_func(func, idx) && !this->adaptor->func_unsupported;
  }

  bool compile(llvm::Module &mod) noexcept;

  bool compile_unknown(const llvm::Instruction *,
                       const ValInfo &,
                       u64) noexcept {
    return false;
  }

  bool compile_inst(const llvm::Instruction *, InstRange) noexcept;

  bool compile_ret(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_load_generic(const llvm::LoadInst *,
                            GenericValuePart &&) noexcept;
  bool compile_load(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_store_generic(const llvm::StoreInst *,
                             GenericValuePart &&) noexcept;
  bool compile_store(const llvm::Instruction *, const ValInfo &, u64) noexcept;
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

  void extract_element(IRValueRef vec,
                       unsigned idx,
                       LLVMBasicValType ty,
                       ScratchReg &out) noexcept;
  void insert_element(ValuePart &vec_ref,
                      unsigned idx,
                      LLVMBasicValType ty,
                      GenericValuePart el) noexcept;
  bool compile_extract_element(const llvm::Instruction *,
                               const ValInfo &,
                               u64) noexcept;
  bool compile_insert_element(const llvm::Instruction *,
                              const ValInfo &,
                              u64) noexcept;
  bool compile_shuffle_vector(const llvm::Instruction *,
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
  bool compile_gep(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_fcmp(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_switch(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_invoke(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_landing_pad(const llvm::Instruction *,
                           const ValInfo &,
                           u64) noexcept;
  bool compile_resume(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  SymRef lookup_type_info_sym(IRValueRef value) noexcept;
  bool compile_intrin(const llvm::IntrinsicInst *, const ValInfo &) noexcept;
  bool compile_is_fpclass(const llvm::IntrinsicInst *) noexcept;
  bool compile_overflow_intrin(const llvm::IntrinsicInst *,
                               OverflowOp) noexcept;
  bool compile_saturating_intrin(const llvm::IntrinsicInst *,
                                 OverflowOp) noexcept;

  bool compile_unreachable(const llvm::UnreachableInst *) noexcept {
    return false;
  }

  bool compile_alloca(const llvm::AllocaInst *) noexcept { return false; }

  bool compile_br(const llvm::Instruction *, const ValInfo &, u64) noexcept {
    return false;
  }

  bool compile_inline_asm(const llvm::CallBase *) { return false; }

  bool compile_icmp(const llvm::ICmpInst *, InstRange) noexcept {
    return false;
  }

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
  if (llvm::timeTraceProfilerEnabled()) {
    llvm::timeTraceProfilerEnd(time_entry);
    time_entry = llvm::timeTraceProfilerBegin("TPDE_CodeGen", "");
  }
}

template <typename Adaptor, typename Derived, typename Config>
std::optional<typename LLVMCompilerBase<Adaptor, Derived, Config>::ValuePartRef>
    LLVMCompilerBase<Adaptor, Derived, Config>::val_ref_constant(
        IRValueRef val, u32 part) noexcept {
  auto *const_val = llvm::cast<llvm::Constant>(val);

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
    TPDE_FATAL("cannot handle constant of illegal/unsupported type");
  }

  // At this point, ty is the basic type of the element and sub_part the part
  // inside the basic type.

  if (auto *gv = llvm::dyn_cast<llvm::GlobalValue>(const_val)) {
    assert(ty == LLVMBasicValType::ptr && sub_part == 0);
    auto local_idx = tpde::ValLocalIdx(this->adaptor->global_lookup.lookup(gv));
    auto *assignment = this->val_assignment(local_idx);
    if (!assignment) {
      this->init_variable_ref(local_idx, static_cast<u32>(local_idx));
      assignment = this->val_assignment(local_idx);
    }
    return ValuePartRef{this, local_idx, assignment, 0, /*owned=*/false};
  }

  if (llvm::isa<llvm::PoisonValue>(const_val) ||
      llvm::isa<llvm::UndefValue>(const_val) ||
      llvm::isa<llvm::ConstantPointerNull>(const_val) ||
      llvm::isa<llvm::ConstantAggregateZero>(const_val)) {
    u32 size = this->adaptor->basic_ty_part_size(ty);
    tpde::RegBank bank = this->adaptor->basic_ty_part_bank(ty);
    static const std::array<u64, 8> zero{};
    assert(size <= zero.size() * sizeof(u64));
    return ValuePartRef(this, zero.data(), size, bank);
  }

  if (auto *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(const_val)) {
    if (ty == LLVMBasicValType::invalid) {
      TPDE_FATAL("illegal vector constant of unsupported type");
    }
    assert(part == 0 && "multi-part vector constants not implemented");
    llvm::StringRef data = cdv->getRawDataValues();
    // TODO: this cast is actually invalid.
    const u64 *data_ptr = reinterpret_cast<const u64 *>(data.data());
    return ValuePartRef(this, data_ptr, data.size(), Config::FP_BANK);
  }

  if (llvm::isa<llvm::ConstantVector>(const_val)) {
    // TODO(ts): check how to handle this
    TPDE_FATAL("non-sequential vector constants should not be legal");
  }

  if (const auto *const_int = llvm::dyn_cast<llvm::ConstantInt>(const_val)) {
    assert(sub_part < (ty == LLVMBasicValType::i128 ? 2 : 1));
    u32 size = this->adaptor->basic_ty_part_size(ty);
    tpde::RegBank bank = this->adaptor->basic_ty_part_bank(ty);
    assert(size <= 8 && "multi-word integer as single part?");
    const u64 *data = const_int->getValue().getRawData();
    return ValuePartRef(this, data[sub_part], size, bank);
  }

  if (const auto *const_fp = llvm::dyn_cast<llvm::ConstantFP>(const_val);
      const_fp != nullptr) {
    // APFloat has no bitwise storage of the floating-point number and
    // bitcastToAPInt constructs a new APInt, so we need to copy the value.
    llvm::APInt int_val = const_fp->getValue().bitcastToAPInt();
    u64 *data = new (const_allocator) u64[int_val.getNumWords()];
    std::memcpy(
        data, int_val.getRawData(), int_val.getNumWords() * sizeof(u64));

    u32 size = this->adaptor->basic_ty_part_size(ty);
    tpde::RegBank bank = this->adaptor->basic_ty_part_bank(ty);
    assert(size <= int_val.getNumWords() * sizeof(u64));
    return ValuePartRef(this, data, size, bank);
  }

  std::string const_str;
  llvm::raw_string_ostream(const_str) << *const_val;
  TPDE_LOG_ERR("encountered unhandled constant {}", const_str);
  TPDE_FATAL("unhandled constant type");
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
    return Assembler::INVALID_SEC_REF;
  }

  bool is_comdat;
  switch (comdat->getSelectionKind()) {
  case llvm::Comdat::Any: is_comdat = true; break;
  case llvm::Comdat::NoDeduplicate: is_comdat = false; break;
  default:
    // ELF only support any/nodeduplicate.
    return Assembler::INVALID_SEC_REF;
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
          this->assembler.sym_add_undef(cn, Assembler::SymBinding::LOCAL);
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
    return Assembler::INVALID_SEC_REF;
  }

  // I'm certain this simplified section assignment code is buggy...
  bool tls = false;
  bool read_only = true;
  bool init_zero = false;
  bool is_func = llvm::isa<llvm::Function>(go);
  bool retain = used_globals.contains(go);
  if (auto *gv = llvm::dyn_cast<llvm::GlobalVariable>(go)) {
    tls = gv->isThreadLocal();
    read_only = gv->isConstant();
    init_zero = gv->getInitializer()->isNullValue();
    assert((!init_zero || !needs_relocs) &&
           "zero-initialized sections must not have relocations");
  }

  llvm::StringRef sec_name = go->getSection();
  const llvm::Comdat *comdat = go->getComdat();

  // If the section name is empty, use the default section.
  if (!retain && !comdat && sec_name.empty()) [[likely]] {
    if (is_func) {
      return this->assembler.get_text_section();
    }

    if (tls) {
      return init_zero ? this->assembler.get_tbss_section()
                       : this->assembler.get_tdata_section();
    }

    if (!read_only && init_zero) {
      return this->assembler.get_bss_section();
    }
    return this->assembler.get_data_section(read_only, needs_relocs);
  }

  SecRef group_sec = get_group_section(go, sym);

  llvm::StringRef def_name;
  unsigned type = SHT_PROGBITS;
  unsigned flags = SHF_ALLOC;
  if (is_func) {
    def_name = ".text";
    flags |= SHF_EXECINSTR;
  } else if (tls) {
    def_name = init_zero ? llvm::StringRef(".tbss") : llvm::StringRef(".tdata");
    type = init_zero ? SHT_NOBITS : SHT_PROGBITS;
    flags |= SHF_WRITE | SHF_TLS;
  } else if (!read_only && init_zero) {
    def_name = ".bss";
    type = SHT_NOBITS;
    flags |= SHF_WRITE;
  } else if (!read_only) {
    def_name = ".data";
    flags |= SHF_WRITE;
  } else if (needs_relocs) {
    def_name = ".data.rel.ro";
    flags |= SHF_WRITE;
  } else {
    def_name = ".rodata";
  }

  if (retain) {
    flags |= SHF_GNU_RETAIN;
  }

  if (sec_name.empty()) {
    sec_name = def_name; // Use default prefix
  }

  llvm::SmallString<512> name_buf;
  if (comdat) {
    llvm::StringRef go_name = go->getName();
    name_buf.reserve(sec_name.size() + 1 + go_name.size());
    name_buf.append(sec_name);
    name_buf.push_back('.');
    // TODO: apply LLVM's name mangling.
    // TODO: do we need to include the platform prefix (_ on Darwin) here?
    name_buf.append(go_name);
    sec_name = name_buf.str();
  }

  // TODO: is it *required* that we merge sections here? For now, don't.

  return this->assembler.create_section(
      sec_name, type, flags, needs_relocs, group_sec);
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
    // TODO: name mangling
    if (!gv.hasName()) {
      TPDE_LOG_ERR("unnamed globals are not implemented");
      return false;
    }

    llvm::StringRef name = gv.getName();

    if (gv.hasAppendingLinkage()) [[unlikely]] {
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

    auto binding = convert_linkage(&gv);
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
          return std::tie(group, priority) < std::tie(rhs.group, rhs.priority);
        }
      };
      tpde::util::SmallVector<Structor, 16> structors;

      // see
      // https://llvm.org/docs/LangRef.html#the-llvm-global-ctors-global-variable
      for (auto &entry : llvm::cast<llvm::ConstantArray>(init)->operands()) {
        const auto *str = llvm::cast<llvm::ConstantStruct>(entry);
        auto *prio = llvm::cast<llvm::ConstantInt>(str->getOperand(0));
        auto *ptr = llvm::cast<llvm::GlobalValue>(str->getOperand(1));
        SecRef group = Assembler::INVALID_SEC_REF;
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

      SecRef secref = Assembler::INVALID_SEC_REF;
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
    if (sec == Assembler::INVALID_SEC_REF) {
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
  const auto alloc_size = layout.getTypeAllocSize(constant->getType());
  assert(off + alloc_size <= data.size());

  // Handle all-zero values quickly.
  if (constant->isNullValue() || llvm::isa<llvm::UndefValue>(constant)) {
    return true;
  }

  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(constant); CI) {
    // TODO: endianness?
    llvm::StoreIntToMemory(CI->getValue(), data.data() + off, alloc_size);
    return true;
  }
  if (auto *CF = llvm::dyn_cast<llvm::ConstantFP>(constant); CF) {
    // TODO: endianness?
    llvm::StoreIntToMemory(
        CF->getValue().bitcastToAPInt(), data.data() + off, alloc_size);
    return true;
  }
  if (auto *CDS = llvm::dyn_cast<llvm::ConstantDataSequential>(constant); CDS) {
    auto d = CDS->getRawDataValues();
    assert(d.size() <= alloc_size);
    std::copy(d.bytes_begin(), d.bytes_end(), data.begin() + off);
    return true;
  }
  if (auto *CA = llvm::dyn_cast<llvm::ConstantArray>(constant); CA) {
    const auto num_elements = CA->getType()->getNumElements();
    const auto element_size =
        layout.getTypeAllocSize(CA->getType()->getElementType());
    bool success = true;
    for (auto i = 0u; i < num_elements; ++i) {
      success &= global_init_to_data(reloc_base,
                                     data,
                                     relocs,
                                     layout,
                                     CA->getAggregateElement(i),
                                     off + i * element_size);
    }
    return success;
  }
  if (auto *CA = llvm::dyn_cast<llvm::ConstantAggregate>(constant); CA) {
    const auto num_elements = CA->getType()->getStructNumElements();
    auto &ctx = constant->getContext();

    auto *ty = CA->getType();
    auto c0 = llvm::ConstantInt::get(ctx, llvm::APInt(32, 0, false));

    bool success = true;
    for (auto i = 0u; i < num_elements; ++i) {
      auto idx = llvm::ConstantInt::get(ctx, llvm::APInt(32, (u64)i, false));
      auto agg_off = layout.getIndexedOffsetInType(ty, {c0, idx});
      success &= global_init_to_data(reloc_base,
                                     data,
                                     relocs,
                                     layout,
                                     CA->getAggregateElement(i),
                                     off + agg_off);
    }
    return success;
  }
  if (auto *GV = llvm::dyn_cast<llvm::GlobalValue>(constant); GV) {
    assert(alloc_size == 8);
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
        auto alloc_size = layout.getTypeAllocSize(expr->getType());
        // TODO: endianness?
        llvm::StoreIntToMemory(CI->getValue(), data.data() + off, alloc_size);
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

  TPDE_LOG_ERR("Encountered unknown constant in global initializer");
  llvm::errs() << *constant << "\n";
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

  sym = this->assembler.sym_add_undef(name, Assembler::SymBinding::GLOBAL);
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
      assert(0);
      continue;
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
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_ret(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  typename Derived::RetBuilder rb{*derived(), *derived()->cur_cc_assigner()};
  if (inst->getNumOperands() != 0) {
    llvm::Value *retval = inst->getOperand(0);
    bool handled = false;
    if (auto ret_ty = retval->getType(); ret_ty->isIntegerTy()) {
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
  auto res = this->result_ref(load);
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
    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ScratchReg &);
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

    ScratchReg res_scratch{derived()};
    if (!(derived()->*encode_fn)(std::move(ptr_op), res_scratch)) {
      return false;
    }

    ValuePartRef res_part = res.part(0);
    this->set_value(res_part, res_scratch);
    return true;
  }

  ScratchReg res_scratch{this};
  switch (this->adaptor->val_info(load).type) {
    using enum LLVMBasicValType;
  case i1:
  case i8:
  case i16:
  case i32:
  case i64: {
    assert(load->getType()->isIntegerTy());
    const auto bit_width = load->getType()->getIntegerBitWidth();
    switch (bit_width) {
    case 1:
    case 8: derived()->encode_loadi8(std::move(ptr_op), res_scratch); break;
    case 16: derived()->encode_loadi16(std::move(ptr_op), res_scratch); break;
    case 24: derived()->encode_loadi24(std::move(ptr_op), res_scratch); break;
    case 32: derived()->encode_loadi32(std::move(ptr_op), res_scratch); break;
    case 40: derived()->encode_loadi40(std::move(ptr_op), res_scratch); break;
    case 48: derived()->encode_loadi48(std::move(ptr_op), res_scratch); break;
    case 56: derived()->encode_loadi56(std::move(ptr_op), res_scratch); break;
    case 64: derived()->encode_loadi64(std::move(ptr_op), res_scratch); break;
    default: assert(0); return false;
    }
    break;
  }
  case ptr: {
    derived()->encode_loadi64(std::move(ptr_op), res_scratch);
    break;
  }
  case i128: {
    ScratchReg res_scratch_high{derived()};
    auto res_low = res.part(0);
    auto res_high = res.part(1);

    derived()->encode_loadi128(
        std::move(ptr_op), res_scratch, res_scratch_high);
    this->set_value(res_low, res_scratch);
    this->set_value(res_high, res_scratch_high);
    return true;
  }
  case v32:
  case f32: {
    derived()->encode_loadf32(std::move(ptr_op), res_scratch);
    break;
  }
  case v64:
  case f64: {
    derived()->encode_loadf64(std::move(ptr_op), res_scratch);
    break;
  }
  case v128:
  case f128: derived()->encode_loadv128(std::move(ptr_op), res_scratch); break;
  case complex: {
    auto ty_idx = this->adaptor->val_info(load).complex_part_tys_idx;
    const LLVMComplexPart *part_descs =
        &this->adaptor->complex_part_types[ty_idx + 1];
    unsigned part_count = part_descs[-1].desc.num_parts;

    // TODO: fuse expr; not easy, because we lose the GVP
    AsmReg ptr_reg = this->gval_as_reg(ptr_op);

    unsigned off = 0;
    for (unsigned i = 0; i < part_count; i++) {
      auto part_ref = res.part(i);
      auto part_addr =
          typename GenericValuePart::Expr{ptr_reg, static_cast<tpde::i32>(off)};
      auto part_ty = part_descs[i].part.type;
      switch (part_ty) {
      case i1:
      case i8:
        derived()->encode_loadi8(std::move(part_addr), res_scratch);
        break;
      case i16:
        derived()->encode_loadi16(std::move(part_addr), res_scratch);
        break;
      case i32:
        derived()->encode_loadi32(std::move(part_addr), res_scratch);
        break;
      case i64:
      case ptr:
        derived()->encode_loadi64(std::move(part_addr), res_scratch);
        break;
      case i128:
        derived()->encode_loadi64(std::move(part_addr), res_scratch);
        break;
      case v32:
      case f32:
        derived()->encode_loadf32(std::move(part_addr), res_scratch);
        break;
      case v64:
      case f64:
        derived()->encode_loadf64(std::move(part_addr), res_scratch);
        break;
      case v128:
      case f128:
        derived()->encode_loadv128(std::move(part_addr), res_scratch);
        break;
      default: assert(0); return false;
      }

      off += part_descs[i].part.size + part_descs[i].part.pad_after;

      this->set_value(part_ref, res_scratch);
    }
    return true;
  }
  default: return false;
  }

  ValuePartRef res_ref = res.part(0);
  this->set_value(res_ref, res_scratch);

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

  switch (ty) {
    using enum LLVMBasicValType;
  case i1:
  case i8:
  case i16:
  case i32:
  case i64: {
    assert(op_val->getType()->isIntegerTy());
    const auto bit_width = op_val->getType()->getIntegerBitWidth();
    switch (bit_width) {
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
    default: assert(0); return false;
    }
    break;
  }
  case ptr:
    derived()->encode_storei64(std::move(ptr_op), op_ref.part(0));
    break;
  case i128:
    derived()->encode_storei128(
        std::move(ptr_op), op_ref.part(0), op_ref.part(1));
    break;
  case v32:
  case f32:
    derived()->encode_storef32(std::move(ptr_op), op_ref.part(0));
    break;
  case v64:
  case f64:
    derived()->encode_storef64(std::move(ptr_op), op_ref.part(0));
    break;
  case v128:
  case f128:
    derived()->encode_storev128(std::move(ptr_op), op_ref.part(0));
    break;
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
        derived()->encode_storei8(std::move(part_addr), std::move(part_ref));
        break;
      case i16:
        derived()->encode_storei16(std::move(part_addr), std::move(part_ref));
        break;
      case i32:
        derived()->encode_storei32(std::move(part_addr), std::move(part_ref));
        break;
      case i64:
      case ptr:
        derived()->encode_storei64(std::move(part_addr), std::move(part_ref));
        break;
      case i128:
        derived()->encode_storei64(std::move(part_addr), std::move(part_ref));
        break;
      case v32:
      case f32:
        derived()->encode_storef32(std::move(part_addr), std::move(part_ref));
        break;
      case v64:
      case f64:
        derived()->encode_storef64(std::move(part_addr), std::move(part_ref));
        break;
      case v128:
      case f128:
        derived()->encode_storev128(std::move(part_addr), std::move(part_ref));
        break;
      default: assert(0); return false;
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
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_int_binary_op(
    const llvm::Instruction *inst, const ValInfo &info, u64 op_val) noexcept {
  auto *inst_ty = inst->getType();
  IntBinaryOp op = typename IntBinaryOp::Value(op_val);

  if (inst_ty->isVectorTy()) [[unlikely]] {
    auto *scalar_ty = inst_ty->getScalarType();
    auto int_width = scalar_ty->getIntegerBitWidth();

    using EncodeFnTy = bool (Derived::*)(
        GenericValuePart &&, GenericValuePart &&, ScratchReg &);
    // fns[op.index()][v64=0/v128=1][8=0/16=1/32=2/64=3]
    static constexpr auto fns = []() constexpr {
      std::array<EncodeFnTy[2][4], IntBinaryOp::num_ops> res{};
      auto entry = [&res](IntBinaryOp op) { return res[op.index()]; };

      // TODO: more consistent naming of encode functions
#define FN_ENTRY(opname, fnbase, sign)                                         \
  entry(opname)[0][0] = &Derived::encode_##fnbase##v8##sign##8;                \
  entry(opname)[0][1] = &Derived::encode_##fnbase##v4##sign##16;               \
  entry(opname)[0][2] = &Derived::encode_##fnbase##v2##sign##32;               \
  entry(opname)[1][0] = &Derived::encode_##fnbase##v16##sign##8;               \
  entry(opname)[1][1] = &Derived::encode_##fnbase##v8##sign##16;               \
  entry(opname)[1][2] = &Derived::encode_##fnbase##v4##sign##32;               \
  entry(opname)[1][3] = &Derived::encode_##fnbase##v2##sign##64;

      FN_ENTRY(IntBinaryOp::add, add, u)
      FN_ENTRY(IntBinaryOp::sub, sub, u)
      FN_ENTRY(IntBinaryOp::mul, mul, u)
      FN_ENTRY(IntBinaryOp::land, and, u)
      FN_ENTRY(IntBinaryOp::lxor, xor, u)
      FN_ENTRY(IntBinaryOp::lor, or, u)
      FN_ENTRY(IntBinaryOp::shl, shl, u)
      FN_ENTRY(IntBinaryOp::shr, lshr, u)
      FN_ENTRY(IntBinaryOp::ashr, ashr, i)
#undef FN_ENTRY

      return res;
    }();

    unsigned width_idx;
    switch (int_width) {
    case 8: width_idx = 0; break;
    case 16: width_idx = 1; break;
    case 32: width_idx = 2; break;
    case 64: width_idx = 3; break;
    default: return false;
    }

    unsigned ty_idx;
    switch (info.type) {
      using enum LLVMBasicValType;
    case v64: ty_idx = 0; break;
    case v128: ty_idx = 1; break;
    default: return false;
    }

    EncodeFnTy encode_fn = fns[op.index()][ty_idx][width_idx];
    if (!encode_fn) {
      return false;
    }

    auto lhs = this->val_ref(inst->getOperand(0));
    auto rhs = this->val_ref(inst->getOperand(1));
    auto [res_vr, res_ref] = this->result_ref_single(inst);
    ScratchReg res{this};
    if (!(derived()->*encode_fn)(lhs.part(0), rhs.part(0), res)) {
      return false;
    }
    this->set_value(res_ref, res);
    return true;
  }

  assert(inst_ty->isIntegerTy());

  const auto int_width = inst_ty->getIntegerBitWidth();
  // Storage for encode immediates
  u64 imm1, imm2;
  if (int_width == 128) {
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

    ScratchReg scratch_low{derived()}, scratch_high{derived()};

    if (op.is_shift()) {
      ValuePartRef shift_amt = rhs.part(0);
      if (shift_amt.is_const()) {
        imm1 = shift_amt.const_data()[0] & 0b111'1111; // amt
        if (imm1 < 64) {
          imm2 = (64 - imm1) & 0b11'1111; // iamt
          if (op == IntBinaryOp::shl) {
            derived()->encode_shli128_lt64(
                lhs.part(0),
                lhs.part(1),
                ValuePartRef(this, imm1, 1, Config::GP_BANK),
                ValuePartRef(this, imm2, 1, Config::GP_BANK),
                scratch_low,
                scratch_high);
          } else if (op == IntBinaryOp::shr) {
            derived()->encode_shri128_lt64(
                lhs.part(0),
                lhs.part(1),
                ValuePartRef(this, imm1, 1, Config::GP_BANK),
                ValuePartRef(this, imm2, 1, Config::GP_BANK),
                scratch_low,
                scratch_high);
          } else {
            assert(op == IntBinaryOp::ashr);
            derived()->encode_ashri128_lt64(
                lhs.part(0),
                lhs.part(1),
                ValuePartRef(this, imm1, 1, Config::GP_BANK),
                ValuePartRef(this, imm2, 1, Config::GP_BANK),
                scratch_low,
                scratch_high);
          }
        } else {
          imm1 -= 64;
          if (op == IntBinaryOp::shl) {
            derived()->encode_shli128_ge64(
                lhs.part(0),
                ValuePartRef(this, imm1, 1, Config::GP_BANK),
                scratch_low,
                scratch_high);
          } else if (op == IntBinaryOp::shr) {
            derived()->encode_shri128_ge64(
                lhs.part(1),
                ValuePartRef(this, imm1, 1, Config::GP_BANK),
                scratch_low,
                scratch_high);
          } else {
            assert(op == IntBinaryOp::ashr);
            derived()->encode_ashri128_ge64(
                lhs.part(1),
                ValuePartRef(this, imm1, 1, Config::GP_BANK),
                scratch_low,
                scratch_high);
          }
        }
      } else {
        if (op == IntBinaryOp::shl) {
          derived()->encode_shli128(lhs.part(0),
                                    lhs.part(1),
                                    std::move(shift_amt),
                                    scratch_low,
                                    scratch_high);
        } else if (op == IntBinaryOp::shr) {
          derived()->encode_shri128(lhs.part(0),
                                    lhs.part(1),
                                    std::move(shift_amt),
                                    scratch_low,
                                    scratch_high);
        } else {
          assert(op == IntBinaryOp::ashr);
          derived()->encode_ashri128(lhs.part(0),
                                     lhs.part(1),
                                     std::move(shift_amt),
                                     scratch_low,
                                     scratch_high);
        }
      }
    } else {
      using EncodeFnTy = bool (Derived::*)(GenericValuePart &&,
                                           GenericValuePart &&,
                                           GenericValuePart &&,
                                           GenericValuePart &&,
                                           ScratchReg &,
                                           ScratchReg &);
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
                                              scratch_low,
                                              scratch_high);
    }

    this->set_value(res.part(0), scratch_low);
    this->set_value(res.part(1), scratch_high);

    return true;
  }
  assert(int_width <= 64);

  // encode functions for 32/64 bit operations
  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ScratchReg &);
  static const std::array<std::array<EncodeFnTy, 2>, 13> encode_ptrs{
      {
       {&Derived::encode_addi32, &Derived::encode_addi64},
       {&Derived::encode_subi32, &Derived::encode_subi64},
       {&Derived::encode_muli32, &Derived::encode_muli64},
       {&Derived::encode_udivi32, &Derived::encode_udivi64},
       {&Derived::encode_sdivi32, &Derived::encode_sdivi64},
       {&Derived::encode_uremi32, &Derived::encode_uremi64},
       {&Derived::encode_sremi32, &Derived::encode_sremi64},
       {&Derived::encode_landi32, &Derived::encode_landi64},
       {&Derived::encode_lori32, &Derived::encode_lori64},
       {&Derived::encode_lxori32, &Derived::encode_lxori64},
       {&Derived::encode_shli32, &Derived::encode_shli64},
       {&Derived::encode_shri32, &Derived::encode_shri64},
       {&Derived::encode_ashri32, &Derived::encode_ashri64},
       }
  };

  auto lhs = this->val_ref(inst->getOperand(0));
  auto rhs = this->val_ref(inst->getOperand(1));
  ValuePartRef lhs_op = lhs.part(0);
  ValuePartRef rhs_op = rhs.part(0);

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

  auto [res_vr, res] = this->result_ref_single(inst);

  auto res_scratch = ScratchReg{derived()};

  (derived()->*(encode_ptrs[op.index()][ext_width / 32 - 1]))(
      std::move(lhs_op), std::move(rhs_op), res_scratch);

  this->set_value(res, res_scratch);

  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_float_binary_op(
    const llvm::Instruction *inst, const ValInfo &val_info, u64 op) noexcept {
  auto *inst_ty = inst->getType();
  auto *scalar_ty = inst_ty->getScalarType();

  auto lhs = this->val_ref(inst->getOperand(0));
  auto rhs = this->val_ref(inst->getOperand(1));

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
    auto res_vr = this->result_ref(inst);
    cb->add_ret(res_vr);
    return true;
  }

  const bool is_double = scalar_ty->isDoubleTy();
  if (!scalar_ty->isFloatTy() && !scalar_ty->isDoubleTy()) {
    return false;
  }

  if (op == FloatBinaryOp::rem) {
    if (inst_ty->isVectorTy()) {
      return false;
    }

    auto cb = derived()->create_call_builder();
    cb->add_arg(lhs.part(0), tpde::CCAssignment{});
    cb->add_arg(rhs.part(0), tpde::CCAssignment{});
    cb->call(get_libfunc_sym(is_double ? LibFunc::fmod : LibFunc::fmodf));
    auto res_vr = this->result_ref(inst);
    cb->add_ret(res_vr);
    return true;
  }

  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ScratchReg &);
  EncodeFnTy encode_fn = nullptr;

  switch (val_info.type) {
    using enum LLVMBasicValType;
  case f32:
    assert(!is_double);
    switch (op) {
    case FloatBinaryOp::add: encode_fn = &Derived::encode_addf32; break;
    case FloatBinaryOp::sub: encode_fn = &Derived::encode_subf32; break;
    case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulf32; break;
    case FloatBinaryOp::div: encode_fn = &Derived::encode_divf32; break;
    default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
    }
    break;
  case f64:
    assert(is_double);
    switch (op) {
    case FloatBinaryOp::add: encode_fn = &Derived::encode_addf64; break;
    case FloatBinaryOp::sub: encode_fn = &Derived::encode_subf64; break;
    case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulf64; break;
    case FloatBinaryOp::div: encode_fn = &Derived::encode_divf64; break;
    default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
    }
    break;
  case v64:
    assert(!is_double);
    switch (op) {
    case FloatBinaryOp::add: encode_fn = &Derived::encode_addv2f32; break;
    case FloatBinaryOp::sub: encode_fn = &Derived::encode_subv2f32; break;
    case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulv2f32; break;
    case FloatBinaryOp::div: encode_fn = &Derived::encode_divv2f32; break;
    default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
    }
    break;
  case v128:
    if (!is_double) {
      switch (op) {
      case FloatBinaryOp::add: encode_fn = &Derived::encode_addv4f32; break;
      case FloatBinaryOp::sub: encode_fn = &Derived::encode_subv4f32; break;
      case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulv4f32; break;
      case FloatBinaryOp::div: encode_fn = &Derived::encode_divv4f32; break;
      default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
      }
    } else {
      switch (op) {
      case FloatBinaryOp::add: encode_fn = &Derived::encode_addv2f64; break;
      case FloatBinaryOp::sub: encode_fn = &Derived::encode_subv2f64; break;
      case FloatBinaryOp::mul: encode_fn = &Derived::encode_mulv2f64; break;
      case FloatBinaryOp::div: encode_fn = &Derived::encode_divv2f64; break;
      default: TPDE_UNREACHABLE("invalid FloatBinaryOp");
      }
    }
    break;
  default: TPDE_UNREACHABLE("invalid basic type for float binary op");
  }

  auto [res_vr, res] = this->result_ref_single(inst);
  ScratchReg res_scratch{derived()};
  if (!(derived()->*encode_fn)(lhs.part(0), rhs.part(0), res_scratch)) {
    return false;
  }
  this->set_value(res, res_scratch);
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_fneg(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  auto *scalar_ty = inst->getType()->getScalarType();

  auto src = this->val_ref(inst->getOperand(0));
  auto [res_vr, res_ref] = this->result_ref_single(inst);
  auto res_scratch = ScratchReg{derived()};
  switch (val_info.type) {
    using enum LLVMBasicValType;
  case f32: derived()->encode_fnegf32(src.part(0), res_scratch); break;
  case f64: derived()->encode_fnegf64(src.part(0), res_scratch); break;
  case f128: derived()->encode_fnegf128(src.part(0), res_scratch); break;
  case v64:
    assert(scalar_ty->isFloatTy());
    derived()->encode_fnegv2f32(src.part(0), res_scratch);
    break;
  case v128:
    if (scalar_ty->isFloatTy()) {
      derived()->encode_fnegv4f32(src.part(0), res_scratch);
    } else {
      assert(scalar_ty->isDoubleTy());
      derived()->encode_fnegv2f64(src.part(0), res_scratch);
    }
    break;
  default: return false;
  }

  this->set_value(res_ref, res_scratch);
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_float_ext_trunc(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  auto *src_val = inst->getOperand(0);
  auto *src_ty = src_val->getType();
  auto *dst_ty = inst->getType();

  auto res_vr = this->result_ref(inst);

  ScratchReg res_scratch{derived()};
  SymRef sym;
  if (src_ty->isDoubleTy() && dst_ty->isFloatTy()) {
    auto src_ref = this->val_ref(src_val);
    derived()->encode_f64tof32(src_ref.part(0), res_scratch);
  } else if (src_ty->isFP128Ty() && dst_ty->isFloatTy()) {
    sym = get_libfunc_sym(LibFunc::trunctfsf2);
  } else if (src_ty->isFP128Ty() && dst_ty->isDoubleTy()) {
    sym = get_libfunc_sym(LibFunc::trunctfdf2);
  } else if (src_ty->isFloatTy() && dst_ty->isDoubleTy()) {
    auto src_ref = this->val_ref(src_val);
    derived()->encode_f32tof64(src_ref.part(0), res_scratch);
  } else if (src_ty->isFloatTy() && dst_ty->isFP128Ty()) {
    sym = get_libfunc_sym(LibFunc::extendsftf2);
  } else if (src_ty->isDoubleTy() && dst_ty->isFP128Ty()) {
    sym = get_libfunc_sym(LibFunc::extenddftf2);
  }

  if (res_scratch.has_reg()) {
    this->set_value(res_vr.part(0), res_scratch);
  } else if (sym.valid()) {
    IRValueRef src_ref = src_val;
    derived()->create_helper_call({&src_ref, 1}, &res_vr, sym);
  } else {
    return false;
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

  if (src_ty->isFP128Ty()) {
    if (saturate) {
      return false;
    }

    LibFunc lf = sign ? LibFunc::fixtfdi : LibFunc::fixunstfdi;
    SymRef sym = get_libfunc_sym(lf);

    auto res_vr = this->result_ref(inst);
    derived()->create_helper_call({&src_val, 1}, &res_vr, sym);
    return true;
  }

  if (!src_ty->isFloatTy() && !src_ty->isDoubleTy()) {
    return false;
  }

  const auto src_double = src_ty->isDoubleTy();

  using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ScratchReg &);
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
  EncodeFnTy fn = fns[src_double][bit_width > 32][sign][saturate];

  if (saturate && bit_width % 32 != 0) {
    // TODO: clamp result to smaller integer bounds
    return false;
  }

  auto src_ref = this->val_ref(src_val);
  auto [res_vr, res_ref] = this->result_ref_single(inst);
  auto res_scratch = ScratchReg{derived()};
  if (!(derived()->*fn)(src_ref.part(0), res_scratch)) {
    return false;
  }

  this->set_value(res_ref, res_scratch);
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_int_to_float(
    const llvm::Instruction *inst, const ValInfo &, u64 sign) noexcept {
  const llvm::Value *src_val = inst->getOperand(0);
  auto *dst_ty = inst->getType();
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

  if (!dst_ty->isFloatTy() && !dst_ty->isDoubleTy()) {
    return false;
  }

  const auto dst_double = dst_ty->isDoubleTy();

  ValueRef src_ref = this->val_ref(src_val);
  ValuePartRef src_op = src_ref.part(0);
  auto [res_vr, res_ref] = this->result_ref_single(inst);
  auto res_scratch = ScratchReg{derived()};

  if (bit_width != 32 && bit_width != 64) {
    unsigned ext = tpde::util::align_up(bit_width, 32);
    src_op = std::move(src_op).into_extended(sign, bit_width, ext);
  }

  if (sign) {
    if (bit_width > 32) {
      if (dst_double) {
        derived()->encode_i64tof64(std::move(src_op), res_scratch);
      } else {
        derived()->encode_i64tof32(std::move(src_op), res_scratch);
      }
    } else {
      if (dst_double) {
        derived()->encode_i32tof64(std::move(src_op), res_scratch);
      } else {
        derived()->encode_i32tof32(std::move(src_op), res_scratch);
      }
    }
  } else {
    if (bit_width > 32) {
      if (dst_double) {
        derived()->encode_u64tof64(std::move(src_op), res_scratch);
      } else {
        derived()->encode_u64tof32(std::move(src_op), res_scratch);
      }
    } else {
      if (dst_double) {
        derived()->encode_u32tof64(std::move(src_op), res_scratch);
      } else {
        derived()->encode_u32tof32(std::move(src_op), res_scratch);
      }
    }
  }

  this->set_value(res_ref, res_scratch);
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_int_trunc(
    const llvm::Instruction *inst, const ValInfo &val_info, u64) noexcept {
  ValueRef res_vr = this->result_ref(inst);
  ValueRef src_vr = this->val_ref(inst->getOperand(0));

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
  case v64: {
    auto dst_width = inst->getType()->getScalarType()->getIntegerBitWidth();
    auto src_width =
        inst->getOperand(0)->getType()->getScalarType()->getIntegerBitWidth();
    // With the currently legal vector types, we only support halving vectors.
    if (dst_width * 2 != src_width) {
      return false;
    }

    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ScratchReg &);
    EncodeFnTy encode_fn = nullptr;
    switch (dst_width) {
    case 8: encode_fn = &Derived::encode_trunc_v8i16_8; break;
    case 16: encode_fn = &Derived::encode_trunc_v4i32_16; break;
    case 32: encode_fn = &Derived::encode_trunc_v2i64_32; break;
    default: return false;
    }
    ScratchReg res{derived()};
    if (!(derived()->*encode_fn)(src_vr.part(0), res)) {
      return false;
    }
    ValuePartRef res_ref = res_vr.part(0);
    this->set_value(res_ref, res);
    return true;
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
      ScratchReg scratch_high{derived()};
      if (!low.has_reg()) {
        low.load_to_reg();
      }
      derived()->encode_fill_with_sign64(low.get_unowned_ref(), scratch_high);
      this->set_value(res_ref_high, scratch_high);
    } else {
      res_ref_high.set_value(ValuePart{u64{0}, 8, res_ref_high.bank()});
    }
  }

  res.part(0).set_value(std::move(low));
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_ptr_to_int(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  if (!inst->getType()->isIntegerTy()) {
    return false;
  }

  // this is a no-op since every operation that depends on it will
  // zero/sign-extend the value anyways
  auto [res_vr, res_ref] = this->result_ref_single(inst);
  res_ref.set_value(this->val_ref(inst->getOperand(0)).part(0));
  return true;
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
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  // at most this should be fine to implement as a copy operation
  // as the values cannot be aggregates
  // TODO: this is not necessarily a no-op for vectors
  const auto src = inst->getOperand(0);

  const auto src_parts = this->adaptor->val_parts(src);
  const auto dst_parts = this->adaptor->val_parts(inst);
  // TODO(ts): support 128bit values
  if (src_parts.count() != 1 || dst_parts.count() != 1) {
    return false;
  }

  auto [_, src_ref] = this->val_ref_single(src);

  ValueRef res_ref = this->result_ref(inst);
  if (src_parts.reg_bank(0) == dst_parts.reg_bank(0)) {
    res_ref.part(0).set_value(std::move(src_ref));
  } else {
    ValuePartRef res_vpr = res_ref.part(0);
    AsmReg src_reg = src_ref.load_to_reg();
    derived()->mov(res_vpr.alloc_reg(), src_reg, res_vpr.part_size());
    res_vpr.set_modified();
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
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *insert = llvm::cast<llvm::InsertValueInst>(inst);
  auto agg = insert->getAggregateOperand();
  auto ins = insert->getInsertedValueOperand();

  unsigned part_count = this->adaptor->val_part_count(agg);
  auto [first_part, last_part] =
      this->adaptor->complex_part_for_index(insert, insert->getIndices());

  ValueRef agg_ref = this->val_ref(agg);
  ValueRef ins_ref = this->val_ref(ins);
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
    IRValueRef vec,
    unsigned idx,
    LLVMBasicValType ty,
    ScratchReg &out_reg) noexcept {
  assert(this->adaptor->val_part_count(vec) == 1);

  auto vec_vr = this->val_ref(vec);
  vec_vr.disown();
  auto vec_ref = vec_vr.part(0);
  this->spill(vec_ref.assignment());

  GenericValuePart addr = derived()->val_spill_slot(vec_ref);
  auto &expr = std::get<typename GenericValuePart::Expr>(addr.state);
  expr.disp += idx * this->adaptor->basic_ty_part_size(ty);

  switch (ty) {
    using enum LLVMBasicValType;
  case i8: derived()->encode_loadi8(std::move(addr), out_reg); break;
  case i16: derived()->encode_loadi16(std::move(addr), out_reg); break;
  case i32: derived()->encode_loadi32(std::move(addr), out_reg); break;
  case i64:
  case ptr: derived()->encode_loadi64(std::move(addr), out_reg); break;
  case f32: derived()->encode_loadf32(std::move(addr), out_reg); break;
  case f64: derived()->encode_loadf64(std::move(addr), out_reg); break;
  default: TPDE_UNREACHABLE("unexpected vector element type");
  }
}

template <typename Adaptor, typename Derived, typename Config>
void LLVMCompilerBase<Adaptor, Derived, Config>::insert_element(
    ValuePart &vec_ref,
    unsigned idx,
    LLVMBasicValType ty,
    GenericValuePart el) noexcept {
  assert(vec_ref.has_assignment());
  vec_ref.unlock(this);
  if (vec_ref.assignment().register_valid()) {
    this->evict(vec_ref.assignment());
  }

  GenericValuePart addr = derived()->val_spill_slot(vec_ref);
  auto &expr = std::get<typename GenericValuePart::Expr>(addr.state);
  expr.disp += idx * this->adaptor->basic_ty_part_size(ty);

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

  auto *vec_ty = llvm::cast<llvm::FixedVectorType>(src->getType());
  unsigned nelem = vec_ty->getNumElements();
  assert((nelem & (nelem - 1)) == 0 && "vector nelem must be power of two");
  assert(index->getType()->getIntegerBitWidth() >= 8);

  auto [res_vr, result] = this->result_ref_single(inst);
  LLVMBasicValType bvt = val_info.type;

  ScratchReg scratch_res{this};
  if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
    unsigned cidx = ci->getZExtValue();
    derived()->extract_element(src, cidx, bvt, scratch_res);

    (void)this->val_ref(src).part(0); // ref-counting
    this->set_value(result, scratch_res);
    return true;
  }

  // TODO: deduplicate with code above somehow?
  // First, copy value into the spill slot.
  auto [_, vec_ref] = this->val_ref_single(src);
  this->spill(vec_ref.assignment());
  vec_ref.unlock();

  // Second, create address. Mask index, out-of-bounds access are just poison.
  ScratchReg idx_scratch{this};
  GenericValuePart addr = derived()->val_spill_slot(vec_ref);
  auto &expr = std::get<typename GenericValuePart::Expr>(addr.state);
  u64 mask = nelem - 1;
  derived()->encode_landi64(this->val_ref(index).part(0),
                            ValuePartRef{this, mask, 8, Config::GP_BANK},
                            idx_scratch);
  assert(expr.scale == 0);
  expr.scale = this->adaptor->basic_ty_part_size(bvt);
  expr.index = std::move(idx_scratch);

  // Third, do the load.
  switch (bvt) {
    using enum LLVMBasicValType;
  case i8: derived()->encode_loadi8(std::move(addr), scratch_res); break;
  case i16: derived()->encode_loadi16(std::move(addr), scratch_res); break;
  case i32: derived()->encode_loadi32(std::move(addr), scratch_res); break;
  case i64:
  case ptr: derived()->encode_loadi64(std::move(addr), scratch_res); break;
  case f32: derived()->encode_loadf32(std::move(addr), scratch_res); break;
  case f64: derived()->encode_loadf64(std::move(addr), scratch_res); break;
  default: TPDE_UNREACHABLE("unexpected vector element type");
  }

  this->set_value(result, scratch_res);
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_insert_element(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  llvm::Value *index = inst->getOperand(2);

  auto *vec_ty = llvm::cast<llvm::FixedVectorType>(inst->getType());
  unsigned nelem = vec_ty->getNumElements();
  assert((nelem & (nelem - 1)) == 0 && "vector nelem must be power of two");
  assert(index->getType()->getIntegerBitWidth() >= 8);

  auto ins = inst->getOperand(1);
  auto [val_ref, val] = this->val_ref_single(ins);

  auto [res_vr, result] = this->result_ref_single(inst);
  auto [bvt, _] = this->adaptor->lower_type(ins->getType());
  assert(bvt != LLVMBasicValType::complex);

  // We do the dynamic insert in the spill slot of result.
  // TODO: reuse spill slot of vec_ref if possible.

  // First, copy value into the result. We must also do this for constant
  // indices, because the value reference must always be initialized.
  result.set_value(this->val_ref(inst->getOperand(0)).part(0));

  if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
    unsigned cidx = ci->getZExtValue();
    derived()->insert_element(result, cidx, bvt, std::move(val));
    // No need for ref counting: all operands and results were ValuePartRefs.
    return true;
  }

  result.unlock();
  // Evict, because we will overwrite the value in the stack slot.
  this->evict(result.assignment());

  // Second, create address. Mask index, out-of-bounds access are just poison.
  ScratchReg idx_scratch{this};
  GenericValuePart addr = derived()->val_spill_slot(result);
  auto &expr = std::get<typename GenericValuePart::Expr>(addr.state);
  u64 mask = nelem - 1;
  derived()->encode_landi64(this->val_ref(index).part(0),
                            ValuePartRef{this, mask, 8, Config::GP_BANK},
                            idx_scratch);
  assert(expr.scale == 0);
  expr.scale = val.part_size();
  expr.index = std::move(idx_scratch);

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
  assert((dst_nelem & (dst_nelem - 1)) == 0 && "invalid dst vector size");
  assert((src_nelem & (src_nelem - 1)) == 0 && "invalid src vector size");

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
  } else if (elem_ty->isIntegerTy(64)) {
    bvt = LLVMBasicValType::i64;
  } else {
    TPDE_UNREACHABLE("invalid element type for shufflevector");
  }

  auto [res_vr, res_ref] = this->result_ref_single(inst);
  // Make sure the results has an allocated register. insert_element will use
  // load_to_reg to lock the value into a register, but if we don't allocate a
  // register here, that will fail because the value is uninitialized.
  res_ref.alloc_reg();
  // But, as the value is uninitialized, the stack slot is also valid. This
  // avoids spilling an uninitialized register in insert_element.
  this->allocate_spill_slot(res_ref.assignment());
  res_ref.assignment().set_stack_valid();

  ScratchReg tmp{this};
  llvm::ArrayRef<int> mask = shuffle->getShuffleMask();
  for (unsigned i = 0; i < dst_nelem; i++) {
    if (mask[i] == llvm::PoisonMaskElem) {
      continue;
    }
    IRValueRef src = unsigned(mask[i]) < src_nelem ? lhs : rhs;
    if (auto *cst = llvm::dyn_cast<llvm::Constant>(src)) {
      auto *cst_elem = cst->getAggregateElement(i);
      u64 const_elem;
      if (llvm::isa<llvm::PoisonValue>(cst_elem)) {
        continue;
      } else if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(cst_elem)) {
        const_elem = ci->getZExtValue();
      } else if (auto *cfp = llvm::dyn_cast<llvm::ConstantFP>(cst_elem)) {
        const_elem = cfp->getValue().bitcastToAPInt().getZExtValue();
      } else {
        TPDE_UNREACHABLE("invalid constant element type");
      }
      auto bank = this->adaptor->basic_ty_part_bank(bvt);
      auto size = this->adaptor->basic_ty_part_size(bvt);
      ValuePartRef const_ref{this, &const_elem, size, bank};
      const_ref.reload_into_specific_fixed(tmp.alloc(bank));
    } else {
      derived()->extract_element(src, mask[i] & (src_nelem - 1), bvt, tmp);
    }
    derived()->insert_element(res_ref, i, bvt, std::move(tmp));
  }
  (void)this->val_ref(lhs).part(0); // ref-counting
  (void)this->val_ref(rhs).part(0); // ref-counting
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_cmpxchg(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *cmpxchg = llvm::cast<llvm::AtomicCmpXchgInst>(inst);
  auto *cmp_val = cmpxchg->getCompareOperand();
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
                                       ScratchReg &,
                                       ScratchReg &);
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

  auto *ptr_val = cmpxchg->getPointerOperand();
  assert(ptr_val->getType()->isPointerTy());
  auto ptr_ref = this->val_ref(ptr_val);

  auto cmp_ref = this->val_ref(cmp_val);
  auto new_ref = this->val_ref(new_val);

  auto res = this->result_ref(cmpxchg);
  auto res_ref = res.part(0);
  auto res_ref_high = res.part(1);

  ScratchReg orig_scratch{derived()};
  ScratchReg succ_scratch{derived()};

  llvm::AtomicOrdering order = cmpxchg->getMergedOrdering();
  EncodeFnTy encode_fn = fns[width_idx][size_t(order)];
  assert(encode_fn && "invalid cmpxchg ordering");
  if (!(derived()->*encode_fn)(ptr_ref.part(0),
                               cmp_ref.part(0),
                               new_ref.part(0),
                               orig_scratch,
                               succ_scratch)) {
    return false;
  }

  this->set_value(res_ref, orig_scratch);
  this->set_value(res_ref_high, succ_scratch);

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
  bool (Derived::*fn)(GenericValuePart &&, GenericValuePart &&, ScratchReg &) =
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
  auto [res_vr, res_ref] = this->result_ref_single(rmw);
  ScratchReg res{this};
  if (!(derived()->*fn)(ptr_ref.part(0), val_ref.part(0), res)) {
    return false;
  }
  this->set_value(res_ref, res);
  return true;
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
    auto flag = CallArg::Flag::none;
    u32 byval_align = 0, byval_size = 0;

    if (call->paramHasAttr(i, llvm::Attribute::AttrKind::ZExt)) {
      flag = CallArg::Flag::zext;
    } else if (call->paramHasAttr(i, llvm::Attribute::AttrKind::SExt)) {
      flag = CallArg::Flag::sext;
    } else if (call->paramHasAttr(i, llvm::Attribute::AttrKind::ByVal)) {
      flag = CallArg::Flag::byval;
      auto &data_layout = this->adaptor->mod->getDataLayout();
      llvm::Type *byval_ty = call->getParamByValType(i);
      byval_size = data_layout.getTypeAllocSize(byval_ty);

      if (auto param_align = call->getParamStackAlign(i)) {
        byval_align = param_align->value();
      } else if (auto param_align = call->getParamAlign(i)) {
        byval_align = param_align->value();
      } else {
        byval_align = data_layout.getABITypeAlign(byval_ty).value();
      }
    } else if (call->paramHasAttr(i, llvm::Attribute::AttrKind::StructRet)) {
      flag = CallArg::Flag::sret;
    }
    assert(!call->paramHasAttr(i, llvm::Attribute::AttrKind::InAlloca));
    assert(!call->paramHasAttr(i, llvm::Attribute::AttrKind::Preallocated));

    cb->add_arg(CallArg{op, flag, byval_align, byval_size});
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
    cb->add_ret(res);
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

  ScratchReg res_scratch{derived()};
  auto res = this->result_ref(inst);

  switch (val_info.type) {
    using enum LLVMBasicValType;
  case i1:
  case i8:
  case i16:
  case i32:
    derived()->encode_select_i32(
        std::move(cond), lhs.part(0), rhs.part(0), res_scratch);
    break;
  case i64:
  case ptr:
    derived()->encode_select_i64(
        std::move(cond), lhs.part(0), rhs.part(0), res_scratch);
    break;
  case f32:
  case v32:
    derived()->encode_select_f32(
        std::move(cond), lhs.part(0), rhs.part(0), res_scratch);
    break;
  case f64:
  case v64:
    derived()->encode_select_f64(
        std::move(cond), lhs.part(0), rhs.part(0), res_scratch);
    break;
  case f128:
  case v128:
    derived()->encode_select_v2u64(
        std::move(cond), lhs.part(0), rhs.part(0), res_scratch);
    break;
  case complex: {
    // Handle case of complex with two i64 as i128, this is extremely hacky...
    // TODO(ts): support full complex types using branches
    const auto parts = this->adaptor->val_parts(inst);
    if (parts.count() != 2 || parts.reg_bank(0) != Config::GP_BANK ||
        parts.reg_bank(1) != Config::GP_BANK) {
      return false;
    }
  }
    [[fallthrough]];
  case i128: {
    ScratchReg res_scratch_high{derived()};
    auto res_ref_high = res.part(1);

    derived()->encode_select_i128(std::move(cond),
                                  lhs.part(0),
                                  lhs.part(1),
                                  rhs.part(0),
                                  rhs.part(1),
                                  res_scratch,
                                  res_scratch_high);
    this->set_value(res_ref_high, res_scratch_high);
    break;
  }
  default: TPDE_UNREACHABLE("invalid select basic type"); break;
  }

  auto res_ref = res.part(0);
  this->set_value(res_ref, res_scratch);
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

  auto [ptr_ref, base] = this->val_ref_single(gep->getPointerOperand());
  if (base.has_assignment() && base.assignment().is_stack_variable()) {
    addr = derived()->create_addr_for_alloca(base.assignment());
    assert(addr.is_expr());
    displacement = expr.disp;
    expr.disp = 0;
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
            off_disp = data_layout.getTypeAllocSize(cur_ty) * idx_val;
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

  expr.disp = displacement;

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
    ScratchReg res_scratch = std::move(*op_reg);
    this->set_value(res_ref, res_scratch);
  } else {
    ScratchReg res_scratch{derived()};
    derived()->mov(res_scratch.alloc_gp(), res_reg, 8);
    this->set_value(res_ref, res_scratch);
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
    default: assert(0 && "unexpected fcmp predicate");
    }

    IRValueRef lhs = cmp->getOperand(0);
    IRValueRef rhs = cmp->getOperand(1);
    std::array<IRValueRef, 2> args{lhs, rhs};

    auto res_vr = this->result_ref(cmp);
    derived()->create_helper_call(args, &res_vr, sym);
    derived()->compile_i32_cmp_zero(res_vr.part(0).load_to_reg(), cmp_pred);

    return true;
  }

  if (!cmp_ty->isFloatTy() && !cmp_ty->isDoubleTy()) {
    return false;
  }

  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ScratchReg &);
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

  ValueRef lhs = this->val_ref(cmp->getOperand(0));
  ValueRef rhs = this->val_ref(cmp->getOperand(1));
  ScratchReg res_scratch{derived()};
  auto [res_vr, res_ref] = this->result_ref_single(cmp);

  if (!(derived()->*fn)(lhs.part(0), rhs.part(0), res_scratch)) {
    return false;
  }

  this->set_value(res_ref, res_scratch);
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_switch(
    const llvm::Instruction *inst, const ValInfo &, u64) noexcept {
  const auto *switch_inst = llvm::cast<llvm::SwitchInst>(inst);
  ValuePartRef cmp_ref{this};
  AsmReg cmp_reg;
  bool width_is_32 = false;
  {
    llvm::Value *cond = switch_inst->getCondition();
    u32 width = cond->getType()->getIntegerBitWidth();
    if (width > 64) {
      return false;
    }

    auto [_, arg_ref] = this->val_ref_single(cond);

    width_is_32 = width <= 32;
    if (u32 dst_width = tpde::util::align_up(width, 32); width != dst_width) {
      cmp_ref = std::move(arg_ref).into_extended(false, width, dst_width);
    } else if (arg_ref.has_assignment()) {
      cmp_ref = std::move(arg_ref).into_temporary();
    } else {
      cmp_ref = std::move(arg_ref);
    }
    cmp_reg = cmp_ref.has_reg() ? cmp_ref.cur_reg() : cmp_ref.load_to_reg();
  }

  // We must not evict any registers in the branching code, as we don't track
  // the individual value states per block. Hence, we must not allocate any
  // registers (e.g., for constants, jump table address) below.
  ScratchReg scratch2{this};
  AsmReg tmp_reg = scratch2.alloc_gp();

  const auto spilled = this->spill_before_branch();
  this->begin_branch_region();

  // get all cases, their target block and sort them in ascending order
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

  // because some blocks might have PHI-values we need to first jump to a
  // label which then fixes the registers and then jumps to the block
  // TODO(ts): check which blocks need PHIs and otherwise jump directly to
  // them? probably better for branch predictor

  tpde::util::SmallVector<typename Assembler::Label, 64> case_labels;
  for (auto i = 0u; i < cases.size(); ++i) {
    case_labels.push_back(this->assembler.label_create());
  }

  const auto default_label = this->assembler.label_create();

  const auto build_range = [&,
                            this](size_t begin, size_t end, const auto &self) {
    assert(begin <= end);
    const auto num_cases = end - begin;
    if (num_cases <= 4) {
      // if there are four or less cases we just compare the values
      // against each of them
      for (auto i = 0u; i < num_cases; ++i) {
        derived()->switch_emit_cmpeq(case_labels[begin + i],
                                     cmp_reg,
                                     tmp_reg,
                                     cases[begin + i].first,
                                     width_is_32);
      }

      derived()->generate_raw_jump(Derived::Jump::jmp, default_label);
      return;
    }

    // check if the density of the values is high enough to warrant building
    // a jump table
    auto range = cases[end - 1].first - cases[begin].first;
    // we will get wrong results if range is -1 so skip the jump table if
    // that is the case
    if (range != 0xFFFF'FFFF'FFFF'FFFF && (range / num_cases) < 8) {
      // for gcc, it seems that if there are less than 8 values per
      // case it will build a jump table so we do that, too

      // the actual range is one greater than the result we get from
      // subtracting so adjust for that
      range += 1;

      tpde::util::SmallVector<typename Assembler::Label, 32> label_vec;
      std::span<typename Assembler::Label> labels;
      if (range == num_cases) {
        labels = std::span{case_labels.begin() + begin, num_cases};
      } else {
        label_vec.resize(range, default_label);
        for (auto i = 0u; i < num_cases; ++i) {
          label_vec[cases[begin + i].first - cases[begin].first] =
              case_labels[begin + i];
        }
        labels = std::span{label_vec.begin(), range};
      }

      // Give target the option to emit a jump table.
      if (derived()->switch_emit_jump_table(default_label,
                                            labels,
                                            cmp_reg,
                                            tmp_reg,
                                            cases[begin].first,
                                            cases[end - 1].first,
                                            width_is_32)) {
        return;
      }
    }

    // do a binary search step
    const auto half_len = num_cases / 2;
    const auto half_value = cases[begin + half_len].first;
    const auto gt_label = this->assembler.label_create();

    // this will cmp against the input value, jump to the case if it is
    // equal or to gt_label if the value is greater. Otherwise it will
    // fall-through
    derived()->switch_emit_binary_step(case_labels[begin + half_len],
                                       gt_label,
                                       cmp_reg,
                                       tmp_reg,
                                       half_value,
                                       width_is_32);
    // search the lower half
    self(begin, begin + half_len, self);

    // and the upper half
    this->label_place(gt_label);
    self(begin + half_len + 1, end, self);
  };

  build_range(0, case_labels.size(), build_range);

  // write out the labels
  this->label_place(default_label);
  // TODO(ts): factor into arch-code?
  derived()->generate_branch_to_block(
      Derived::Jump::jmp,
      this->adaptor->block_lookup_idx(switch_inst->getDefaultDest()),
      false,
      false);

  for (auto i = 0u; i < cases.size(); ++i) {
    this->label_place(case_labels[i]);
    derived()->generate_branch_to_block(
        Derived::Jump::jmp, cases[i].second, false, false);
  }

  this->end_branch_region();
  this->release_spilled_regs(spilled);

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

    unwind_label = this->assembler.label_create();
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

  this->assembler.except_add_call_site(off_before_call,
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
        assert(llvm::dyn_cast<llvm::GlobalValue>(C));
        sym = lookup_type_info_sym(C);
      }
      this->assembler.except_add_action(i == 0, sym);
    } else {
      assert(landing_pad->isFilter(i));
      auto *C = landing_pad->getClause(i);
      assert(C->getType()->isArrayTy());
      if (C->getType()->getArrayNumElements() == 0) {
        this->assembler.except_add_empty_spec_action(i == 0);
      } else {
        TPDE_LOG_ERR("Exception filters with non-zero length arrays "
                     "not supported");
        return false;
      }
    }
  }

  if (is_cleanup) {
    assert(num_clauses != 0);
    this->assembler.except_add_cleanup_action();
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
        IRValueRef value) noexcept {
  for (const auto &[val, sym] : type_info_syms) {
    if (val == value) {
      return sym;
    }
  }

  const auto sym = global_sym(llvm::cast<llvm::GlobalValue>(value));

  u32 off;
  u8 tmp[8] = {};
  auto rodata = this->assembler.get_data_section(true, true);
  const auto addr_sym = this->assembler.sym_def_data(
      rodata, {}, {tmp, sizeof(tmp)}, 8, Assembler::SymBinding::LOCAL, &off);
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
    ScratchReg res{derived()};
    derived()->encode_loadreli64(ptr.part(0), off.part(0), res);
    this->set_value(res_ref, res);
    return true;
  }
  case llvm::Intrinsic::threadlocal_address: {
    auto gv = llvm::cast<llvm::GlobalValue>(inst->getOperand(0));
    auto [res_vr, res_ref] = this->result_ref_single(inst);
    // TODO: optimize for different TLS access models
    ScratchReg res =
        derived()->tls_get_addr(global_sym(gv), tpde::TLSModel::GlobalDynamic);
    this->set_value(res_ref, res);
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
        GenericValuePart &&, GenericValuePart &&, ScratchReg &);
    EncodeFnTy fn;
    switch (intrin_id) {
      using enum llvm::Intrinsic::IndependentIntrinsics;
    case minnum:
      fn = is_double ? &Derived::encode_minnumf64 : &Derived::encode_minnumf32;
      break;
    case maxnum:
      fn = is_double ? &Derived::encode_maxnumf64 : &Derived::encode_maxnumf32;
      break;
    case copysign:
      fn = is_double ? &Derived::encode_copysignf64
                     : &Derived::encode_copysignf32;
      break;
    default: TPDE_UNREACHABLE("invalid binary fp intrinsic");
    }

    auto lhs = this->val_ref(inst->getOperand(0));
    auto rhs = this->val_ref(inst->getOperand(1));
    auto [res_vr, res_ref] = this->result_ref_single(inst);
    ScratchReg res{derived()};
    if (!(derived()->*fn)(lhs.part(0), rhs.part(0), res)) {
      return false;
    }
    this->set_value(res_ref, res);
    return true;
  }
  case llvm::Intrinsic::fabs: {
    auto *val = inst->getOperand(0);
    auto *ty = val->getType();

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    ScratchReg res_scratch{derived()};

    if (ty->isDoubleTy()) {
      derived()->encode_fabsf64(this->val_ref(val).part(0), res_scratch);
    } else if (ty->isFloatTy()) {
      derived()->encode_fabsf32(this->val_ref(val).part(0), res_scratch);
    } else if (ty->isFP128Ty()) {
      derived()->encode_fabsf128(this->val_ref(val).part(0), res_scratch);
    } else {
      return false;
    }
    this->set_value(res_ref, res_scratch);
    return true;
  }
  case llvm::Intrinsic::sqrt: {
    auto *val = inst->getOperand(0);
    auto *ty = val->getType();
    if (!ty->isFloatTy() && !ty->isDoubleTy()) {
      return false;
    }

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    ScratchReg res_scratch{derived()};
    if (ty->isDoubleTy()) {
      derived()->encode_sqrtf64(this->val_ref(val).part(0), res_scratch);
    } else {
      derived()->encode_sqrtf32(this->val_ref(val).part(0), res_scratch);
    }
    this->set_value(res_ref, res_scratch);
    return true;
  }
  case llvm::Intrinsic::fmuladd: {
    auto op1_ref = this->val_ref(inst->getOperand(0));
    auto op2_ref = this->val_ref(inst->getOperand(1));
    auto op3_ref = this->val_ref(inst->getOperand(2));

    if (inst->getType()->isFP128Ty()) {
      auto cb1 = derived()->create_call_builder();
      cb1->add_arg(op1_ref.part(0), tpde::CCAssignment{});
      cb1->add_arg(op2_ref.part(0), tpde::CCAssignment{});
      cb1->call(get_libfunc_sym(LibFunc::multf3));
      ValuePartRef tmp{this, Config::FP_BANK};
      cb1->add_ret(tmp, tpde::CCAssignment{});

      auto cb2 = derived()->create_call_builder();
      cb2->add_arg(std::move(tmp), tpde::CCAssignment{});
      cb2->add_arg(op3_ref.part(0), tpde::CCAssignment{});
      cb2->call(get_libfunc_sym(LibFunc::addtf3));
      auto res_vr2 = this->result_ref(inst);
      cb2->add_ret(res_vr2);
      return true;
    }

    if (!inst->getType()->isFloatTy() && !inst->getType()->isDoubleTy()) {
      return false;
    }

    const auto is_double = inst->getOperand(0)->getType()->isDoubleTy();

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    ScratchReg res_scratch{derived()};
    if (is_double) {
      derived()->encode_fmaf64(
          op1_ref.part(0), op2_ref.part(0), op3_ref.part(0), res_scratch);
    } else {
      derived()->encode_fmaf32(
          op1_ref.part(0), op2_ref.part(0), op3_ref.part(0), res_scratch);
    }
    this->set_value(res_ref, res_scratch);
    return true;
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
      ScratchReg lo{derived()}, hi{derived()};
      derived()->encode_absi128(val_ref.part(0), val_ref.part(1), lo, hi);
      this->set_value(res_ref.part(0), lo);
      this->set_value(res_ref.part(1), hi);
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

    ScratchReg res{derived()};
    auto [res_vr, res_ref] = this->result_ref_single(inst);
    if (width <= 32) {
      derived()->encode_absi32(std::move(op), res);
    } else {
      derived()->encode_absi64(std::move(op), res);
    }
    this->set_value(res_ref, res);
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

    ScratchReg res{derived()};
    auto [res_vr, res_ref] = this->result_ref_single(inst);
    using EncodeFnTy = bool (Derived::*)(
        GenericValuePart &&, GenericValuePart &&, ScratchReg &);
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
    if (!(derived()->*encode_fn)(std::move(lhs), std::move(rhs), res)) {
      return false;
    }
    this->set_value(res_ref, res);
    return true;
  }
  case llvm::Intrinsic::ptrmask: {
    // ptrmask is just an integer and.
    ValueRef lhs = this->val_ref(inst->getOperand(0));
    ValueRef rhs = this->val_ref(inst->getOperand(1));
    auto [res_vr, res] = this->result_ref_single(inst);
    auto res_scratch = ScratchReg{derived()};
    derived()->encode_landi64(lhs.part(0), rhs.part(0), res_scratch);
    this->set_value(res, res_scratch);
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

    ScratchReg res{derived()};

    // TODO: generate better code for constant amounts.
    bool shift_left = intrin_id == llvm::Intrinsic::fshl;
    if (inst->getOperand(0) == inst->getOperand(1)) {
      // Better code for rotate.
      using EncodeFnTy = bool (Derived::*)(
          GenericValuePart &&, GenericValuePart &&, ScratchReg &);
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
      if (!(derived()->*fn)(lhs.part(0), amt.part(0), res)) {
        return false;
      }
    } else {
      using EncodeFnTy = bool (Derived::*)(GenericValuePart &&,
                                           GenericValuePart &&,
                                           GenericValuePart &&,
                                           ScratchReg &);
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
      if (!(derived()->*fn)(lhs.part(0), rhs.part(0), amt.part(0), res)) {
        return false;
      }
    }

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    this->set_value(res_ref, res);
    return true;
  }
  case llvm::Intrinsic::bswap: {
    auto *val = inst->getOperand(0);
    if (!val->getType()->isIntegerTy()) {
      return false;
    }
    const auto width = val->getType()->getIntegerBitWidth();
    if (width % 16 || width > 64) {
      return false;
    }

    using EncodeFnTy = bool (Derived::*)(GenericValuePart &&, ScratchReg &);
    static constexpr std::array<EncodeFnTy, 4> encode_fns = {
        &Derived::encode_bswapi16,
        &Derived::encode_bswapi32,
        &Derived::encode_bswapi48,
        &Derived::encode_bswapi64,
    };
    EncodeFnTy encode_fn = encode_fns[width / 16 - 1];

    ScratchReg res{derived()};
    if (!(derived()->*encode_fn)(this->val_ref(val).part(0), res)) {
      return false;
    }

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    this->set_value(res_ref, res);
    return true;
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

    ScratchReg res{derived()};
    if (width <= 32) {
      derived()->encode_ctpopi32(std::move(op), res);
    } else {
      derived()->encode_ctpopi64(std::move(op), res);
    }

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    this->set_value(res_ref, res);
    return true;
  }
  case llvm::Intrinsic::ctlz:
  case llvm::Intrinsic::cttz: {
    auto *val = inst->getOperand(0);
    assert(val->getType()->isIntegerTy());
    const auto width = val->getType()->getIntegerBitWidth();
    if (width != 8 && width != 16 && width != 32 && width != 64) {
      return false;
    }

    const auto zero_is_poison =
        !llvm::cast<llvm::ConstantInt>(inst->getOperand(1))->isZero();

    auto val_ref = this->val_ref(val);
    auto [res_vr, res_ref] = this->result_ref_single(inst);
    ScratchReg res{derived()};

    if (intrin_id == llvm::Intrinsic::ctlz) {
      switch (width) {
      case 8:
        if (zero_is_poison) {
          derived()->encode_ctlzi8_zero_poison(val_ref.part(0), res);
        } else {
          derived()->encode_ctlzi8(val_ref.part(0), res);
        }
        break;
      case 16:
        if (zero_is_poison) {
          derived()->encode_ctlzi16_zero_poison(val_ref.part(0), res);
        } else {
          derived()->encode_ctlzi16(val_ref.part(0), res);
        }
        break;
      case 32:
        if (zero_is_poison) {
          derived()->encode_ctlzi32_zero_poison(val_ref.part(0), res);
        } else {
          derived()->encode_ctlzi32(val_ref.part(0), res);
        }
        break;
      case 64:
        if (zero_is_poison) {
          derived()->encode_ctlzi64_zero_poison(val_ref.part(0), res);
        } else {
          derived()->encode_ctlzi64(val_ref.part(0), res);
        }
        break;
      default: TPDE_UNREACHABLE("invalid size");
      }
    } else {
      assert(intrin_id == llvm::Intrinsic::cttz);
      switch (width) {
      case 8:
        if (zero_is_poison) {
          derived()->encode_cttzi32_zero_poison(val_ref.part(0), res);
        } else {
          derived()->encode_cttzi8(val_ref.part(0), res);
        }
        break;
      case 16:
        if (zero_is_poison) {
          derived()->encode_cttzi32_zero_poison(val_ref.part(0), res);
        } else {
          derived()->encode_cttzi16(val_ref.part(0), res);
        }
        break;
      case 32:
        if (zero_is_poison) {
          derived()->encode_cttzi32_zero_poison(val_ref.part(0), res);
        } else {
          derived()->encode_cttzi32(val_ref.part(0), res);
        }
        break;
      case 64:
        if (zero_is_poison) {
          derived()->encode_cttzi64_zero_poison(val_ref.part(0), res);
        } else {
          derived()->encode_cttzi64(val_ref.part(0), res);
        }
        break;
      default: TPDE_UNREACHABLE("invalid size");
      }
    }

    this->set_value(res_ref, res);
    return true;
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
      ScratchReg shifted{this};
      if (width < 32) {
        derived()->encode_shli32(std::move(op), std::move(amt_val), shifted);
      } else {
        derived()->encode_shli64(std::move(op), std::move(amt_val), shifted);
      }
      op = std::move(shifted);
    }

    ScratchReg res{derived()};
    if (width <= 32) {
      derived()->encode_bitreversei32(std::move(op), res);
    } else {
      derived()->encode_bitreversei64(std::move(op), res);
    }

    auto [res_vr, res_ref] = this->result_ref_single(inst);
    this->set_value(res_ref, res);
    return true;
  }
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
    auto *type = inst->getOperand(0);
    assert(llvm::isa<llvm::GlobalValue>(type));

    // not the most efficient but it's OK
    const auto type_info_sym = lookup_type_info_sym(type);
    const u64 idx = this->assembler.except_type_idx_for_sym(type_info_sym);

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

  ScratchReg res_scratch{derived()};
  auto [res_vr, res_ref] = this->result_ref_single(inst);
  auto [op_vr, op_ref] = this->val_ref_single(op);

  auto zero_ref = ValuePartRef{this, 0, 4, Config::GP_BANK};

  // handle common case
#define TEST(cond, name)                                                       \
  if (test == cond) {                                                          \
    if (is_double) {                                                           \
      derived()->encode_is_fpclass_##name##_double(                            \
          std::move(zero_ref), std::move(op_ref), res_scratch);                \
    } else {                                                                   \
      derived()->encode_is_fpclass_##name##_float(                             \
          std::move(zero_ref), std::move(op_ref), res_scratch);                \
    }                                                                          \
    this->set_value(res_ref, res_scratch);                                     \
    return true;                                                               \
  }

  TEST(IS_NAN, nan)
  TEST(IS_INF, inf)
  TEST(IS_NORM, norm)
#undef TEST

  // we OR' together the results from each test so initialize the result with
  // zero
  zero_ref.reload_into_specific_fixed(res_scratch.alloc_gp());

  op_ref.load_to_reg();

  using EncodeFnTy =
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ScratchReg &);
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
      // note that the std::move(res_scratch) here creates a new ScratchReg that
      // manages the register inside the GenericValuePart and res_scratch
      // becomes invalid by the time the encode function is entered
      (derived()->*encode_fns[i][is_double])(
          std::move(res_scratch), op_ref.get_unowned_ref(), res_scratch);
    }
  }

  this->set_value(res_ref, res_scratch);
  return true;
}

template <typename Adaptor, typename Derived, typename Config>
bool LLVMCompilerBase<Adaptor, Derived, Config>::compile_overflow_intrin(
    const llvm::IntrinsicInst *inst, OverflowOp op) noexcept {
  ValueRef lhs = this->val_ref(inst->getOperand(0));
  ValueRef rhs = this->val_ref(inst->getOperand(1));

  auto *ty = inst->getOperand(0)->getType();
  assert(ty->isIntegerTy());
  const auto width = ty->getIntegerBitWidth();

  if (width == 128) {
    auto lhs_ref = lhs.part(0);
    auto rhs_ref = rhs.part(0);
    GenericValuePart lhs_op_high = lhs.part(1);
    GenericValuePart rhs_op_high = rhs.part(1);
    ScratchReg res_val{derived()}, res_val_high{derived()}, res_of{derived()};

    if (!derived()->handle_overflow_intrin_128(op,
                                               std::move(lhs_ref),
                                               std::move(lhs_op_high),
                                               std::move(rhs_ref),
                                               std::move(rhs_op_high),
                                               res_val,
                                               res_val_high,
                                               res_of)) {
      return false;
    }

    auto res_ref = this->result_ref(inst);
    auto res_ref_val = res_ref.part(0);
    auto res_ref_high = res_ref.part(1);
    auto res_ref_of = res_ref.part(2);
    this->set_value(res_ref_val, res_val);
    this->set_value(res_ref_high, res_val_high);
    this->set_value(res_ref_of, res_of);
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
      GenericValuePart &&, GenericValuePart &&, ScratchReg &, ScratchReg &);
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

  ScratchReg res_val{derived()}, res_of{derived()};

  if (!(derived()->*encode_fn)(lhs.part(0), rhs.part(0), res_val, res_of)) {
    return false;
  }

  auto res_ref = this->result_ref(inst);
  auto res_ref_val = res_ref.part(0);
  auto res_ref_of = res_ref.part(1);
  this->set_value(res_ref_val, res_val);
  this->set_value(res_ref_of, res_of);
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
      bool (Derived::*)(GenericValuePart &&, GenericValuePart &&, ScratchReg &);
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
  ScratchReg res{derived()};
  if (!(derived()->*encode_fn)(lhs.part(0), rhs.part(0), res)) {
    return false;
  }

  auto [res_vr, res_ref_val] = this->result_ref_single(inst);
  this->set_value(res_ref_val, res);
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
