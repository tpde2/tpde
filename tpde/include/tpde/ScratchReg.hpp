// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

namespace tpde {

/// Owned unspillable and unevictable temporary register with RAII semantics.
template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
struct CompilerBase<Adaptor, Derived, Config>::RawScratchReg {
private:
  // TODO(ts): get this using the CompilerConfig?
  AsmReg reg = AsmReg::make_invalid();

public:
  /// Constructor, no register.
  explicit RawScratchReg() = default;

  explicit RawScratchReg(const RawScratchReg &) = delete;
  RawScratchReg(RawScratchReg &&other) { *this = std::move(other); }

  ~RawScratchReg() { assert(!reg.valid() && "must call reset() explicitly"); }

  RawScratchReg &operator=(const RawScratchReg &) = delete;
  RawScratchReg &operator=(RawScratchReg &&);

  /// Whether a register is currently allocated.
  bool has_reg() const { return reg.valid(); }

  /// The allocated register.
  AsmReg cur_reg() const {
    assert(has_reg());
    return reg;
  }

  /// Allocate a specific register.
  AsmReg alloc_specific(CompilerBase *compiler, AsmReg reg);

  /// Allocate register in the specified bank. Does nothing if it holds a
  /// register in the specified bank. Evicts a register if required.
  AsmReg alloc(CompilerBase *compiler, RegBank bank);

  /// Release register, marking it as unused; returns the register.
  AsmReg release(CompilerBase *compiler) {
    AsmReg res = reg;
    reset(compiler);
    return res;
  }

  /// Deallocate register.
  void reset(CompilerBase *compiler);

  /// @internal Forcefully change register without updating register file.
  /// Avoid.
  void force_set_reg(AsmReg reg) { this->reg = reg; }
};

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::RawScratchReg &
    CompilerBase<Adaptor, Derived, Config>::RawScratchReg::operator=(
        RawScratchReg &&other) {
  if (this == &other) {
    return *this;
  }

  assert(!reg.valid() && "must call reset() explicitly");
  this->reg = other.reg;
  other.reg = AsmReg::make_invalid();
  return *this;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::AsmReg
    CompilerBase<Adaptor, Derived, Config>::RawScratchReg::alloc_specific(
        CompilerBase *compiler, AsmReg reg) {
  assert(compiler->may_change_value_state());
  assert(!compiler->register_file.is_fixed(reg));
  reset(compiler);

  if (compiler->register_file.is_used(reg)) {
    compiler->evict_reg(reg);
  }

  compiler->register_file.mark_used(reg, INVALID_VAL_LOCAL_IDX, 0);
  compiler->register_file.mark_clobbered(reg);
  compiler->register_file.mark_fixed(reg);
  this->reg = reg;
  return reg;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
CompilerBase<Adaptor, Derived, Config>::AsmReg
    CompilerBase<Adaptor, Derived, Config>::RawScratchReg::alloc(
        CompilerBase *compiler, RegBank bank) {
  assert(compiler->may_change_value_state());

  auto &reg_file = compiler->register_file;
  if (!reg.invalid()) {
    assert(bank == reg_file.reg_bank(reg));
    return reg;
  }

  reg = compiler->select_reg(bank);
  reg_file.mark_used(reg, INVALID_VAL_LOCAL_IDX, 0);
  reg_file.mark_clobbered(reg);
  reg_file.mark_fixed(reg);
  return reg;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::RawScratchReg::reset(
    CompilerBase *compiler) {
  if (reg.invalid()) {
    return;
  }

  compiler->register_file.unmark_fixed(reg);
  compiler->register_file.unmark_used(reg);
  reg = AsmReg::make_invalid();
}

/// Owned unspillable and unevictable temporary register with RAII semantics.
template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
struct CompilerBase<Adaptor, Derived, Config>::ScratchReg : RawScratchReg {
private:
  CompilerBase *compiler;

public:
  /// Constructor, no register.
  explicit ScratchReg(CompilerBase *compiler) : compiler(compiler) {}

  explicit ScratchReg(const ScratchReg &) = delete;
  ScratchReg(ScratchReg &&other)
      : RawScratchReg(std::move(other)), compiler(other.compiler) {}

  ~ScratchReg() { reset(); }

  ScratchReg &operator=(const ScratchReg &) = delete;
  ScratchReg &operator=(ScratchReg &&other) {
    if (this != &other) {
      reset();
      RawScratchReg::operator=(std::move(other));
    }
    return *this;
  }

  ScratchReg &operator=(RawScratchReg &&other) {
    reset();
    RawScratchReg::operator=(std::move(other));
    return *this;
  }

  /// Allocate a specific register.
  AsmReg alloc_specific(AsmReg reg) {
    return RawScratchReg::alloc_specific(compiler, reg);
  }

  /// Allocate a general-purpose register.
  AsmReg alloc_gp() { return RawScratchReg::alloc(compiler, Config::GP_BANK); }

  /// Allocate register in the specified bank. Does nothing if it holds a
  /// register in the specified bank. Evicts a register if required.
  AsmReg alloc(RegBank bank) { return RawScratchReg::alloc(compiler, bank); }

  /// Release register, marking it as unused; returns the register.
  AsmReg release() { return RawScratchReg::release(compiler); }

  /// Deallocate register.
  void reset() { RawScratchReg::reset(compiler); }
};

} // namespace tpde
