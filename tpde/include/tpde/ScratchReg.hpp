// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

namespace tpde {

/// Owned unspillable and unevictable temporary register with RAII semantics.
template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
struct CompilerBase<Adaptor, Derived, Config>::ScratchReg {
private:
  CompilerBase *compiler;
  // TODO(ts): get this using the CompilerConfig?
  AsmReg reg = AsmReg::make_invalid();

public:
  /// Constructor, no register.
  explicit ScratchReg(CompilerBase *compiler) : compiler(compiler) {}

  explicit ScratchReg(const ScratchReg &) = delete;
  ScratchReg(ScratchReg &&) noexcept;

  ~ScratchReg() noexcept { reset(); }

  ScratchReg &operator=(const ScratchReg &) = delete;
  ScratchReg &operator=(ScratchReg &&) noexcept;

  /// Whether a register is currently allocated.
  bool has_reg() const noexcept { return reg.valid(); }

  /// The allocated register.
  AsmReg cur_reg() const noexcept {
    assert(has_reg());
    return reg;
  }

  /// Allocate a specific register.
  AsmReg alloc_specific(AsmReg reg) noexcept;

  /// Allocate a general-purpose register.
  AsmReg alloc_gp() noexcept { return alloc(Config::GP_BANK); }

  /// Allocate register in the specified bank. Does nothing if it holds a
  /// register in the specified bank. Evicts a register if required.
  AsmReg alloc(RegBank bank) noexcept;

  /// Release register, marking it as unused; returns the register.
  AsmReg release() noexcept {
    AsmReg res = reg;
    reset();
    return res;
  }

  /// Deallocate register.
  void reset() noexcept;

  /// @internal Forcefully change register without updating register file.
  /// Avoid.
  void force_set_reg(AsmReg reg) noexcept { this->reg = reg; }
};

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
CompilerBase<Adaptor, Derived, Config>::ScratchReg::ScratchReg(
    ScratchReg &&other) noexcept {
  this->compiler = other.compiler;
  this->reg = other.reg;
  other.reg = AsmReg::make_invalid();
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::ScratchReg &
    CompilerBase<Adaptor, Derived, Config>::ScratchReg::operator=(
        ScratchReg &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  reset();
  this->compiler = other.compiler;
  this->reg = other.reg;
  other.reg = AsmReg::make_invalid();
  return *this;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::AsmReg
    CompilerBase<Adaptor, Derived, Config>::ScratchReg::alloc_specific(
        AsmReg reg) noexcept {
  assert(compiler->may_change_value_state());
  assert(!compiler->register_file.is_fixed(reg));
  reset();

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
    CompilerBase<Adaptor, Derived, Config>::ScratchReg::alloc(
        RegBank bank) noexcept {
  assert(compiler->may_change_value_state());

  auto &reg_file = compiler->register_file;
  if (!reg.invalid()) {
    assert(bank == reg_file.reg_bank(reg));
    return reg;
  }

  reg = compiler->select_reg(bank, /*exclusion_mask=*/0);
  reg_file.mark_used(reg, INVALID_VAL_LOCAL_IDX, 0);
  reg_file.mark_clobbered(reg);
  reg_file.mark_fixed(reg);
  return reg;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::ScratchReg::reset() noexcept {
  if (reg.invalid()) {
    return;
  }

  compiler->register_file.unmark_fixed(reg);
  compiler->register_file.unmark_used(reg);
  reg = AsmReg::make_invalid();
}
} // namespace tpde
