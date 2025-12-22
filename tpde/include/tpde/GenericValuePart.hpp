// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <array>
#include <cstring>
#include <span>

namespace tpde {

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
struct CompilerBase<Adaptor, Derived, Config>::GenericValuePart {
  struct Expr {
    std::variant<AsmReg, ScratchReg> base;
    std::variant<AsmReg, ScratchReg> index;
    i64 scale;
    i64 disp;

    explicit Expr() : base{AsmReg::make_invalid()}, scale{0}, disp{0} {}

    explicit Expr(AsmReg base, i64 disp = 0)
        : base(base), scale(0), disp(disp) {}

    explicit Expr(ScratchReg &&base, i64 disp = 0)
        : base(std::move(base)), scale(0), disp(disp) {}

    AsmReg base_reg() const {
      if (std::holds_alternative<AsmReg>(base)) {
        return std::get<AsmReg>(base);
      }
      return std::get<ScratchReg>(base).cur_reg();
    }

    [[nodiscard]] bool has_base() const {
      if (std::holds_alternative<AsmReg>(base)) {
        return std::get<AsmReg>(base).valid();
      }
      return true;
    }

    AsmReg index_reg() const {
      assert(scale != 0 && "index_reg() called on invalid index");
      assert((scale != 1 || has_base()) &&
             "Expr with unscaled index must have base");
      if (std::holds_alternative<AsmReg>(index)) {
        return std::get<AsmReg>(index);
      }
      return std::get<ScratchReg>(index).cur_reg();
    }

    [[nodiscard]] bool has_index() const { return scale != 0; }
  };

  // TODO(ts): evaluate the use of std::variant
  // TODO(ts): I don't like the ValuePartRefs but we also don't want to
  // force all the operands into registers at the start of the encoding...
  std::variant<std::monostate, ValuePartRef, ScratchReg, Expr> state;

  GenericValuePart() = default;

  GenericValuePart(GenericValuePart &) = delete;

  GenericValuePart(GenericValuePart &&other) {
    state = std::move(other.state);
    other.state = std::monostate{};
  }

  GenericValuePart &operator=(const GenericValuePart &) = delete;

  GenericValuePart &operator=(GenericValuePart &&other) {
    if (this == &other) {
      return *this;
    }
    state = std::move(other.state);
    other.state = std::monostate{};
    return *this;
  }

  // salvaging
  GenericValuePart(ScratchReg &&reg) : state{std::move(reg)} {
    assert(std::get<ScratchReg>(state).has_reg());
  }

  // salvaging
  GenericValuePart(ValuePartRef &&ref) : state{std::move(ref)} {}

  GenericValuePart(Expr expr) : state{std::move(expr)} {}

  [[nodiscard]] bool is_expr() const {
    return std::holds_alternative<Expr>(state);
  }

  [[nodiscard]] bool is_imm() const {
    auto *ptr = std::get_if<ValuePartRef>(&state);
    return ptr && ptr->is_const();
  }

  u32 imm_size() const {
    assert(is_imm());
    return val_ref().part_size();
  }

  [[nodiscard]] u64 imm64() const {
    assert(imm_size() <= 8);
    return val_ref().const_data()[0];
  }

  [[nodiscard]] const ValuePartRef &val_ref() const {
    return std::get<ValuePartRef>(state);
  }

  [[nodiscard]] ValuePartRef &val_ref() {
    return std::get<ValuePartRef>(state);
  }

  void reset() { state = std::monostate{}; }
};
} // namespace tpde
