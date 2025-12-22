// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/ValueAssignment.hpp"

#include <cstring>
#include <span>

namespace tpde {

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
class CompilerBase<Adaptor, Derived, Config>::ValuePart {
private:
  struct ConstantData {
    AsmReg reg = AsmReg::make_invalid();
    bool has_assignment = false;
    bool owned;
    bool is_const : 1;
    bool const_inline : 1;
    union {
      const u64 *data;
      u64 inline_data;
    };
    RegBank bank;
    u32 size;
  };

  struct ValueData {
    AsmReg reg = AsmReg::make_invalid(); // only valid if fixed/locked
    bool has_assignment = true;
    bool owned;
    ValLocalIdx local_idx;
    u32 part;
    ValueAssignment *assignment;
  };

  union {
    ConstantData c;
    ValueData v;
  } state;

public:
  ValuePart() : state{ConstantData{.is_const = false}} {}

  ValuePart(RegBank bank)
      : state{
            ConstantData{.is_const = false, .bank = bank}
  } {
    assert(bank.id() < Config::NUM_BANKS);
  }

  ValuePart(ValLocalIdx local_idx,
            ValueAssignment *assignment,
            u32 part,
            bool owned)
      : state{
            .v = ValueData{
                           .owned = owned,
                           .local_idx = local_idx,
                           .part = part,
                           .assignment = assignment,
                           }
  } {
    assert(this->assignment().variable_ref() ||
           state.v.assignment->references_left);
    assert(!owned || state.v.assignment->references_left == 1);
  }

  ValuePart(const u64 *data, u32 size, RegBank bank)
      : state{
            .c = ConstantData{.is_const = true,
                              .const_inline = false,
                              .data = data,
                              .bank = bank,
                              .size = size}
  } {
    assert(data && "constant data must not be null");
    assert(bank.id() < Config::NUM_BANKS);
  }

  ValuePart(const u64 val, u32 size, RegBank bank)
      : state{
            .c = ConstantData{.is_const = true,
                              .const_inline = true,
                              .inline_data = val,
                              .bank = bank,
                              .size = size}
  } {
    assert(size <= sizeof(val));
    assert(bank.id() < Config::NUM_BANKS);
  }

  explicit ValuePart(const ValuePart &) = delete;

  ValuePart(ValuePart &&other) : state{other.state} {
    other.state.c = ConstantData{.is_const = false, .bank = bank()};
  }

  ~ValuePart() {
    assert(!state.c.reg.valid() && "must call reset() on ValuePart explicitly");
  }

  ValuePart &operator=(const ValuePart &) = delete;

  ValuePart &operator=(ValuePart &&other) {
    if (this == &other) {
      return *this;
    }
    assert(!state.c.reg.valid() && "must call reset() on ValuePart explicitly");
    this->state = other.state;
    other.state.c = ConstantData{.is_const = false, .bank = bank()};
    return *this;
  }

  bool has_assignment() const { return state.v.has_assignment; }

  bool is_const() const { return !state.c.has_assignment && state.c.is_const; }

  bool is_owned() const {
    assert(has_assignment());
    return state.c.owned;
  }

  [[nodiscard]] AssignmentPartRef assignment() const {
    assert(has_assignment());
    return AssignmentPartRef{state.v.assignment, state.v.part};
  }

  /// If it is known that the value part has a register, this function can be
  /// used to quickly access it
  AsmReg cur_reg() const {
    assert(state.v.reg.valid());
    return state.v.reg;
  }

  /// Current register or none, even if the value is unlocked and could be
  /// evicted by any other operation.
  AsmReg cur_reg_unlocked() const {
    if (state.v.reg.valid()) {
      return state.v.reg;
    }
    if (has_assignment()) {
      if (auto ap = assignment(); ap.register_valid()) {
        return ap.get_reg();
      }
    }
    return AsmReg::make_invalid();
  }

  /// Is the value part currently in the specified register?
  bool is_in_reg(AsmReg reg) const {
    if (has_reg()) {
      return cur_reg() == reg;
    }
    if (has_assignment()) {
      auto ap = assignment();
      return ap.register_valid() && ap.get_reg() == reg;
    }
    return false;
  }

  bool has_reg() const { return state.v.reg.valid(); }

private:
  template <bool Reload>
  void alloc_reg_impl(CompilerBase *compiler);
  AsmReg alloc_specific_impl(CompilerBase *compiler, AsmReg reg, bool reload);

public:
  /// Allocate and lock a register for the value part, *without* reloading the
  /// value. Asserts that no register is currently allocated.
  AsmReg alloc_reg(CompilerBase *compiler) {
    alloc_reg_impl</*Reload=*/false>(compiler);
    return cur_reg();
  }

  /// Allocate and lock a register for the value part, *without* reloading the
  /// value. Does nothing if a register is already allocated.
  AsmReg cur_reg_or_alloc(CompilerBase *compiler) {
    if (!has_reg()) {
      alloc_reg_impl</*Reload=*/false>(compiler);
    }
    return cur_reg();
  }

  /// Allocate register, but try to reuse the register from ref first. This
  /// method is complicated and must be used carefully. If ref is locked in a
  /// register and owns the register (can_salvage()), the ownership of the
  /// register is transferred to this ValuePart without modifying the value.
  /// Otherwise, a new register is allocated.
  ///
  /// Usage example:
  ///   AsmReg operand_reg = operand_ref.load_to_reg();
  ///   AsmReg result_reg = result_ref.alloc_try_reuse(operand_ref);
  ///   if (operand_reg == result_reg) {
  ///     // reuse successful
  ///     ASM(ADD64ri, result_reg, 1);
  ///   } else {
  ///     ASM(LEA64rm, result_reg, FE_MEM(FE_NOREG, 1, operand_reg, 1));
  ///   }
  AsmReg alloc_try_reuse(CompilerBase *compiler, ValuePart &ref) {
    assert(ref.has_reg());
    if (!has_assignment() || !assignment().register_valid()) {
      assert(!has_assignment() || !assignment().fixed_assignment());
      if (ref.can_salvage()) {
        set_value(compiler, std::move(ref));
        if (has_assignment()) {
          lock(compiler);
        }
        return cur_reg();
      }
    }
    return alloc_reg(compiler);
  }

  /// Allocate and lock a specific register for the value part, spilling the
  /// register if it is currently used (must not be fixed), *without* reloading
  /// or copying the value into the new register. The value must not be locked.
  /// An existing assignment register is discarded. Value part must not be a
  /// fixed assignment.
  void alloc_specific(CompilerBase *compiler, AsmReg reg) {
    alloc_specific_impl(compiler, reg, false);
  }

  /// Allocate, fill, and lock a register for the value part, reloading from
  /// the stack or materializing the constant if necessary. Requires that the
  /// value is currently unlocked (i.e., has_reg() is false).
  AsmReg load_to_reg(CompilerBase *compiler) {
    alloc_reg_impl</*Reload=*/true>(compiler);
    return cur_reg();
  }

  /// Allocate, fill, and lock a specific register for the value part, spilling
  /// the register if it is currently used (must not be fixed). The value is
  /// moved (assignment updated) or reloaded to this register. Value part must
  /// not be a fixed assignment.
  ///
  /// \warning Do not overwrite the register content as it is not saved
  /// \note The target register or the current value part may not be fixed
  void load_to_specific(CompilerBase *compiler, AsmReg reg) {
    alloc_specific_impl(compiler, reg, true);
  }

  /// Copy value into a different register.
  AsmReg reload_into_specific_fixed(CompilerBase *compiler,
                                    AsmReg reg,
                                    unsigned size = 0);

  /// For a locked value, get an unonwed ValuePart referring to the register.
  ValuePart get_unowned() {
    assert(has_reg());
    ValuePart res{bank()};
    res.state.c =
        ConstantData{.reg = cur_reg(), .owned = false, .is_const = false};
    return res;
  }

  /// Move into a temporary register, reuse an existing register if possible.
  ValuePart into_temporary(CompilerBase *compiler) && {
    if (is_const()) {
      if (state.c.const_inline) {
        ValuePart res{state.c.inline_data, state.c.size, state.c.bank};
        res.load_to_reg(compiler);
        return res;
      } else {
        ValuePart res{state.c.data, state.c.size, state.c.bank};
        res.load_to_reg(compiler);
        return res;
      }
    }

    // TODO: implement this. This needs size information to copy the value.
    assert((has_assignment() || state.c.owned) &&
           "into_temporary from unowned ValuePart not implemented");
    ValuePart res{bank()};
    res.set_value(compiler, std::move(*this));
    if (!res.has_reg()) [[unlikely]] {
      assert(res.is_const());
      res.load_to_reg(compiler);
    }
    return res;
  }

  /// Move into a scratch register, reuse an existing register if possible.
  ScratchReg into_scratch(CompilerBase *compiler) && {
    // TODO: implement this. This needs size information to copy the value.
    assert((has_assignment() || state.c.owned || state.c.is_const) &&
           "into_scratch from unowned ValuePart not implemented");
    ScratchReg res{compiler};
    if (can_salvage()) {
      res.alloc_specific(salvage(compiler));
    } else {
      reload_into_specific_fixed(compiler, res.alloc(bank()));
    }
    return res;
  }

  /// Extend integer value, reuse existing register if possible. Constants are
  /// extended without allocating a register.
  ValuePart
      into_extended(CompilerBase *compiler, bool sign, u32 from, u32 to) && {
    assert(from < to && "invalid integer extension sizes");
    if (is_const() && to <= 64) {
      u64 val = const_data()[0];
      u64 extended = sign ? util::sext(val, from) : util::zext(val, from);
      return ValuePart{extended, (to + 7) / 8, state.c.bank};
    }
    ValuePart res{bank()};
    Reg src_reg = has_reg() ? cur_reg() : load_to_reg(compiler);
    if (can_salvage()) {
      res.set_value(compiler, std::move(*this));
      assert(src_reg == res.cur_reg());
    } else {
      res.alloc_reg(compiler);
    }
    compiler->derived()->generate_raw_intext(
        res.cur_reg(), src_reg, sign, from, to);
    return res;
  }

  void lock(CompilerBase *compiler);
  void unlock(CompilerBase *compiler);

  void set_modified() {
    assert(has_reg() && has_assignment());
    assignment().set_modified(true);
  }

  /// Set the value to the value of a different value part, possibly taking
  /// ownership of allocated registers. If this value part has an assignment,
  /// the value part will be unlocked.
  void set_value(CompilerBase *compiler, ValuePart &&other);

  /// Set the value to the value of the scratch register, taking ownership of
  /// the register.
  void set_value(CompilerBase *compiler, ScratchReg &&other);

  /// Set the value to the value of the specified register, possibly taking
  /// ownership of the register. Intended for filling in arguments/calls results
  /// which inherently get stored to fixed registers. There must not be a
  /// currently locked register.
  void set_value_reg(CompilerBase *compiler, AsmReg reg);

  bool can_salvage() const {
    if (!has_assignment()) {
      return state.c.owned && state.c.reg.valid();
    }

    return state.v.owned && assignment().register_valid();
  }

private:
  AsmReg salvage_keep_used(CompilerBase *compiler);

public:
  // only call when can_salvage returns true and a register is known to be
  // allocated
  AsmReg salvage(CompilerBase *compiler) {
    AsmReg reg = salvage_keep_used(compiler);
    compiler->register_file.unmark_used(reg);
    return reg;
  }

  ValLocalIdx local_idx() const {
    assert(has_assignment());
    return state.v.local_idx;
  }

  u32 part() const {
    assert(has_assignment());
    return state.v.part;
  }

  RegBank bank() const {
    return !has_assignment() ? state.c.bank : assignment().bank();
  }

  u32 part_size() const {
    return !has_assignment() ? state.c.size : assignment().part_size();
  }

  std::span<const u64> const_data() const {
    assert(is_const());
    if (state.c.const_inline) {
      return {&state.c.inline_data, 1};
    }
    return {state.c.data, (state.c.size + 7) / 8};
  }

  /// Reset the reference to the value part
  void reset(CompilerBase *compiler);
};

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
template <bool Reload>
void CompilerBase<Adaptor, Derived, Config>::ValuePart::alloc_reg_impl(
    CompilerBase *compiler) {
  // The caller has no control over the selected register, so it must assume
  // that this function evicts some register. This is not permitted if the value
  // state ought to be the same.
  assert(compiler->may_change_value_state());
  assert(!state.c.reg.valid());

  RegBank bank;
  if (has_assignment()) {
    auto ap = assignment();
    if (ap.register_valid()) {
      lock(compiler);
      return;
    }

    bank = ap.bank();
  } else {
    bank = state.c.bank;
  }

  Reg reg = compiler->select_reg(bank);
  auto &reg_file = compiler->register_file;
  reg_file.mark_clobbered(reg);
  if (has_assignment()) {
    reg_file.mark_used(reg, state.v.local_idx, state.v.part);
    // Essentially lock(), except that we know that the lock count is zero.
    // We must lock the value here, otherwise, load_from_stack could evict the
    // register again.
    reg_file.mark_fixed(reg);
    state.v.reg = reg;
    auto ap = assignment();
    ap.set_reg(reg);
    ap.set_register_valid(true);

    if constexpr (Reload) {
      compiler->derived()->reload_to_reg(reg, ap);
    } else {
      assert(!ap.stack_valid() && "alloc_reg called on initialized value");
    }
  } else {
    reg_file.mark_used(reg, INVALID_VAL_LOCAL_IDX, 0);
    reg_file.mark_fixed(reg);
    state.c.reg = reg;
    state.c.owned = true;

    if constexpr (Reload) {
      assert(is_const() && "cannot reload temporary value");
      compiler->derived()->materialize_constant(
          const_data().data(), state.c.bank, state.c.size, reg);
    }
  }
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::AsmReg
    CompilerBase<Adaptor, Derived, Config>::ValuePart::alloc_specific_impl(
        CompilerBase *compiler, AsmReg reg, const bool reload) {
  assert(!state.c.reg.valid());

  if (has_assignment()) {
    auto ap = assignment();
    assert(!ap.fixed_assignment());

    if (ap.register_valid() && ap.get_reg() == reg) {
      lock(compiler);
      return ap.get_reg();
    }
  }

  auto &reg_file = compiler->register_file;
  if (reg_file.is_used(reg)) {
    compiler->evict_reg(reg);
  }

  reg_file.mark_clobbered(reg);
  if (has_assignment()) {
    assert(compiler->may_change_value_state());

    reg_file.mark_used(reg, state.v.local_idx, state.v.part);
    auto ap = assignment();
    auto old_reg = AsmReg::make_invalid();
    if (ap.register_valid()) {
      old_reg = ap.get_reg();
    }

    ap.set_reg(reg);
    ap.set_register_valid(true);

    // We must lock the value here, otherwise, load_from_stack could evict the
    // register again.
    lock(compiler);

    if (reload) {
      if (old_reg.valid()) {
        compiler->derived()->mov(reg, old_reg, ap.part_size());
        reg_file.unmark_used(old_reg);
      } else {
        compiler->derived()->reload_to_reg(reg, ap);
      }
    } else {
      assert(!ap.stack_valid() && "alloc_reg with valid stack slot");
    }
  } else {
    reg_file.mark_used(reg, INVALID_VAL_LOCAL_IDX, 0);
    reg_file.mark_fixed(reg);

    if (reload) {
      if (state.c.reg.valid()) {
        // TODO: size
        compiler->derived()->mov(reg, state.c.reg, 8);
        reg_file.unmark_fixed(state.c.reg);
        reg_file.unmark_used(state.c.reg);
      } else {
        assert(is_const() && "cannot reload temporary value");
        compiler->derived()->materialize_constant(
            const_data().data(), state.c.bank, state.c.size, reg);
      }
    }

    state.c.reg = reg;
    state.c.owned = true;
  }

  return reg;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::AsmReg
    CompilerBase<Adaptor, Derived, Config>::ValuePart::
        reload_into_specific_fixed(CompilerBase *compiler,
                                   AsmReg reg,
                                   unsigned size) {
  if (is_const()) {
    compiler->derived()->materialize_constant(
        const_data().data(), state.c.bank, state.c.size, reg);
    return reg;
  }
  if (!has_assignment()) {
    assert(has_reg());
    assert(reg != cur_reg());
    // TODO: value size
    assert(size != 0);
    compiler->derived()->mov(reg, cur_reg(), size);
    return reg;
  }

  auto ap = assignment();
  if (has_reg()) {
    assert(cur_reg() != reg);
    compiler->derived()->mov(reg, cur_reg(), ap.part_size());
  } else if (ap.register_valid()) {
    assert(ap.get_reg() != reg);

    compiler->derived()->mov(reg, ap.get_reg(), ap.part_size());
  } else {
    assert(!ap.fixed_assignment());
    compiler->derived()->reload_to_reg(reg, ap);
  }

  compiler->register_file.mark_clobbered(reg);
  return reg;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::ValuePart::lock(
    CompilerBase *compiler) {
  assert(has_assignment());
  assert(!has_reg());
  auto ap = assignment();
  assert(ap.register_valid());

  const auto reg = ap.get_reg();
  compiler->register_file.inc_lock_count(reg);
  state.v.reg = reg;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::ValuePart::unlock(
    CompilerBase *compiler) {
  assert(has_assignment());
  if (!state.v.reg.valid()) {
    return;
  }

  compiler->register_file.dec_lock_count(state.v.reg);
  state.v.reg = AsmReg::make_invalid();
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::ValuePart::set_value(
    CompilerBase *compiler, ValuePart &&other) {
  assert(this != &other && "cannot assign ValuePart to itself");
  auto &reg_file = compiler->register_file;
  if (!has_assignment()) {
    assert(!is_const()); // probably don't want to allow mutating constants

    // This is a temporary, which might currently have a register. We want to
    // have a temporary register that holds the value at the end.
    if (!other.has_assignment()) {
      // When other is a temporary/constant, just take the value and drop our
      // own register (if we have any).
      reset(compiler);
      *this = std::move(other);
      return;
    }

    if (!other.can_salvage()) {
      // We cannot take the register of other, so copy the value
      AsmReg cur_reg = alloc_reg(compiler);
      other.reload_into_specific_fixed(compiler, cur_reg);
      other.reset(compiler);
      return;
    }

    // We can take the register of other.
    reset(compiler);

    state.c.reg = other.salvage_keep_used(compiler);
    state.c.owned = true;
    reg_file.mark_fixed(state.c.reg);
    reg_file.update_reg_assignment(state.c.reg, INVALID_VAL_LOCAL_IDX, 0);
    return;
  }

  // Update the value of the assignment part
  auto ap = assignment();
  assert(!ap.variable_ref() && "cannot update variable ref");

  if (ap.fixed_assignment() || !other.can_salvage()) {
    if constexpr (WithAsserts) {
      // alloc_reg has the assertion that stack_valid must be false to prevent
      // accidental loss of information. set_value behaves more like an explicit
      // assignment, so we permit this overwrite -- but need to disable the
      // assertion.
      ap.set_modified(true);
    }
    // Source value owns no register or it is not reusable: copy value
    AsmReg cur_reg = alloc_reg(compiler);
    other.reload_into_specific_fixed(compiler, cur_reg, ap.part_size());
    other.reset(compiler);
    unlock(compiler);
    ap.set_register_valid(true);
    ap.set_modified(true);
    return;
  }

  // Reuse register of other assignment
  if (ap.register_valid()) {
    // If we currently have a register, drop it
    unlock(compiler);
    auto cur_reg = ap.get_reg();
    assert(!reg_file.is_fixed(cur_reg));
    reg_file.unmark_used(cur_reg);
  }

  AsmReg new_reg = other.salvage_keep_used(compiler);
  reg_file.update_reg_assignment(new_reg, local_idx(), part());
  ap.set_reg(new_reg);
  ap.set_register_valid(true);
  ap.set_modified(true);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::ValuePart::set_value(
    CompilerBase *compiler, ScratchReg &&other) {
  assert(compiler->may_change_value_state());

  auto &reg_file = compiler->register_file;

  // We could support this, but there shouldn't bee the need for that.
  assert(other.has_reg() && "cannot initialize with invalid register");
  Reg value_reg = other.cur_reg();
  assert(reg_file.is_fixed(value_reg));
  assert(reg_file.is_used(value_reg));
  assert(reg_file.is_clobbered(value_reg));
  assert(!state.c.reg.valid() &&
         "attempted to overwrite already initialized and locked ValuePartRef");

  if (!has_assignment()) {
    assert(!is_const() && "cannot mutate constant ValuePartRef");
    state.c.reg = value_reg;
    state.c.owned = true;
    assert(reg_file.reg_local_idx(value_reg) == INVALID_VAL_LOCAL_IDX);
    assert(reg_file.reg_part(value_reg) == 0);
    other.force_set_reg(AsmReg::make_invalid());
    return;
  }

  // Update the value of the assignment part
  auto ap = assignment();
  assert(!ap.variable_ref() && "cannot update variable ref");

  if (ap.fixed_assignment()) {
    // For fixed assignments, copy the value into the fixed register.
    auto cur_reg = ap.get_reg();
    assert(reg_file.is_used(cur_reg));
    assert(reg_file.is_fixed(cur_reg));
    assert(reg_file.reg_local_idx(cur_reg) == local_idx());
    assert(ap.register_valid() && !ap.stack_valid() &&
           "invalid state for fixed assignment");
    assert(cur_reg != value_reg);
    compiler->derived()->mov(cur_reg, value_reg, ap.part_size());
    other.reset();
    return;
  }

  // Otherwise, take the register.
  assert(!ap.register_valid() && !ap.stack_valid() &&
         "attempted to overwrite already initialized ValuePartRef");

  // ScratchReg's reg is fixed and used => unfix, keep used, update assignment
  reg_file.unmark_fixed(value_reg);
  reg_file.update_reg_assignment(value_reg, local_idx(), part());
  ap.set_reg(value_reg);
  ap.set_register_valid(true);
  ap.set_modified(true);
  other.force_set_reg(AsmReg::make_invalid());
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::ValuePart::set_value_reg(
    CompilerBase *compiler, AsmReg value_reg) {
  assert(compiler->may_change_value_state());

  auto &reg_file = compiler->register_file;

  // We could support this, but there shouldn't bee the need for that.
  assert(value_reg.valid() && "cannot initialize with invalid register");
  assert(!state.c.reg.valid() &&
         "attempted to overwrite already initialized and locked ValuePartRef");

  if (!has_assignment()) {
    assert(!is_const() && "cannot mutate constant ValuePartRef");
    state.c.reg = value_reg;
    state.c.owned = true;
    reg_file.mark_used(state.c.reg, INVALID_VAL_LOCAL_IDX, 0);
    reg_file.mark_fixed(state.c.reg);
    return;
  }

  // Update the value of the assignment part
  auto ap = assignment();
  assert(!ap.variable_ref() && "cannot update variable ref");

  if (ap.fixed_assignment()) {
    // For fixed assignments, copy the value into the fixed register.
    auto cur_reg = ap.get_reg();
    assert(reg_file.is_used(cur_reg));
    assert(reg_file.is_fixed(cur_reg));
    assert(reg_file.reg_local_idx(cur_reg) == local_idx());
    // TODO: can this happen? If so, conditionally emit move.
    assert(cur_reg != value_reg);
    compiler->derived()->mov(cur_reg, value_reg, ap.part_size());
    ap.set_register_valid(true);
    ap.set_modified(true);
    return;
  }

  // Otherwise, take the register.
  assert(!ap.register_valid() && !ap.stack_valid() &&
         "attempted to overwrite already initialized ValuePartRef");

  reg_file.mark_used(value_reg, local_idx(), part());
  reg_file.mark_clobbered(value_reg);
  ap.set_reg(value_reg);
  ap.set_register_valid(true);
  ap.set_modified(true);
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
typename CompilerBase<Adaptor, Derived, Config>::AsmReg
    CompilerBase<Adaptor, Derived, Config>::ValuePart::salvage_keep_used(
        CompilerBase *compiler) {
  assert(compiler->may_change_value_state());
  assert(can_salvage());
  if (!has_assignment()) {
    AsmReg reg = state.c.reg;
    compiler->register_file.unmark_fixed(reg);
    state.c.reg = AsmReg::make_invalid();
    return reg;
  }

  auto ap = assignment();
  assert(ap.register_valid());
  auto cur_reg = ap.get_reg();

  unlock(compiler);
  assert(ap.fixed_assignment() || !compiler->register_file.is_fixed(cur_reg));
  if (ap.fixed_assignment()) {
    compiler->register_file.dec_lock_count(cur_reg); // release fixed register
    --compiler->assignments.cur_fixed_assignment_count[ap.bank().id()];
  }

  ap.set_register_valid(false);
  ap.set_fixed_assignment(false);
  return cur_reg;
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::ValuePart::reset(
    CompilerBase *compiler) {
  AsmReg reg = state.c.reg;
  if (!reg.valid()) {
    return;
  }

  // In debug builds, touch assignment to catch cases where the assignment was
  // already free'ed.
  assert(!has_assignment() || assignment().modified() || true);

  if (state.c.owned) {
    if (has_assignment()) {
      AssignmentPartRef ap = assignment();
      bool fixed = ap.fixed_assignment();
      ap.set_register_valid(false);
      ap.set_fixed_assignment(false);
      compiler->register_file.dec_lock_count_must_zero(reg, fixed ? 2 : 1);
      if (fixed) {
        --compiler->assignments.cur_fixed_assignment_count[ap.bank().id()];
      }
    } else {
      compiler->register_file.unmark_fixed(reg);
    }
    compiler->register_file.unmark_used(reg);
  } else if (has_assignment()) {
    compiler->register_file.dec_lock_count(reg);
  }

  state.c.reg = AsmReg::make_invalid();
}

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
struct CompilerBase<Adaptor, Derived, Config>::ValuePartRef : ValuePart {
  CompilerBase *compiler;

  template <typename... Args>
  ValuePartRef(CompilerBase *compiler, Args &&...args)
      : ValuePart(std::forward<Args>(args)...), compiler(compiler) {}

  explicit ValuePartRef(const ValuePartRef &) = delete;

  ValuePartRef(ValuePartRef &&other)
      : ValuePart(std::move(other)), compiler(other.compiler) {}

  ~ValuePartRef() { reset(); }

  ValuePartRef &operator=(const ValuePartRef &) = delete;

  ValuePartRef &operator=(ValuePartRef &&other) {
    if (this == &other) {
      return *this;
    }
    reset();
    ValuePart::operator=(std::move(other));
    return *this;
  }

  ValuePartRef &operator=(ValuePart &&other) {
    reset();
    ValuePart::operator=(std::move(other));
    return *this;
  }

  AsmReg alloc_reg() { return ValuePart::alloc_reg(compiler); }

  AsmReg cur_reg_or_alloc() { return ValuePart::cur_reg_or_alloc(compiler); }

  AsmReg alloc_try_reuse(ValuePart &ref) {
    return ValuePart::alloc_try_reuse(compiler, ref);
  }

  void alloc_specific(AsmReg reg) { ValuePart::alloc_specific(compiler, reg); }

  AsmReg load_to_reg() { return ValuePart::load_to_reg(compiler); }

  void load_to_specific(AsmReg reg) {
    ValuePart::load_to_specific(compiler, reg);
  }

  AsmReg reload_into_specific_fixed(AsmReg reg, unsigned size = 0) {
    return ValuePart::reload_into_specific_fixed(compiler, reg, size);
  }

  AsmReg reload_into_specific_fixed(CompilerBase *compiler,
                                    AsmReg reg,
                                    unsigned size = 0) {
    return ValuePart::reload_into_specific_fixed(compiler, reg, size);
  }

  ValuePartRef get_unowned_ref() {
    return ValuePartRef{compiler, ValuePart::get_unowned()};
  }

  ValuePartRef into_temporary() && {
    return ValuePartRef{
        compiler,
        std::move(*static_cast<ValuePart *>(this)).into_temporary(compiler)};
  }

  ScratchReg into_scratch() && {
    return std::move(*static_cast<ValuePart *>(this)).into_scratch(compiler);
  }

  ValuePartRef into_extended(bool sign, u32 from, u32 to) && {
    return ValuePartRef{compiler,
                        std::move(*static_cast<ValuePart *>(this))
                            .into_extended(compiler, sign, from, to)};
  }

  void lock() { ValuePart::lock(compiler); }
  void unlock() { ValuePart::unlock(compiler); }

  void set_value(ValuePart &&other) {
    ValuePart::set_value(compiler, std::move(other));
  }

  void set_value(ScratchReg &&other) {
    ValuePart::set_value(compiler, std::move(other));
  }

  void set_value_reg(AsmReg value_reg) {
    ValuePart::set_value_reg(compiler, value_reg);
  }

  AsmReg salvage() { return ValuePart::salvage(compiler); }

  void reset() { ValuePart::reset(compiler); }
};

} // namespace tpde
