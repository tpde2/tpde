// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/ValueAssignment.hpp"

#include <cstring>

namespace tpde {

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
struct CompilerBase<Adaptor, Derived, Config>::ValueRef {
  struct AssignmentData {
    /// 0 = unowned reference/invalid, 1 = ref-counted, 2 = owned
    uint8_t mode;
    ValLocalIdx local_idx;
    ValueAssignment *assignment;
  };
  static_assert(ValRefSpecialStruct<AssignmentData>);

  union {
    AssignmentData a;
    Derived::ValRefSpecial s;
  } state;

  CompilerBase *compiler;

  ValueRef(CompilerBase *compiler) noexcept
      : state{AssignmentData()}, compiler(compiler) {}

  ValueRef(CompilerBase *compiler, ValLocalIdx local_idx) noexcept
      : state{AssignmentData{
                           .local_idx = local_idx,
                           .assignment = compiler->val_assignment(local_idx),
                           }
  }, compiler(compiler) {
    assert(!state.a.assignment->pending_free && "access of free'd assignment");

    // Extended liveness checks in debug builds.
#ifndef NDEBUG
    if (!variable_ref()) {
      const auto &liveness =
          compiler->analyzer.liveness_info(state.a.local_idx);
      assert(liveness.last >= compiler->cur_block_idx &&
             "ref-counted value used outside of its live range");
      assert(state.a.assignment->references_left != 0);
      if (state.a.assignment->references_left == 1 && !liveness.last_full) {
        assert(liveness.last == compiler->cur_block_idx &&
               "liveness of non-last-full value must end at last use");
      }
    }
#endif

    if (variable_ref()) {
      state.a.mode = 0;
    } else if (state.a.assignment->references_left <= 1 &&
               !state.a.assignment->delay_free) {
      state.a.mode = 2;
    } else {
      state.a.mode = 1;
    }
  }

  template <typename... T>
  ValueRef(CompilerBase *compiler, T &&...args) noexcept
      : state{.s = typename Derived::ValRefSpecial(std::forward<T>(args)...)},
        compiler(compiler) {
    assert(state.a.mode >= 4);
  }

private:
  // Private copy constructor.
  ValueRef(const ValueRef &other) = default;

public:
  ValueRef(ValueRef &&other) noexcept
      : state{other.state}, compiler(other.compiler) {
    other.state.a = AssignmentData{};
  }

  ~ValueRef() noexcept { reset(); }

  ValueRef &operator=(const ValueRef &) = delete;

  ValueRef &operator=(ValueRef &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    reset();
    assert(compiler == other.compiler);
    this->state = other.state;
    other.state.a.mode = 0;
    return *this;
  }

  bool has_assignment() const noexcept { return state.a.mode < 4; }

  [[nodiscard]] ValueAssignment *assignment() const noexcept {
    assert(has_assignment());
    assert(state.a.assignment != nullptr);
    return state.a.assignment;
  }

  /// Convert into an unowned reference; must be called before first part is
  /// accessed.
  void disown() noexcept {
    if (has_assignment()) {
      state.a.mode = 0;
    }
  }

  /// Get an unowned reference to this value. Previously accessed parts might
  /// already have been destroyed if the value is in its last use.
  ValueRef disowned() noexcept {
    ValueRef res = *this;
    res.disown();
    return res;
  }

  ValLocalIdx local_idx() const noexcept {
    assert(has_assignment());
    return state.a.local_idx;
  }

  ValuePartRef part(unsigned part) noexcept TPDE_LIFETIMEBOUND {
    if (has_assignment()) {
      return ValuePartRef{
          compiler, local_idx(), state.a.assignment, part, state.a.mode == 2};
    }
    return compiler->derived()->val_part_ref_special(state.s, part);
  }

  /// Like part(), but the returned part is always unowned and will not release
  /// registers of the value assignment when reset.
  ValuePartRef part_unowned(unsigned part) noexcept TPDE_LIFETIMEBOUND {
    if (has_assignment()) {
      return ValuePartRef{
          compiler, local_idx(), state.a.assignment, part, false};
    }
    return compiler->derived()->val_part_ref_special(state.s, part);
  }

  /// Reset the reference to the value part
  void reset() noexcept;

  bool variable_ref() const noexcept {
    assert(has_assignment());
    return state.a.assignment->variable_ref;
  }
};

template <IRAdaptor Adaptor, typename Derived, CompilerConfig Config>
void CompilerBase<Adaptor, Derived, Config>::ValueRef::reset() noexcept {
  if (state.a.mode == 1 || state.a.mode == 2) {
    state.a.mode = 0;

    assert(!state.a.assignment->pending_free && "access of free'd assignment");

    auto &ref_count = state.a.assignment->references_left;
    assert(ref_count != 0);
    if (--ref_count == 0) {
      ValLocalIdx local_idx = state.a.local_idx;
      if (!state.a.assignment->delay_free) {
        compiler->free_assignment(local_idx);
      } else {
        // need to wait until release
        TPDE_LOG_TRACE("Delay freeing assignment for value {}",
                       static_cast<u32>(local_idx));
        const auto &liveness = compiler->analyzer.liveness_info(local_idx);
        auto &free_list_head =
            compiler->assignments.delayed_free_lists[u32(liveness.last)];
        state.a.assignment->next_delayed_free_entry = free_list_head;
        state.a.assignment->pending_free = true;
        free_list_head = local_idx;
      }
    }
  }

#ifndef NDEBUG
  state.a.assignment = nullptr;
  state.a.local_idx = INVALID_VAL_LOCAL_IDX;
#endif
}
} // namespace tpde
