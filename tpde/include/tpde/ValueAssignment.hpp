// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/ValLocalIdx.hpp"
#include "tpde/base.hpp"
#include "tpde/util/AddressSanitizer.hpp"
#include "tpde/util/BumpAllocator.hpp"

#include <array>
#include <cassert>
#include <cstddef>

namespace tpde {

// disable Wpedantic here since these are compiler extensions
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

// GCC < 15 does not support the flexible array member in the union
// see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53548
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ < 15
  #define GNU_ZERO 0
#else
  #define GNU_ZERO
#endif

// TODO(ts): add option to use fixed size assignments if there is no need
// for arbitrarily many parts since that comes with a 1-3% cost in
// compile-time
struct ValueAssignment {
  using Part = u16;

  union {
    /// Offset from the frame pointer.
    i32 frame_off;
    /// For variable-references, frame_off is unused so it can be used
    /// to store an index into a custom structure in case there is
    /// special handling for variable references
    u32 var_ref_custom_idx;
  };

  u32 part_count;

  // we want tight packing and with this construct we get a base size of
  // 16 bytes for values with only one part (the majority)
  union {
    ValueAssignment *next_free_list_entry;

    struct {
      union {
        ValLocalIdx next_delayed_free_entry;
        u32 references_left;
      };
      u8 max_part_size;

      /// Whether the assignment is in a delayed free list and the reference
      /// count is therefore invalid (zero).
      bool pending_free : 1;

      /// Whether the assignment is a single-part variable reference.
      bool variable_ref : 1;

      /// Whether the variable reference refers to a stack slot. Otherwise, this
      /// var_ref_custom_idx is used to identify the variable.
      bool stack_variable : 1;

      /// Whether to delay the free when the reference count reaches zero.
      /// (This is liveness.last_full, copied here for faster access).
      bool delay_free : 1;

      // TODO: get the type of parts from Derived
      union {
        Part first_part;
        Part parts[GNU_ZERO];
      };
    };
  };

  u32 size() const noexcept {
    assert(!variable_ref && "variable-ref has no allocation size");
    return part_count * max_part_size;
  }
};

#undef GNU_ZERO
#pragma GCC diagnostic pop

class AssignmentAllocator {
private:
  // Free list 0 holds VASize = sizeof(ValueAssignment).
  // Free list 1 holds VASize rounded to the next larger power of two.
  // Free list 2 holds twice as much as free list 1. Etc.
  static constexpr size_t SlabSize = 16 * 1024;

  // TODO: define a sensible limit on the maximum number of parts and reduce the
  // number of free lists. Currently the limit is a little under 2**31; limiting
  // the number to e.g. 2**16 would allow halving the number of free lists,
  // where most are unused in the vast majority of cases.
  static constexpr u32 NumFreeLists = 29;
  static constexpr u32 FirstPartOff = offsetof(ValueAssignment, parts);

public:
  static constexpr u32 NumPartsIncluded =
      (sizeof(ValueAssignment) - FirstPartOff) / sizeof(ValueAssignment::Part);

private:
  /// Allocator for small assignments that get stored in free lists. Larger
  /// assignments are allocated directly on the heap.
  util::BumpAllocator<SlabSize> alloc;
  /// Free lists for small ValueAssignments
  std::array<ValueAssignment *, NumFreeLists> fixed_free_lists{};

public:
  AssignmentAllocator() noexcept = default;

  ValueAssignment *allocate(u32 part_count) noexcept {
    if (part_count > NumPartsIncluded) [[unlikely]] {
      return allocate_slow(part_count);
    }
    if (auto *res = fixed_free_lists[0]) {
      util::unpoison_memory_region(res, sizeof(ValueAssignment));
      fixed_free_lists[0] = res->next_free_list_entry;
      return res;
    }
    return new (alloc) ValueAssignment;
  }

  ValueAssignment *allocate_slow(u32 part_count,
                                 bool skip_free_list = false) noexcept;

  void deallocate(ValueAssignment *assignment) noexcept {
    if (assignment->part_count > NumPartsIncluded) [[unlikely]] {
      deallocate_slow(assignment);
      return;
    }
    assignment->next_free_list_entry = fixed_free_lists[0];
    fixed_free_lists[0] = assignment;
    util::poison_memory_region(assignment, sizeof(ValueAssignment));
  }

  void deallocate_slow(ValueAssignment *) noexcept;

  void reset() noexcept {
    alloc.reset();
    fixed_free_lists = {};
  }
};

} // namespace tpde
