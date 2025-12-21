// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/ValueAssignment.hpp"

#include <limits>

namespace tpde {

namespace {

struct AssignmentAllocInfo {
  u32 size;
  u32 alloc_size;
  u32 free_list_idx;

  AssignmentAllocInfo(u32 part_count) noexcept;
};

AssignmentAllocInfo::AssignmentAllocInfo(u32 part_count) noexcept {
  constexpr u32 VASize = sizeof(ValueAssignment);
  constexpr u32 PartSize = sizeof(ValueAssignment::Part);

  size = VASize;
  alloc_size = VASize;
  free_list_idx = 0;
  if (part_count > AssignmentAllocator::NumPartsIncluded) {
    assert(part_count < (std::numeric_limits<u32>::max() - VASize) / PartSize);
    size += (part_count - AssignmentAllocator::NumPartsIncluded) * PartSize;
    // Round size to next power of two.
    static_assert((VASize & (VASize - 1)) == 0,
                  "non-power-of-two ValueAssignment size untested");
    constexpr u32 clz_off = util::cnt_lz<u32>(VASize >> 1);
    free_list_idx = clz_off - util::cnt_lz<u32>(size - 1);
    alloc_size = u32{1} << (32 - clz_off + free_list_idx);
  }
}

} // end anonymous namespace

ValueAssignment *
    AssignmentAllocator::allocate_slow(uint32_t part_count,
                                       bool skip_free_list) noexcept {
  AssignmentAllocInfo aai(part_count);

  if (!skip_free_list) {
    auto &free_list = fixed_free_lists[aai.free_list_idx];
    if (auto *assignment = free_list) {
      util::unpoison_memory_region(assignment, aai.size);
      free_list = assignment->next_free_list_entry;
      return assignment;
    }
  }

  auto *buf = alloc.allocate(aai.alloc_size, alignof(ValueAssignment));
  return new (reinterpret_cast<ValueAssignment *>(buf)) ValueAssignment{};
}

void AssignmentAllocator::deallocate_slow(
    ValueAssignment *assignment) noexcept {
  AssignmentAllocInfo aai(assignment->part_count);
  assignment->next_free_list_entry = fixed_free_lists[aai.free_list_idx];
  fixed_free_lists[aai.free_list_idx] = assignment;
  util::poison_memory_region(assignment, aai.size);
}

} // namespace tpde
