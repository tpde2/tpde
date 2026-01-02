// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "tpde/util/SmallVector.hpp"

#include "tpde/base.hpp"
#include "tpde/util/AddressSanitizer.hpp"

#include <cstdlib>
#include <cstring>

namespace tpde::util {

static size_t calc_new_capacity(size_t cap, size_t min_size) {
  size_t new_cap = 2 * cap + 1;
  return new_cap < min_size ? min_size : new_cap;
}

void *SmallVectorUntypedBase::grow_malloc(size_type min_size,
                                          size_type elem_sz,
                                          size_type &new_cap,
                                          size_type align) {
  new_cap = calc_new_capacity(cap, min_size);
  void *new_alloc = std::aligned_alloc(align, new_cap * elem_sz);
  if (!new_alloc) {
    TPDE_FATAL("SmallVector allocation failed");
  }
  return new_alloc;
}

void SmallVectorUntypedBase::grow_trivial(size_type min_size,
                                          size_type elem_sz,
                                          size_type align) {
  size_type new_cap = calc_new_capacity(cap, min_size);
  void *new_alloc;
  if (is_small()) {
    new_alloc = std::aligned_alloc(align, new_cap * elem_sz);
    if (!new_alloc) {
      TPDE_FATAL("SmallVector allocation failed");
    }
    if (sz > 0) {
      std::memcpy(new_alloc, ptr, sz * elem_sz);
    }
    poison_memory_region(ptr, cap * elem_sz);
  } else {
    // cppreference is really unclear on this...
    // does realloc on a pointer from aligned_alloc stay aligned?
    new_alloc = std::realloc(ptr, new_cap * elem_sz);
    if (!new_alloc) {
      TPDE_FATAL("SmallVector allocation failed");
    }
  }
  ptr = new_alloc;
  cap = new_cap;
  poison_memory_region((char *)ptr + sz * elem_sz, (cap - sz) * elem_sz);
}

SmallVectorUntypedBase::SmallVectorUntypedBase(size_type cap)
    : ptr(small_ptr()), sz(0), cap(cap) {}

bool SmallVectorUntypedBase::is_small() const { return ptr == small_ptr(); }

void *SmallVectorUntypedBase::get_small_elements() const {
  return const_cast<void *>(static_cast<const void *>(this + 1)); // (cz) cursed
};
} // end namespace tpde::util
