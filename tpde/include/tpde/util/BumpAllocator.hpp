// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/util/AddressSanitizer.hpp"
#include "tpde/util/SmallVector.hpp"
#include "tpde/util/misc.hpp"

#include <algorithm>
#include <cstddef>
#include <new>

namespace tpde::util {

template <size_t SlabSize = 4096>
class BumpAllocator {
  uintptr_t cur = 0;
  uintptr_t end = 0;
  SmallVector<void *> slabs;
  SmallVector<std::pair<void *, size_t>, 0> large_slabs;

  static constexpr auto SLAB_ALIGNMENT =
      std::align_val_t(alignof(std::max_align_t));

  /// Slab size for the i-th slab.
  static constexpr size_t slab_size(size_t i) {
    // Double slab size every 8 slabs.
    return SlabSize << (i / 8);
  }

public:
  BumpAllocator() = default;
  ~BumpAllocator() {
    deallocate_slabs();
    deallocate_large_slabs();
  }

  BumpAllocator(const BumpAllocator &) = delete;

  BumpAllocator &operator=(const BumpAllocator &) = delete;

  /// Deallocate all but the first slab and reset current pointer to beginning.
  void reset() {
    if (!slabs.empty()) {
      deallocate_slabs(1);
      slabs.resize(1);
      util::poison_memory_region(slabs[0], slab_size(0));
      cur = reinterpret_cast<uintptr_t>(slabs[0]);
      end = cur + slab_size(0);
    }
    deallocate_large_slabs();
  }

  void *allocate(size_t size, size_t align) {
    assert(size > 0 && "cannot perform zero-sized allocation");
    uintptr_t aligned = align_up(cur, align);
    uintptr_t alloc_end = aligned + size;
    if (alloc_end <= end) [[likely]] {
      cur = alloc_end;
      util::unpoison_memory_region(reinterpret_cast<void *>(aligned), size);
      return reinterpret_cast<void *>(aligned);
    }
    return allocate_slab(size, align);
  }

  void *allocate_slab(size_t size, [[maybe_unused]] size_t align) {
    assert(align <= alignof(std::max_align_t) && "alignment type unsupported");
    size_t slab_sz = slab_size(slabs.size());
    if (size > slab_sz) [[unlikely]] {
      void *slab = allocate_mem(size, SLAB_ALIGNMENT);
      large_slabs.emplace_back(slab, size);
      return slab;
    }

    void *slab = allocate_mem(slab_sz, SLAB_ALIGNMENT);
    util::poison_memory_region(slab, slab_sz);
    slabs.push_back(slab);
    cur = reinterpret_cast<uintptr_t>(slab) + size;
    end = reinterpret_cast<uintptr_t>(slab) + slab_sz;
    util::unpoison_memory_region(slab, size);
    return slab;
  }

private:
  void *allocate_mem(size_t size, std::align_val_t align) {
    return ::operator new(size, align, std::nothrow);
  }

  void deallocate_mem(void *ptr,
                      [[maybe_unused]] size_t size,
                      std::align_val_t align) {
#ifdef __cpp_sized_deallocation
    ::operator delete(ptr, size, align);
#else
    ::operator delete(ptr, align);
#endif
  }

  void deallocate_slabs(size_t skip = 0) {
    for (size_t i = skip; i < slabs.size(); ++i) {
      deallocate_mem(slabs[i], slab_size(i), SLAB_ALIGNMENT);
    }
  }

  void deallocate_large_slabs() {
    for (auto [slab, size] : large_slabs) {
      deallocate_mem(slab, size, SLAB_ALIGNMENT);
    }
    large_slabs.clear();
  }
};

template <typename T>
struct BumpAllocatorDeleter {
  constexpr BumpAllocatorDeleter() = default;
  void operator()(T *ptr) const { std::destroy_at(ptr); }
};

template <typename T>
using BumpAllocUniquePtr = std::unique_ptr<T, BumpAllocatorDeleter<T>>;

} // namespace tpde::util

template <size_t SlabSize>
void *operator new(size_t s, tpde::util::BumpAllocator<SlabSize> &a) {
  return a.allocate(s, std::min(s, alignof(std::max_align_t)));
}

template <size_t SlabSize>
void *operator new[](size_t s, tpde::util::BumpAllocator<SlabSize> &a) {
  return a.allocate(s, std::min(s, alignof(std::max_align_t)));
}

template <size_t SlabSize>
void operator delete(void *, tpde::util::BumpAllocator<SlabSize> &) {}

template <size_t SlabSize>
void operator delete[](void *, tpde::util::BumpAllocator<SlabSize> &) {}
