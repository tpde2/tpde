// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/util/AddressSanitizer.hpp"
#include "tpde/util/SmallVector.hpp"

#include <cstddef>
#include <memory>

namespace tpde::util {

/// A vector with stable element addresses.
template <typename T, size_t SegmentSize = 64>
class SegmentedVector {
public:
  using value_type = T;

private:
  T *cur = nullptr;
  T *end = nullptr;
  SmallVector<T *> segments;
  unsigned used_segments = 0;
  std::allocator<T> allocator;

public:
  SegmentedVector() = default;
  ~SegmentedVector() {
    clear();
    deallocate_segments();
  }

  SegmentedVector(const SegmentedVector &) = delete;

  SegmentedVector &operator=(const SegmentedVector &) = delete;

  /// Get the number of elements in the vector.
  size_t size() const { return used_segments * SegmentSize - (end - cur); }

  const T &operator[](size_t idx) const {
    assert(idx < size());
    return segments[idx / SegmentSize][idx % SegmentSize];
  }

  T &operator[](size_t idx) {
    assert(idx < size());
    return segments[idx / SegmentSize][idx % SegmentSize];
  }

  void clear() {
    if (used_segments == 0) {
      assert(cur == nullptr && end == nullptr && segments.empty());
      return;
    }
    for (unsigned i = 0; i < used_segments - 1; ++i) {
      std::destroy(segments[i], segments[i] + SegmentSize);
      util::poison_memory_region(segments[i], SegmentSize * sizeof(T));
    }
    T *last_start = segments[used_segments - 1];
    assert(last_start + SegmentSize == end);
    assert(last_start <= cur && cur <= end);
    std::destroy(last_start, cur);
    util::poison_memory_region(last_start, SegmentSize * sizeof(T));

    used_segments = 1;
    cur = segments[0];
    end = segments[0] + SegmentSize;
  }

  void push_back(const T &value) { std::construct_at(allocate(), value); }

  void push_back(T &&value) { std::construct_at(allocate(), std::move(value)); }

  template <typename... Args>
  T &emplace_back(Args &&...args) {
    return *std::construct_at(allocate(), std::forward<Args>(args)...);
  }

private:
  T *allocate() {
    if (cur == end) [[unlikely]] {
      next_segment();
    }
    util::unpoison_memory_region(cur, sizeof(T));
    return cur++;
  }

  void next_segment() {
    assert(cur == end);
    if (used_segments == segments.size()) {
      T *segment = allocator.allocate(SegmentSize);
      util::poison_memory_region(segment, SegmentSize * sizeof(T));
      segments.push_back(segment);
    }
    cur = segments[used_segments++];
    end = cur + SegmentSize;
  }

  void deallocate_segments() {
    for (T *segment : segments) {
      allocator.deallocate(segment, SegmentSize);
    }
  }
};

} // namespace tpde::util
