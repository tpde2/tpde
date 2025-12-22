// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <optional>

#include "SmallVector.hpp"
#include "misc.hpp"
#include "tpde/base.hpp"

namespace tpde::util {
/// BitSet implemented on top of SmallVector
template <u32 InternalCapacity>
struct SmallBitSet {
  // we divide by 64 bits for the SmallVector
  static_assert((InternalCapacity % 64) == 0);

  SmallVector<u64, InternalCapacity / 64> data;
  u32 bit_size = 0;

  SmallBitSet() = default;

  SmallBitSet(const SmallBitSet &other)
      : data(other.data), bit_size(other.bit_size) {}

  SmallBitSet(SmallBitSet &&other) {
    data = std::move(other.data);
    bit_size = other.bit_size;
  }

  SmallBitSet &operator=(const SmallBitSet &other) {
    if (&other != this) {
      data = other.data;
      bit_size = other.bit_size;
    }
    return *this;
  }

  SmallBitSet &operator=(SmallBitSet &&other) {
    if (&other != this) {
      data = std::move(data);
      bit_size = other.bit_size;
    }
    return *this;
  }

  void clear() {
    data.clear();
    bit_size = 0;
  }

  void resize(const u32 bit_size) {
    const auto byte_size = align_up(bit_size, 64) / 64;
    if (bit_size > this->bit_size && this->bit_size != 0) {
      const auto last_bits = this->bit_size & 63;
      const auto set_mask = (1ull << last_bits) - 1;
      data.back() = data.back() & set_mask;
    }

    data.resize(byte_size);
    this->bit_size = bit_size;
  }

  void zero() {
    for (auto &v : data) {
      v = 0;
    }
  }

  void push_back(const bool val) {
    if (bit_size / 64 != (bit_size - 1) / 64) {
      data.push_back(0);
      if (val) {
        mark_set(bit_size);
      }
      ++bit_size;
      return;
    }
    if (val) {
      mark_set(bit_size++);
    } else {
      mark_unset(bit_size++);
    }
  }

  [[nodiscard]] bool is_set(const u32 idx) const {
    const u32 elem_idx = idx >> 6;
    const u64 bit = 1ull << (idx & 63);
    return data[elem_idx] & bit;
  }

  void mark_set(const u32 idx) {
    const u32 elem_idx = idx >> 6;
    const u64 bit = 1ull << (idx & 63);
    data[elem_idx] |= bit;
  }

  void mark_unset(const u32 idx) {
    const u32 elem_idx = idx >> 6;
    const u64 bit = 1ull << (idx & 63);
    data[elem_idx] &= ~bit;
  }

  std::optional<u32> first_set() {
    for (u32 i = 0; i < data.size(); ++i) {
      if (data[i] == 0) {
        continue;
      }
      return i + util::cnt_tz(data[i]);
    }
    return {};
  }
};
} // namespace tpde::util
