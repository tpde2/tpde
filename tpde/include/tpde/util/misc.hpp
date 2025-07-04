// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <type_traits>

#include "tpde/base.hpp"

namespace tpde::util {

template <typename T>
constexpr T align_down(T val, std::type_identity_t<T> align) {
  // align must be a power of two
  assert(align > 0 && (align & (align - 1)) == 0);
  return val & ~(align - 1);
}

template <typename T>
constexpr T align_up(T val, std::type_identity_t<T> align) {
  // align must be a power of two
  assert(align > 0 && (align & (align - 1)) == 0);
  return (val + (align - 1)) & ~(align - 1);
}

template <typename T>
T cnt_tz(T) {
  static_assert(false);
}

template <>
constexpr inline u8 cnt_tz<u8>(const u8 val) {
  return __builtin_ctz(val);
}

template <>
constexpr inline u16 cnt_tz<u16>(const u16 val) {
  return __builtin_ctz(val);
}

template <>
constexpr inline u32 cnt_tz<u32>(const u32 val) {
  return __builtin_ctz(val);
}

template <>
constexpr inline u64 cnt_tz<u64>(const u64 val) {
  return __builtin_ctzll(val);
}

template <typename T>
T cnt_lz(T) {
  static_assert(false);
}

template <>
constexpr u8 cnt_lz<u8>(const u8 val) {
  return __builtin_clz((u32)val) - 24;
}

template <>
constexpr u16 cnt_lz<u16>(const u16 val) {
  return __builtin_clz((u32)val) - 16;
}

template <>
constexpr u32 cnt_lz<u32>(const u32 val) {
  return __builtin_clz(val);
}

template <>
constexpr u64 cnt_lz<u64>(const u64 val) {
  return __builtin_clzll(val);
}

inline u64 zext(const u64 val, const unsigned bits) {
  assert(bits > 0 && bits < 64 && "invalid sext bit width");
  return val & ((u64{1} << bits) - 1);
}

inline i64 sext(const u64 val, const unsigned bits) {
  assert(bits > 0 && bits < 64 && "invalid sext bit width");
  return (i64)(val << (64 - bits)) >> (64 - bits);
}

constexpr unsigned uleb_len(u64 value) {
  return value ? (64 - cnt_lz(value) + 6) / 7 : 1;
}

constexpr unsigned uleb_write(u8 *dst, u64 value) noexcept {
  u8 *base = dst;
  while (true) {
    u8 write = value & 0b0111'1111;
    value >>= 7;
    if (value == 0) {
      *dst++ = write;
      return dst - base;
    }
    *dst++ = write | 0b1000'0000;
  }
}

constexpr unsigned sleb_write(u8 *dst, i64 value) noexcept {
  u8 *base = dst;
  while (true) {
    u8 write = value & 0b0111'1111;
    value >>= 7;
    if ((value == 0 && (value & 0x40) == 0) || (value == -1 && value & 0x40)) {
      *dst++ = write;
      return dst - base;
    }
    *dst++ = write | 0b1000'0000;
  }
}

template <bool Reverse = false>
struct BitSetIterator {
  u64 set;

  struct Iter {
    u64 set;

    Iter &operator++() noexcept {
      if constexpr (Reverse) {
        set = set & ~(1ull << (63 - cnt_lz(set)));
      } else {
        set = set & (set - 1);
      }
      return *this;
    }

    [[nodiscard]] u64 operator*() const noexcept {
      assert(set != 0);
      if constexpr (Reverse) {
        return 63 - cnt_lz(set);
      } else {
        return cnt_tz(set);
      }
    }

    [[nodiscard]] bool operator!=(const Iter &rhs) const noexcept {
      return rhs.set != set;
    }
  };

  explicit BitSetIterator(const u64 set) : set(set) {}

  [[nodiscard]] Iter begin() const noexcept { return Iter{.set = set}; }

  [[nodiscard]] Iter end() const noexcept { return Iter{.set = 0}; }
};

} // namespace tpde::util
