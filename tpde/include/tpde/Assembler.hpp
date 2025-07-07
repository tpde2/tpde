// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/base.hpp"
#include "tpde/util/SmallVector.hpp"
#include <cstring>

namespace tpde {

struct SymRef {
private:
  u32 val;

public:
  /// Invalid symbol reference
  constexpr SymRef() : val(0) {}

  explicit constexpr SymRef(u32 id) : val(id) {}

  u32 id() const { return val; }
  bool valid() const { return val != 0; }

  bool operator==(const SymRef &other) const { return other.val == val; }
};

struct SecRef {
private:
  u32 val;

public:
  /// Invalid symbol reference
  constexpr SecRef() : val(0) {}

  explicit constexpr SecRef(u32 id) : val(id) {}

  u32 id() const { return val; }
  bool valid() const { return val != 0; }

  bool operator==(const SecRef &other) const { return other.val == val; }
};

struct DataSection {
  /// 256 bytes inline storage is enough for 10 relocations, which is a typical
  /// number for a single function (relevant for COMDAT sections with one
  /// section per function).
  using StorageTy = util::SmallVector<u8, 256>;

  /// Section data.
  StorageTy data;

  u64 addr = 0;  ///< Address (file-format-specific).
  u64 vsize = 0; ///< Size of virtual section, otherwise data.size() is valid.
  u32 type = 0;  ///< Type (file-format-specific).
  u32 flags = 0; ///< Flags (file-format-specific).
  u32 name = 0;  ///< Name (file-format-specific, can also be index, etc.).
  u32 align = 1; ///< Alignment (bytes).

  /// Section symbol, or signature symbol for SHT_GROUP sections.
  SymRef sym;

private:
  SecRef sec_ref;

public:
  /// Generic field for target-specific data.
  void *target_info = nullptr;

  /// Whether the section is virtual, i.e. has no content.
  bool is_virtual;

#ifndef NDEBUG
  /// Whether the section is currently in use by a SectionWriter.
  bool locked = false;
#endif

  DataSection(SecRef ref) noexcept : sec_ref(ref) {}

  SecRef get_ref() const noexcept { return sec_ref; }

  size_t size() const { return is_virtual ? vsize : data.size(); }

  template <typename T>
  void write(const T &t) noexcept {
    assert(!locked);
    assert(!is_virtual);
    size_t off = data.size();
    data.resize_uninitialized(data.size() + sizeof(T));
    std::memcpy(data.data() + off, &t, sizeof(T));
  }
};

} // namespace tpde

#undef ARG
