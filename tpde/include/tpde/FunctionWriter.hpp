// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/Assembler.hpp"
#include "tpde/base.hpp"
#include <cstring>

namespace tpde {

/// Helper class to write function text.
template <typename Derived>
class FunctionWriter {
protected:
  DataSection *section = nullptr;
  u8 *data_begin = nullptr;
  u8 *data_cur = nullptr;
  u8 *data_reserve_end = nullptr;

public:
  /// Growth size for more_space; adjusted exponentially after every grow.
  u32 growth_size = 0x10000;

  FunctionWriter() noexcept = default;

  ~FunctionWriter() {
    assert(data_cur == data_reserve_end &&
           "must flush section writer before destructing");
  }

protected:
  Derived *derived() noexcept { return static_cast<Derived *>(this); }

public:
  /// Get the SecRef of the current section.
  SecRef get_sec_ref() const noexcept { return get_section().get_ref(); }

  /// Get the current section.
  DataSection &get_section() const noexcept {
    assert(section != nullptr);
    return *section;
  }

  /// Switch section writer to new section; must be flushed.
  void switch_section(DataSection &new_section) noexcept {
    assert(data_cur == data_reserve_end &&
           "must flush section writer before switching sections");
    section = &new_section;
    data_begin = section->data.data();
    data_cur = data_begin + section->data.size();
    data_reserve_end = data_cur;
  }

  /// Get the current offset into the section.
  size_t offset() const noexcept { return data_cur - data_begin; }

  /// Get the current allocated size of the section.
  size_t allocated_size() const noexcept {
    return data_reserve_end - data_begin;
  }

  /// Pointer to beginning of section data.
  u8 *begin_ptr() noexcept { return data_begin; }

  /// Modifiable pointer to current writing position of the section. Must not
  /// be moved beyond the allocated region.
  u8 *&cur_ptr() noexcept { return data_cur; }

  void ensure_space(size_t size) noexcept {
    assert(data_reserve_end >= data_cur);
    if (size_t(data_reserve_end - data_cur) < size) [[unlikely]] {
      derived()->more_space(size);
    }
  }

  void more_space(size_t size) noexcept;

  template <std::integral T>
  void write_unchecked(T t) noexcept {
    assert(size_t(data_reserve_end - data_cur) >= sizeof(T));
    std::memcpy(data_cur, &t, sizeof(T));
    data_cur += sizeof(T);
  }

  template <std::integral T>
  void write(T t) noexcept {
    ensure_space(sizeof(T));
    write_unchecked<T>(t);
  }

  void flush() noexcept {
    if (data_cur != data_reserve_end) {
      section->data.resize(offset());
      data_reserve_end = data_cur;
#ifndef NDEBUG
      section->locked = false;
#endif
    }
  }

  void align(size_t align) noexcept {
    assert(align > 0 && (align & (align - 1)) == 0);
    ensure_space(align);
    // permit optimization when align is a constant.
    std::memset(cur_ptr(), 0, align);
    data_cur = data_begin + util::align_up(offset(), align);
    section->align = std::max(section->align, u32(align));
  }
};

template <typename Derived>
void FunctionWriter<Derived>::more_space(size_t size) noexcept {
  size_t cur_size = section->data.size();
  size_t new_size;
  if (cur_size + size <= section->data.capacity()) {
    new_size = section->data.capacity();
  } else {
    new_size = cur_size + (size <= growth_size ? growth_size : size);

    // Grow by factor 1.5
    growth_size = growth_size + (growth_size >> 1);
    // Max 16 MiB per grow.
    growth_size = growth_size < 0x1000000 ? growth_size : 0x1000000;
  }

  const size_t off = offset();
  section->data.resize_uninitialized(new_size);
#ifndef NDEBUG
  thread_local uint8_t rand = 1;
  std::memset(section->data.data() + off, rand += 2, new_size - off);
  section->locked = true;
#endif

  data_begin = section->data.data();
  data_cur = data_begin + off;
  data_reserve_end = data_begin + section->data.size();
}

} // namespace tpde
