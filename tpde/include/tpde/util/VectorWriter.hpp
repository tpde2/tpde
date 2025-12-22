// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstring>

#include "tpde/base.hpp"
#include "tpde/util/SmallVector.hpp"
#include "tpde/util/misc.hpp"

namespace tpde::util {

class VectorWriter {
  SmallVectorBase<u8> *vector = nullptr;
  u8 *cur;
  u8 *end;

public:
  VectorWriter() = default;
  VectorWriter(SmallVectorBase<u8> &vector)
      : vector(&vector),
        cur(vector.data() + vector.size()),
        end(vector.data() + vector.size()) {}
  VectorWriter(SmallVectorBase<u8> &vector, size_t off)
      : vector(&vector),
        cur(vector.data() + off),
        end(vector.data() + vector.size()) {}
  ~VectorWriter() { flush(); }

  VectorWriter(const VectorWriter &) = delete;
  VectorWriter(VectorWriter &&other) { *this = std::move(other); }

  VectorWriter &operator=(const VectorWriter &) = delete;
  VectorWriter &operator=(VectorWriter &&other) {
    flush();
    vector = other.vector;
    cur = other.cur;
    end = other.end;
    other.vector = nullptr;
    return *this;
  }

  void reserve(size_t extra) {
    assert(end >= cur);
    if (size_t(end - cur) < extra) [[unlikely]] {
      size_t off = size();
      if (vector->capacity() < off + extra) {
        // This will grow exponentially.
        vector->reserve(off + extra);
      }
      vector->resize_uninitialized(vector->capacity());
      cur = vector->data() + off;
      end = vector->data() + vector->size();
    }
  }

  void flush() {
    if (vector && cur != end) {
      vector->resize(size());
      end = cur;
    }
  }

  size_t size() const { return cur - vector->data(); }
  size_t capacity() const { return vector->size(); }

  u8 *data() { return vector->data(); }

  void skip_unchecked(size_t size) {
    assert(size_t(end - cur) >= size);
    cur += size;
  }

  void skip(size_t size) {
    reserve(size);
    skip_unchecked(size);
  }

  void unskip(size_t n) {
    assert(n <= size());
    cur -= n;
  }

  void zero_unchecked(size_t n) {
    assert(size_t(end - cur) >= n);
    TPDE_NOALIAS(this, cur);
    std::memset(cur, 0, n);
    cur += n;
  }

  void zero(size_t n) {
    reserve(n);
    zero_unchecked(n);
  }

  void write_unchecked(std::span<const u8> data) {
    assert(size_t(end - cur) >= data.size());
    TPDE_NOALIAS(this, cur);
    std::memcpy(cur, data.data(), data.size());
    cur += data.size();
  }

  void write(std::span<const u8> data) {
    reserve(data.size());
    write_unchecked(data);
  }

  template <std::integral T>
  void write_unchecked(T t) {
    assert(size_t(end - cur) >= sizeof(T));
    TPDE_NOALIAS(this, cur);
    std::memcpy(cur, &t, sizeof(T));
    cur += sizeof(T);
  }

  template <std::integral T>
  void write(T t) {
    reserve(sizeof(T));
    write_unchecked<T>(t);
  }

  void write_uleb_unchecked(uint64_t value) {
    assert(size_t(end - cur) >= uleb_len(value));
    cur += uleb_write(cur, value);
  }

  void write_uleb(uint64_t value) {
    reserve(10);
    write_uleb_unchecked(value);
  }

  void write_sleb_unchecked(int64_t value) {
    // TODO: implement and use sleb_len
    assert(size_t(end - cur) >= 10);
    cur += sleb_write(cur, value);
  }

  void write_sleb(int64_t value) {
    reserve(10);
    write_sleb_unchecked(value);
  }
};

} // end namespace tpde::util
