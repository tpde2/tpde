// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/util/AddressSanitizer.hpp"

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace tpde::util {

class SmallVectorUntypedBase {
public:
  using size_type = std::size_t;
  using difference_type = ptrdiff_t;

protected:
  void *ptr;
  size_type sz;
  size_type cap;

  SmallVectorUntypedBase() = delete;
  SmallVectorUntypedBase(size_type cap);

  void *small_ptr(this auto &&self) {
    static_assert(!std::same_as<decltype(self), SmallVectorUntypedBase>);
    return static_cast<void *>(self.get_small_elements());
  };

  bool is_small() const;

  void *get_small_elements() const;

  void *grow_malloc(size_type min_size,
                    size_type elem_sz,
                    size_type &new_cap,
                    size_type align);
  void grow_trivial(size_type min_size, size_type elem_sz, size_type align);

public:
  size_type size() const { return sz; }
  size_type capacity() const { return cap; }
  bool empty() const { return size() == 0; }
};

template <typename T>
class SmallVectorBase : public SmallVectorUntypedBase {
  // TODO: support types with larger alignment
  static_assert(alignof(T) <= alignof(SmallVectorUntypedBase),
                "SmallVector only supports types with pointer-sized alignment");

public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using iterator = T *;
  using const_iterator = const T *;

  static constexpr bool IsTrivial = std::is_trivially_copy_constructible_v<T> &&
                                    std::is_trivially_move_constructible_v<T> &&
                                    std::is_trivially_destructible_v<T>;

protected:
  SmallVectorBase(size_type cap) : SmallVectorUntypedBase(cap) {
    poison_memory_region(ptr, cap * sizeof(T));
  }

  ~SmallVectorBase() {
    std::destroy(begin(), end());
    if (!is_small()) {
      free(ptr);
    }
  }

  SmallVectorBase &operator=(SmallVectorBase &&other) {
    clear();
    if (!other.is_small()) {
      ptr = other.ptr;
      sz = other.sz;
      cap = other.cap;
      other.ptr = other.small_ptr();
      other.sz = 0;
      other.cap = 0;
    } else {
      reserve(other.size());
      unpoison_memory_region(begin(), other.size() * sizeof(T));
      std::uninitialized_move(other.begin(), other.end(), begin());
      sz = other.size();
      other.clear();
    }
    return *this;
  }

public:
  pointer data() { return reinterpret_cast<T *>(ptr); }
  const_pointer data() const { return reinterpret_cast<const T *>(ptr); }

  iterator begin() { return data(); }
  iterator end() { return data() + sz; }
  const_iterator begin() const { return data(); }
  const_iterator end() const { return data() + sz; }
  const_iterator cbegin() const { return data(); }
  const_iterator cend() const { return data() + sz; }

  /// Maximum number of elements the vector can hold.
  static size_type max_size() { return size_type(-1) / sizeof(T); }

  reference operator[](size_type idx) {
    assert(idx < size());
    return data()[idx];
  }
  const_reference operator[](size_type idx) const {
    assert(idx < size());
    return data()[idx];
  }

  reference front() { return (*this)[0]; }
  const_reference front() const { return (*this)[0]; }
  reference back() { return (*this)[size() - 1]; }
  const_reference back() const { return (*this)[size() - 1]; }

private:
  void ensure_space(size_type num_elems) {
    assert(num_elems <= max_size() - size());
    if (size() + num_elems > capacity()) [[unlikely]] {
      grow(size() + num_elems);
    }
  }

  void grow(size_type new_size)
    requires IsTrivial
  {
    grow_trivial(new_size, sizeof(T), alignof(T));
  }

  void grow(size_type new_size)
    requires(!IsTrivial);

public:
  /// Append an element. elem must not be a reference inside to the vector.`
  void push_back(const T &elem) {
    ensure_space(1);
    unpoison_memory_region(end(), sizeof(T));
    ::new (reinterpret_cast<void *>(end())) T(elem);
    sz += 1;
  }

  /// Append an element. elem must not be a reference inside to the vector.`
  void push_back(T &&elem) {
    ensure_space(1);
    unpoison_memory_region(end(), sizeof(T));
    ::new (reinterpret_cast<void *>(end())) T(::std::move(elem));
    sz += 1;
  }

  template <typename... ArgT>
  reference emplace_back(ArgT &&...args) {
    ensure_space(1);
    unpoison_memory_region(end(), sizeof(T));
    ::new (reinterpret_cast<void *>(end())) T(::std::forward<ArgT>(args)...);
    sz += 1;
    return back();
  }

  void pop_back() {
    back().~T();
    poison_memory_region(end(), sizeof(T));
    sz -= 1;
  }

  void reserve(size_type new_cap) {
    if (cap < new_cap) {
      grow(new_cap);
    }
  }

private:
  template <bool Initialize>
  void resize(size_type new_size) {
    if (sz > new_size) {
      std::destroy(begin() + new_size, end());
      poison_memory_region(begin() + new_size, (sz - new_size) * sizeof(T));
      sz = new_size;
    } else if (sz < new_size) {
      reserve(new_size);
      unpoison_memory_region(end(), (new_size - sz) * sizeof(T));
      for (pointer it = end(), e = begin() + new_size; it != e; ++it) {
        Initialize ? ::new (it) T() : ::new (it) T;
      }
      sz = new_size;
    }
  }

public:
  void resize(size_type new_size) { resize<true>(new_size); }
  void resize_uninitialized(size_type new_size) { resize<false>(new_size); }

  void resize(size_type new_size, const T &init) {
    if (sz > new_size) {
      std::destroy(begin() + new_size, end());
      poison_memory_region(begin() + new_size, (sz - new_size) * sizeof(T));
      sz = new_size;
    } else if (sz < new_size) {
      reserve(new_size);
      unpoison_memory_region(end(), (new_size - sz) * sizeof(T));
      std::uninitialized_fill(begin() + sz, begin() + new_size, init);
      sz = new_size;
    }
  }

  iterator erase(iterator start_it, iterator end_it) {
    iterator res = start_it;
    while (end_it != end()) {
      *start_it++ = std::move(*end_it++);
    }
    std::destroy(start_it, end_it);
    sz = start_it - begin();
    return res;
  }

  template <typename It>
  void append(It start_it, It end_it) {
    size_type n = std::distance(start_it, end_it);
    reserve(sz + n);
    unpoison_memory_region(end(), n * sizeof(T));
    std::uninitialized_copy(start_it, end_it, end());
    sz += n;
  }

  void clear() {
    std::destroy(begin(), end());
    poison_memory_region(begin(), sz * sizeof(T));
    sz = 0;
  }
};

template <class T>
void SmallVectorBase<T>::grow(size_type new_size)
  requires(!IsTrivial)
{
  size_type new_cap;
  T *new_alloc =
      static_cast<T *>(grow_malloc(new_size, sizeof(T), new_cap, alignof(T)));
  std::uninitialized_move(begin(), end(), new_alloc);
  poison_memory_region(new_alloc + sz, (new_cap - sz) * sizeof(T));
  std::destroy(begin(), end());
  if (!is_small()) {
    free(ptr);
  } else {
    poison_memory_region(data(), cap * sizeof(T));
  }
  ptr = new_alloc;
  cap = new_cap;
}

template <typename T>
constexpr size_t SmallVectorDefaultSize = sizeof(T) < 256 ? 256 / sizeof(T) : 1;

template <class T, size_t N = SmallVectorDefaultSize<T>>
class SmallVector : public SmallVectorBase<T> {
  alignas(T) char elements[N == 0 ? 1 : N * sizeof(T)];

  friend class SmallVectorUntypedBase;

protected:
  void *get_small_elements() const { return elements; }

public:
  SmallVector() : SmallVectorBase<T>(N) {}

  SmallVector(const SmallVector &) = delete;
  SmallVector &operator=(const SmallVector &) = delete;
  // TODO: Implement this if required.
#if 0
  SmallVector(SmallVector &other) : SmallVectorBase<T>(N) {
    SmallVectorBase<T>::operator=(other);
  }
  SmallVector(SmallVectorBase<T> &other) : SmallVectorBase<T>(N) {
    SmallVectorBase<T>::operator=(other);
  }
#endif

  SmallVector(SmallVector &&other) : SmallVectorBase<T>(N) {
    SmallVectorBase<T>::operator=(std::move(other));
  }
  SmallVector(SmallVectorBase<T> &&other) : SmallVectorBase<T>(N) {
    SmallVectorBase<T>::operator=(std::move(other));
  }

#if 0
  SmallVector &operator=(SmallVector &other) {
    SmallVectorBase<T>::operator=(other);
    return *this;
  }
  SmallVector &operator=(SmallVectorBase<T> &other) {
    SmallVectorBase<T>::operator=(other);
    return *this;
  }
#endif

  SmallVector &operator=(SmallVector &&other) {
    SmallVectorBase<T>::operator=(std::move(other));
    return *this;
  }
  SmallVector &operator=(SmallVectorBase<T> &&other) {
    SmallVectorBase<T>::operator=(std::move(other));
    return *this;
  }
};

} // end namespace tpde::util
