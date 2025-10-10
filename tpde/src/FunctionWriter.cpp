// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/FunctionWriter.hpp"

namespace tpde {

void FunctionWriterBase::more_space(size_t size) noexcept {
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

} // end namespace tpde
