// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/StringTable.hpp"

#include <cstring>

namespace tpde {

size_t StringTable::add(std::string_view str) {
  if (str.empty()) {
    return 0;
  }

  // TODO: use hash table for deduplication
  size_t off = strtab.size();
  strtab.resize_uninitialized(strtab.size() + str.size() + 1);
  strtab[strtab.size() - 1] = '\0';
  std::memcpy(strtab.data() + off, str.data(), str.size());
  return off;
}

size_t StringTable::add_prefix(std::string_view prefix, std::string_view str) {
  // TODO: use hash table for deduplication
  size_t off = strtab.size();
  strtab.resize_uninitialized(strtab.size() + prefix.size() + str.size() + 1);
  strtab[strtab.size() - 1] = '\0';
  std::memcpy(strtab.data() + off, prefix.data(), prefix.size());
  std::memcpy(strtab.data() + off + prefix.size(), str.data(), str.size());
  return off;
}

} // end namespace tpde
