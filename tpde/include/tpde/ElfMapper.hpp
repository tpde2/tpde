// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <string_view>

#include "base.hpp"
#include "tpde/AssemblerElf.hpp"
#include "tpde/util/SmallVector.hpp"
#include "tpde/util/function_ref.hpp"

namespace tpde {

class ElfMapper {
public:
  // TODO: use C++26 std::function_ref
  using SymbolResolver = util::function_ref<void *(std::string_view)>;

private:
  u8 *mapped_addr;
  size_t mapped_size;
  u32 registered_frame_off = 0;

  u32 local_sym_count = 0;
  util::SmallVector<void *, 64> sym_addrs;

public:
  ElfMapper() noexcept = default;
  ~ElfMapper() { reset(); }

  ElfMapper(const ElfMapper &) = delete;
  ElfMapper(ElfMapper &&) = delete;

  ElfMapper &operator=(const ElfMapper &) = delete;
  ElfMapper &operator=(ElfMapper &&) = delete;

  void reset() noexcept;

  bool map(AssemblerElf &assembler, SymbolResolver resolver) noexcept;

  void *get_sym_addr(SymRef sym) noexcept;
};

} // namespace tpde
