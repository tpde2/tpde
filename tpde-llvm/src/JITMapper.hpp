// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/AssemblerElf.hpp"
#include "tpde/ElfMapper.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/GlobalValue.h>

namespace tpde_llvm {

class JITMapperImpl {
  using GlobalMap = llvm::DenseMap<const llvm::GlobalValue *, tpde::SymRef>;

  tpde::elf::ElfMapper mapper;

  GlobalMap globals;

public:
  JITMapperImpl(GlobalMap &&globals) : globals(std::move(globals)) {}

  /// Map the ELF from the assembler into memory, returns true on success.
  bool map(tpde::elf::AssemblerElf &, tpde::elf::ElfMapper::SymbolResolver);

  void *lookup_global(llvm::GlobalValue *gv) const {
    return mapper.get_sym_addr(globals.lookup(gv));
  }

  std::pair<void *, size_t> get_mapped_range() const {
    return mapper.get_mapped_range();
  }
};

} // namespace tpde_llvm
