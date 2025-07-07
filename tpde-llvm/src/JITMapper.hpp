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

  tpde::ElfMapper mapper;

  GlobalMap globals;

public:
  JITMapperImpl(GlobalMap &&globals) : globals(std::move(globals)) {}

  /// Map the ELF from the assembler into memory, returns true on success.
  bool map(tpde::AssemblerElfBase &, tpde::ElfMapper::SymbolResolver) noexcept;

  void *lookup_global(llvm::GlobalValue *gv) noexcept {
    return mapper.get_sym_addr(globals.lookup(gv));
  }
};

} // namespace tpde_llvm
