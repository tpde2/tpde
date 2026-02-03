// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "JITMapper.hpp"

#include "tpde-llvm/LLVMCompiler.hpp"
#include "tpde/AssemblerElf.hpp"
#include "tpde/ElfMapper.hpp"

#include <llvm/Support/TimeProfiler.h>

namespace tpde_llvm {

bool JITMapperImpl::map(tpde::elf::AssemblerElf &assembler,
                        tpde::elf::ElfMapper::SymbolResolver resolver) {
  llvm::TimeTraceScope time_scope("TPDE_JITMap");
  return mapper.map(assembler, resolver);
}

JITMapper::JITMapper(std::unique_ptr<JITMapperImpl> impl)
    : impl(std::move(impl)) {}

JITMapper::~JITMapper() = default;

JITMapper::JITMapper(JITMapper &&other) = default;
JITMapper &JITMapper::operator=(JITMapper &&other) = default;

void *JITMapper::lookup_global(llvm::GlobalValue *gv) const {
  return impl ? impl->lookup_global(gv) : nullptr;
}

std::pair<void *, size_t> JITMapper::get_mapped_range() const {
  return impl ? impl->get_mapped_range() : std::make_pair(nullptr, 0ull);
}

} // namespace tpde_llvm
