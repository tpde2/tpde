// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "JITMapper.hpp"

#include "tpde-llvm/LLVMCompiler.hpp"
#include "tpde/AssemblerElf.hpp"
#include "tpde/ElfMapper.hpp"

#include <llvm/Support/TimeProfiler.h>

namespace tpde_llvm {

bool JITMapperImpl::map(tpde::AssemblerElf &assembler,
                        tpde::ElfMapper::SymbolResolver resolver) noexcept {
  llvm::TimeTraceScope time_scope("TPDE_JITMap");
  return mapper.map(assembler, resolver);
}

JITMapper::JITMapper(std::unique_ptr<JITMapperImpl> impl) noexcept
    : impl(std::move(impl)) {}

JITMapper::~JITMapper() = default;

JITMapper::JITMapper(JITMapper &&other) noexcept = default;
JITMapper &JITMapper::operator=(JITMapper &&other) noexcept = default;

void *JITMapper::lookup_global(llvm::GlobalValue *gv) noexcept {
  return impl ? impl->lookup_global(gv) : nullptr;
}

} // namespace tpde_llvm
