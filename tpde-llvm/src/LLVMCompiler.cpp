// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde-llvm/LLVMCompiler.hpp"

#include <llvm/TargetParser/Triple.h>
#include <memory>

#include "arm64/LLVMCompilerArm64.hpp"
#include "x64/LLVMCompilerX64.hpp"

namespace tpde_llvm {

LLVMCompiler::~LLVMCompiler() = default;

std::unique_ptr<LLVMCompiler> LLVMCompiler::create(const llvm::Triple &triple) {
  switch (triple.getArch()) {
  case llvm::Triple::x86_64: return x64::create_compiler(triple);
  case llvm::Triple::aarch64: return arm64::create_compiler(triple);
  default: return nullptr;
  }
}

} // namespace tpde_llvm
