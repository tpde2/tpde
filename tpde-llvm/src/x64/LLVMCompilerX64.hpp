// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <memory>

#include "tpde-llvm/LLVMCompiler.hpp"

namespace llvm {
class Triple;
} // namespace llvm

namespace tpde_llvm::x64 {

std::unique_ptr<LLVMCompiler> create_compiler(const llvm::Triple &);

} // namespace tpde_llvm::x64
