// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>

#include "tpde-llvm/LLVMCompiler.hpp"

namespace tpde_llvm {

/// ORC Compiler functor using TPDE-LLVM, which can transparently fall back to
/// LLVM's SimpleCompiler (if a target machine is provided). Intended as a
/// typical drop-in replacement for llvm::orc::SimpleCompiler.
class OrcCompiler : public llvm::orc::IRCompileLayer::IRCompiler {
  std::unique_ptr<LLVMCompiler> owned_compiler;
  LLVMCompiler *compiler;
  llvm::TargetMachine *tm;

public:
  /// Constructor. If the TargetMachine non-null, a failure within TPDE
  /// (e.g., due to unsupported IR constructs) will fall back to LLVM.
  OrcCompiler(LLVMCompiler *compiler, llvm::TargetMachine *tm = nullptr)
      : IRCompiler({}), compiler(compiler), tm(tm) {}
  /// Constructor. If the TargetMachine non-null, a failure within TPDE
  /// (e.g., due to unsupported IR constructs) will fall back to LLVM.
  OrcCompiler(const llvm::Triple &triple, llvm::TargetMachine *tm = nullptr)
      : IRCompiler({}),
        owned_compiler(LLVMCompiler::create(triple)),
        compiler(owned_compiler.get()),
        tm(tm) {}
  /// Constructor, compatible with llvm::orc::SimpleCompiler.
  OrcCompiler(llvm::TargetMachine &tm)
      : IRCompiler({}),
        owned_compiler(LLVMCompiler::create(tm.getTargetTriple())),
        compiler(owned_compiler.get()),
        tm(&tm) {}

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
      operator()(llvm::Module &) override;
};

/// A very simple thread-safe version of OrcCompiler, intended as a typical
/// drop-in replacement for llvm::orc::ConcurrentIRCompiler. This is not the
/// most efficient way, applications could (and probably should) cache a
/// LLVMCompiler and TargetMachine in thread-local storage.
class ConcurrentOrcCompiler : public llvm::orc::IRCompileLayer::IRCompiler {
  llvm::orc::JITTargetMachineBuilder jtmb;

public:
  /// Constructor. For every compilation, a new LLVMCompiler is constructed
  /// based on the target triple of the JITTargetMachineBuilder. Likewise, for
  /// every fallback, a new TargetMachine is constructed.
  ConcurrentOrcCompiler(llvm::orc::JITTargetMachineBuilder jtmb)
      : IRCompiler({}), jtmb(std::move(jtmb)) {}

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
      operator()(llvm::Module &) override;
};

} // namespace tpde_llvm
