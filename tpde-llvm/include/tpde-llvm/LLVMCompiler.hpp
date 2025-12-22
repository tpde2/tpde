// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace llvm {
class GlobalValue;
class Module;
class Triple;
} // namespace llvm

namespace tpde_llvm {

class JITMapperImpl;

/// In-memory mapper for JIT execution. Memory and registered unwind info will
/// be released on destruction.
class JITMapper {
private:
  friend class JITMapperImpl;

  std::unique_ptr<JITMapperImpl> impl;

public:
  explicit JITMapper(std::unique_ptr<JITMapperImpl> impl);
  ~JITMapper();

  JITMapper(const JITMapper &) = delete;
  JITMapper(JITMapper &&other);

  JITMapper &operator=(const JITMapper &) = delete;
  JITMapper &operator=(JITMapper &&other);

  /// Get the address for a global, which must be contained in the compiled
  /// module.
  void *lookup_global(llvm::GlobalValue *);

  /// Indicate whether compilation and in-memory mapping was successful.
  operator bool() const { return impl != nullptr; }
};

/// Compiler for LLVM modules
class LLVMCompiler {
protected:
  LLVMCompiler() = default;

public:
  virtual ~LLVMCompiler();

  LLVMCompiler(const LLVMCompiler &) = delete;
  LLVMCompiler &operator=(const LLVMCompiler &) = delete;

  /// Create a compiler for the specified target triple; returns null if the
  /// triple is not supported. The only supported code model is small, the only
  /// supported relocation model is PIC.
  static std::unique_ptr<LLVMCompiler> create(const llvm::Triple &triple);

  /// Compile the module to an object file and emit it into the buffer. The
  /// module might be modified during compilation.
  /// \returns true on success.
  virtual bool compile_to_elf(llvm::Module &mod, std::vector<uint8_t> &buf) = 0;

  /// Compile the module and map it into memory, calling resolver to resolve
  /// references to external symbols. This function will also register unwind
  /// information. The module might be modified during compilation.
  virtual JITMapper
      compile_and_map(llvm::Module &mod,
                      std::function<void *(std::string_view)> resolver) = 0;
};

} // namespace tpde_llvm
