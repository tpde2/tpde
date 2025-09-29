// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde-llvm/OrcCompiler.hpp"

#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/TargetParser/Triple.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "tpde-llvm/LLVMCompiler.hpp"

namespace tpde_llvm {

namespace {

class VectorMemoryBuffer : public llvm::MemoryBuffer {
  std::vector<uint8_t> data;

public:
  VectorMemoryBuffer(std::vector<uint8_t> &&v) : data(std::move(v)) {
    const char *ptr = reinterpret_cast<const char *>(data.data());
    init(ptr, ptr + data.size(), /*RequiresNullTerminator=*/false);
  }

  llvm::MemoryBuffer::BufferKind getBufferKind() const override {
    return llvm::MemoryBuffer::MemoryBuffer_Malloc;
  }
};

} // anonymous namespace

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
    OrcCompiler::operator()(llvm::Module &mod) {
  std::vector<uint8_t> buf;
  if (compiler && compiler->compile_to_elf(mod, buf)) {
    return std::make_unique<VectorMemoryBuffer>(std::move(buf));
  }
  if (tm) {
    return llvm::orc::SimpleCompiler(*tm)(mod);
  }
  return llvm::createStringError("TPDE compilation failed");
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
    ConcurrentOrcCompiler::operator()(llvm::Module &mod) {
  std::vector<uint8_t> buf;
  auto compiler = LLVMCompiler::create(jtmb.getTargetTriple());
  if (compiler && compiler->compile_to_elf(mod, buf)) {
    return std::make_unique<VectorMemoryBuffer>(std::move(buf));
  }
  auto tm = llvm::cantFail(jtmb.createTargetMachine());
  return llvm::orc::SimpleCompiler(*tm)(mod);
}

} // namespace tpde_llvm
