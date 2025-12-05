// SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "tpde-llvm/LLVMCompiler.hpp"
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Plugins/PassPlugin.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/Support/raw_ostream.h>

namespace {

__attribute__((visibility("default"))) extern "C" ::llvm::PassPluginLibraryInfo
    llvmGetPassPluginInfo() {
  return llvm::PassPluginLibraryInfo{
      .APIVersion = LLVM_PLUGIN_API_VERSION,
      .PluginName = "TPDE-LLVM",
      .PluginVersion = "1",
      .PreCodeGenCallback = [](llvm::Module &mod,
                               llvm::TargetMachine &,
                               llvm::CodeGenFileType file_type,
                               llvm::raw_pwrite_stream &os) -> bool {
        auto &Ctx = mod.getContext();
        if (file_type != llvm::CodeGenFileType::ObjectFile) {
          Ctx.emitError("TPDE only support compiling to object files");
          return false;
        }

        llvm::TimeTraceScope time_scope("TPDE");
        auto compiler = tpde_llvm::LLVMCompiler::create(mod.getTargetTriple());
        std::vector<uint8_t> buf;
        if (compiler && compiler->compile_to_elf(mod, buf)) {
          os.write(reinterpret_cast<char *>(buf.data()), buf.size());
          return true;
        }

        Ctx.diagnose(llvm::DiagnosticInfoGeneric{"TPDE compilation failed",
                                                 llvm::DS_Warning});
        return false;
      },
  };
}

} // end anonymous namespace
