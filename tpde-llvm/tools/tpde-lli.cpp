// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/JITLink/EHFrameSupport.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#if LLVM_VERSION_MAJOR >= 20
  #include <llvm/ExecutionEngine/Orc/EHFrameRegistrationPlugin.h>
#endif
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutorProcessControl.h>
#include <llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#if LLVM_VERSION_MAJOR >= 21
  #include <llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h>
#endif
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include "tpde-llvm/LLVMCompiler.hpp"

#include <dlfcn.h>
#include <iostream>
#include <memory>

#ifdef TPDE_LOGGING
  #include <spdlog/spdlog.h>
#endif

#define ARGS_NOEXCEPT
#include <args/args.hxx>

static llvm::ExitOnError exit_on_err;

int main(int argc, char *argv[]) {
  args::ArgumentParser parser("TPDE LLI");
  args::HelpFlag help(parser, "help", "Display help", {'h', "help"});

  args::ValueFlag<unsigned> log_level(
      parser,
      "log_level",
      "Set the log level to 0=NONE, 1=ERR, 2=WARN(default), 3=INFO, 4=DEBUG, "
      ">5=TRACE",
      {'l', "log-level"},
      2);

  args::Flag orc(parser, "orc", "Use LLVM ORC", {"orc"});

  args::Positional<std::string> ir_path(
      parser, "ir_path", "Path to the input IR file", "-");

  parser.ParseCLI(argc, argv);
  if (parser.GetError() == args::Error::Help) {
    std::cout << parser;
    return 0;
  }

  if (parser.GetError() != args::Error::None) {
    std::cerr << "Error parsing arguments: " << parser.GetErrorMsg() << '\n';
    return 1;
  }

#ifdef TPDE_LOGGING
  {
    spdlog::level::level_enum level = spdlog::level::off;
    switch (log_level.Get()) {
    case 0: level = spdlog::level::off; break;
    case 1: level = spdlog::level::err; break;
    case 2: level = spdlog::level::warn; break;
    case 3: level = spdlog::level::info; break;
    case 4: level = spdlog::level::debug; break;
    default:
      assert(level >= 5);
      level = spdlog::level::trace;
      break;
    }

    spdlog::set_level(level);
  }
#endif

  llvm::LLVMContext context;
  llvm::SMDiagnostic diag{};

  auto mod = llvm::parseIRFile(ir_path.Get(), diag, context);
  if (!mod) {
    diag.print(argv[0], llvm::errs());
    return 1;
  }

  llvm::Function *main_fn = mod->getFunction("main");
  if (!main_fn) {
    std::cerr << "module has no main function\n";
    return 1;
  }

  std::string triple_str = llvm::sys::getProcessTriple();
  llvm::Triple triple(triple_str);
  auto compiler = tpde_llvm::LLVMCompiler::create(triple);
  if (!compiler) {
    std::cerr << "Unknown architecture: " << triple_str << "\n";
    return 1;
  }

  if (!orc) {
    auto mapper = compiler->compile_and_map(*mod, [](std::string_view name) {
      return ::dlsym(RTLD_DEFAULT, std::string(name).c_str());
    });
    void *main_addr = mapper.lookup_global(main_fn);
    if (!main_addr) {
      std::cerr << "JIT compilation failed\n";
      return 1;
    }
    return ((int (*)(int, char **))main_addr)(0, nullptr);
  }

  std::vector<uint8_t> buf;
  if (!compiler->compile_to_elf(*mod, buf)) {
    std::cerr << "Failed to compile\n";
    return 1;
  }
  llvm::StringRef buf_strref(reinterpret_cast<char *>(buf.data()), buf.size());
  auto obj_membuf = llvm::MemoryBuffer::getMemBuffer(buf_strref, "", false);
  assert(obj_membuf->getBufferSize());

  size_t page_size = getpagesize();
  llvm::orc::ExecutionSession es(
      exit_on_err(llvm::orc::SelfExecutorProcessControl::Create()));
  llvm::orc::MapperJITLinkMemoryManager memory_manager(
      page_size, std::make_unique<llvm::orc::InProcessMemoryMapper>(page_size));
  llvm::orc::ObjectLinkingLayer object_layer(es, memory_manager);
  llvm::orc::JITDylib &dylib = exit_on_err(es.createJITDylib("<main>"));

  // TODO: correct global prefix for MachO/COFF platforms
  dylib.addGenerator(exit_on_err(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          /*GlobalPrefix=*/'\0')));

  exit_on_err(object_layer.add(dylib, std::move(obj_membuf)));

#if LLVM_VERSION_MAJOR >= 21
  object_layer.addPlugin(
      exit_on_err(llvm::orc::EHFrameRegistrationPlugin::Create(es)));
#else
  object_layer.addPlugin(std::make_unique<llvm::orc::EHFrameRegistrationPlugin>(
      es, std::make_unique<llvm::jitlink::InProcessEHFrameRegistrar>()));
#endif

  llvm::orc::ExecutorSymbolDef sym = exit_on_err(es.lookup(&dylib, "main"));
  uintptr_t main_addr = static_cast<uintptr_t>(sym.getAddress().getValue());
  int ret = ((int (*)(int, char **))main_addr)(0, nullptr);

  exit_on_err(es.endSession());
  return ret;
}
