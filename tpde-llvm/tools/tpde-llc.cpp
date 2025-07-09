// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include "tpde-llvm/LLVMCompiler.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

#ifdef TPDE_LOGGING
  #include <spdlog/spdlog.h>
#endif

#define ARGS_NOEXCEPT
#include <args/args.hxx>

int main(int argc, char *argv[]) {
  args::ArgumentParser parser("TPDE for LLVM");
  args::HelpFlag help(parser, "help", "Display help", {'h', "help"});

  args::ValueFlag<unsigned> log_level(
      parser,
      "log_level",
      "Set the log level to 0=NONE, 1=ERR, 2=WARN(default), 3=INFO, 4=DEBUG, "
      ">5=TRACE",
      {'l', "log-level"},
      2);
  args::Flag print_ir(parser, "print_ir", "Print LLVM-IR", {"print-ir"});
  args::Flag regular_exit(
      parser, "regular_exit", "Exit regularly (no _Exit)", {"regular-exit"});

  args::ValueFlag<std::string> target(
      parser, "target", "Target architecture", {"target"}, args::Options::None);

  args::ValueFlag<std::string> obj_out_path(
      parser,
      "obj_path",
      "Path where the output object file should be written",
      {'o', "obj-out"},
      "-");

  args::ImplicitValueFlag<std::string> time_trace(
      parser,
      "time_trace",
      "Enable time tracing and write output to specified file",
      {"time-trace"},
      args::Options::None);

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

  if (time_trace) {
    llvm::timeTraceProfilerInitialize(0, argv[0]);
  }

  llvm::LLVMContext context;
  llvm::SMDiagnostic diag{};

  std::unique_ptr<llvm::Module> mod;
  {
    llvm::TimeTraceScope time_scope("Parse IR");
    mod = llvm::parseIRFile(ir_path.Get(), diag, context);
    if (!mod) {
      diag.print(argv[0], llvm::errs());
      return 1;
    }
  }

  if (print_ir) {
    mod->print(llvm::outs(), nullptr);
  }

#if LLVM_VERSION_MAJOR >= 21
  std::string triple_str = mod->getTargetTriple().str();
#else
  std::string triple_str = mod->getTargetTriple();
#endif
  if (target) {
    triple_str = target.Get();
  } else if (triple_str.empty()) {
    triple_str = llvm::sys::getDefaultTargetTriple();
  }
  llvm::Triple triple(triple_str);
  auto compiler = tpde_llvm::LLVMCompiler::create(triple);
  if (!compiler) {
    std::cerr << "Unknown architecture: " << triple_str << "\n";
    return 1;
  }

  std::vector<uint8_t> buf;
  {
    llvm::TimeTraceScope time_scope("Compile");
    if (!compiler->compile_to_elf(*mod, buf)) {
      std::cerr << "Failed to compile\n";
      return 1;
    }
  }

#ifndef NDEBUG
  // In debug builds, assert that compiling the module a second time in the same
  // compiler instance yields the same result.
  std::vector<uint8_t> buf2;
  if (!compiler->compile_to_elf(*mod, buf2)) {
    assert(false && "second compilation failed");
  }
  if (buf.size() != buf2.size() ||
      !std::equal(buf.begin(), buf.end(), buf2.begin())) {
    std::cerr << "result mismatch from second compilation!\n";
    std::cerr << "  sizeA=" << buf.size() << " sizeB=" << buf2.size() << "\n";
    assert(false);
  }
#endif

  if (obj_out_path.Get() == "-") {
    std::cout.write(reinterpret_cast<const char *>(buf.data()), buf.size());
    std::cout << std::flush;
  } else {
    std::ofstream out{obj_out_path.Get().c_str(), std::ios::binary};
    out.write(reinterpret_cast<const char *>(buf.data()), buf.size());
  }

  if (time_trace) {
    if (auto err = llvm::timeTraceProfilerWrite(time_trace.Get(),
                                                obj_out_path.Get())) {
      llvm::handleAllErrors(std::move(err), [&](const llvm::StringError &serr) {
        llvm::errs() << serr.getMessage() << "\n";
      });
    } else {
      llvm::timeTraceProfilerCleanup();
    }
  }

  if (print_ir) {
    llvm::outs() << "\nIR after modification:\n\n";
    mod->print(llvm::outs(), nullptr);
  }

  if (!regular_exit) {
    std::_Exit(0);
  }

  return 0;
}
