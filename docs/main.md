\mainpage

TPDE is a fast compiler back-end framework that adapts to existing SSA IRs.
The primary goal is low-latency compilation while maintaining reasonable (`-O0`) code quality, e.g., as baseline compiler for JIT compilation or unoptimized builds.
Currently, TPDE only targets ELF-based x86-64 and AArch64 (Armv8.1) platforms.

This repository contains:

- \subpage tpde-main "TPDE": the core compiler framework.
- \ref tpde-encodegen "TPDE-Encodegen": a utility for easing the use of TPDE by deriving code generators through LLVM's Machine IR.
- \subpage tpde-llvm-main "TPDE-LLVM": a standalone back-end for LLVM-IR, which compiles 10--20x faster than LLVM -O0 with similar code quality, usable as library (e.g., for JIT), as tool (`tpde-llc`), and integrated in Clang/Flang (with a patch).

### Getting Started

Requirements/external dependencies:

- GNU-compatible C++20 compiler (e.g., Clang 19+, GCC 14+)
- LLVM/Clang 20.1 or 19.1 (only for tests, TPDE-LLVM and TPDE-Encodegen)
  - Prefer LLVM 20.1. LLVM 19.1 should work, but some tests are disabled due to different code generation. More recent LLVM versions typically work, but some tests will fail.
  - Note: for tests, also the LLVM tools (`lit`, `llc`, `llvm-objdump`, `llvm-readelf`, `llvm-readobj`, `llvm-dwarfdump`, `llvm-as`, `FileCheck`, etc.) are required.
- Python 3.10+
- Other dependencies are bundled as submodules.

It is possible to build and use the core TPDE framework without LLVM (`-DTPDE_ENABLE_ENCODEGEN=OFF -DTPDE_ENABLE_LLVM=OFF -DTPDE_INCLUDE_TESTS=OFF`).

```shell
# Clone recursive due to submodules
git clone https://github.com/tpde2/tpde.git --recursive
cd tpde
# Configure
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
# Compile
cmake --build build
# Optionally: run tests
cmake --build build --target check-tpde
```

### Publications

- Tobias Schwarz, Tobias Kamm, and Alexis Engelke. TPDE: A Fast Adaptable Compiler Back-End Framework. [arXiv:2505.22610](https://arxiv.org/abs/2505.22610) [cs.PL]. 2025.

### License

Generally: Apache-2.0 WITH LLVM-exception. (Detailed license information is attached to every file. Dependencies may have different licenses.)
