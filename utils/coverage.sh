#!/bin/sh
# SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Script to generate coverage statistics after running tests.
# Overwrite environment variables TPDE_BUILD_DIR LLVM_COV LLVM_PROFDATA to
# adjust paths.

set -xeuo pipefail
TPDE_SRC_DIR=$(dirname "$0")/..
: "${LLVM_COV:=llvm-cov}"
: "${LLVM_PROFDATA:=llvm-profdata}"
: "${LLVM_CXXFILT:=llvm-cxxfilt}"
: "${TPDE_BUILD_DIR:=${TPDE_SRC_DIR}/build-coverage}"
TPDE_COVERAGE_DIR=$(realpath "${TPDE_BUILD_DIR}")/coverage
# Non-debug build, we don't want non-covered branches for asserts.
cmake -S "${TPDE_SRC_DIR}" -B "${TPDE_BUILD_DIR}" -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_FLAGS_RELEASE="-fno-global-isel -O0 -DNDEBUG" \
  -DCMAKE_CXX_FLAGS_RELEASE="-fno-global-isel -O0 -DNDEBUG" \
  -DCMAKE_BUILD_TYPE=Release -DTPDE_ENABLE_COVERAGE=ON
rm -rf "${TPDE_COVERAGE_DIR}"
LLVM_PROFILE_FILE="${TPDE_COVERAGE_DIR}/%4m-%p.profraw" \
  cmake --build "${TPDE_BUILD_DIR}" --target check-tpde || true
"${LLVM_PROFDATA}" merge -sparse -o "${TPDE_COVERAGE_DIR}/coverage.profdata" \
  "${TPDE_COVERAGE_DIR}"/*.profraw
"${LLVM_COV}" show -format=html -ignore-filename-regex=/deps/ \
  -show-instantiations=false \
  -show-branches=count -Xdemangler "${LLVM_CXXFILT}" -Xdemangler -n \
  -output-dir="${TPDE_BUILD_DIR}/coverage-html" \
  -object "${TPDE_BUILD_DIR}/tpde-llvm/tpde-llc" \
  -object "${TPDE_BUILD_DIR}/tpde-llvm/tpde-lli" \
  -object "${TPDE_BUILD_DIR}/tpde/tpde_test" \
  -instr-profile="${TPDE_COVERAGE_DIR}/coverage.profdata"
"${LLVM_COV}" report -ignore-filename-regex=/deps/ \
  -object "${TPDE_BUILD_DIR}/tpde-llvm/tpde-llc" \
  -object "${TPDE_BUILD_DIR}/tpde-llvm/tpde-lli" \
  -object "${TPDE_BUILD_DIR}/tpde/tpde_test" \
  -instr-profile="${TPDE_COVERAGE_DIR}/coverage.profdata"
