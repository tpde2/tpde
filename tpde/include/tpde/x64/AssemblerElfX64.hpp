// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/AssemblerElf.hpp"
#include <fadec-enc2.h>

namespace tpde::x64 {

/// The x86_64-specific implementation for the AssemblerElf
struct AssemblerElfX64 : AssemblerElf<AssemblerElfX64> {
  using Base = AssemblerElf<AssemblerElfX64>;

  static const TargetInfoElf TARGET_INFO;

  explicit AssemblerElfX64() = default;
};

} // namespace tpde::x64
