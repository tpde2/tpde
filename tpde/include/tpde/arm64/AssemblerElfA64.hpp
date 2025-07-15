// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/AssemblerElf.hpp"
#include <disarm64.h>

namespace tpde::a64 {

/// The AArch64-specific implementation for the AssemblerElf
struct AssemblerElfA64 : AssemblerElf<AssemblerElfA64> {
  using Base = AssemblerElf<AssemblerElfA64>;

  static const TargetInfoElf TARGET_INFO;

  explicit AssemblerElfA64() = default;
};

} // namespace tpde::a64
