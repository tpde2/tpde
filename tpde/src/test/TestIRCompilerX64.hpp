// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "TestIR.hpp"
#include "tpde/base.hpp"
#include <vector>

namespace tpde::test {
std::vector<u8> compile_ir_x64(TestIR *ir, bool no_fixed_assignments);
} // namespace tpde::test
