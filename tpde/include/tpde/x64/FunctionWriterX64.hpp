// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/FunctionWriter.hpp"
#include <fadec-enc2.h>

namespace tpde::x64 {

/// Helper class to write function text.
class FunctionWriterX64 : public FunctionWriter<FunctionWriterX64> {
public:
  void align(size_t align) noexcept {
    u32 old_off = offset();
    FunctionWriter::align(align);
    // Pad text section with NOPs.
    if (u32 cur_off = offset(); cur_off > old_off) {
      fe64_NOP(cur_ptr() - (cur_off - old_off), cur_off - old_off);
    }
  }
};

} // namespace tpde::x64
