// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/FunctionWriter.hpp"
#include "tpde/base.hpp"
#include <fadec-enc2.h>

namespace tpde::x64 {

/// Helper class to write function text for X64.
class FunctionWriterX64 : public FunctionWriter<FunctionWriterX64> {
  friend class FunctionWriter<FunctionWriterX64>;

  static const TargetCIEInfo CIEInfo;

  // LEA tmp, [rip+table]; MOVSX idx, [tmp+4*idx]; ADD tmp, idx; JMP tmp
  // NB: MOVSX can be 5 bytes in case tmp is r13.
  static constexpr u32 JumpTableCodeSize = 18;

public:
  FunctionWriterX64() : FunctionWriter(CIEInfo) {}

  void align(size_t align) {
    u32 old_off = offset();
    FunctionWriter::align(align);
    // Pad text section with NOPs.
    if (u32 cur_off = offset(); cur_off > old_off) {
      fe64_NOP(cur_ptr() - (cur_off - old_off), cur_off - old_off);
    }
  }

  JumpTable &create_jump_table(u32 size, Reg idx, Reg tmp) {
    JumpTable &jt = alloc_jump_table(size, idx, tmp);
    ensure_space(JumpTableCodeSize);
    cur_ptr() += JumpTableCodeSize;
    return jt;
  }

private:
  void handle_fixups();
};

} // namespace tpde::x64
