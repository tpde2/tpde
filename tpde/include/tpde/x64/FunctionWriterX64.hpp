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

public:
  FunctionWriterX64() noexcept : FunctionWriter(CIEInfo) {}

  void align(size_t align) noexcept {
    u32 old_off = offset();
    FunctionWriter::align(align);
    // Pad text section with NOPs.
    if (u32 cur_off = offset(); cur_off > old_off) {
      fe64_NOP(cur_ptr() - (cur_off - old_off), cur_off - old_off);
    }
  }

private:
  void handle_fixups() noexcept;
};

inline void FunctionWriterX64::handle_fixups() noexcept {
  for (const LabelFixup &fixup : label_fixups) {
    u32 label_off = label_offset(fixup.label);
    u8 *dst_ptr = begin_ptr() + fixup.off;
    switch (fixup.kind) {
    case LabelFixupKind::X64_JMP_OR_MEM_DISP: {
      // fix the jump immediate
      u32 value = (label_off - fixup.off) - 4;
      std::memcpy(dst_ptr, &value, sizeof(u32));
      break;
    }
    case LabelFixupKind::X64_JUMP_TABLE: {
      const auto table_off = *reinterpret_cast<u32 *>(dst_ptr);
      const auto diff = (i32)label_off - (i32)table_off;
      std::memcpy(dst_ptr, &diff, sizeof(u32));
      break;
    }
    default: TPDE_UNREACHABLE("unexpected label fixup kind");
    }
  }
}

} // namespace tpde::x64
