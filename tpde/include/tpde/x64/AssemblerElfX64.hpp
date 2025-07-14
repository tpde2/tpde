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

  class SectionWriter : public Base::SectionWriterBase<SectionWriter> {
  public:
    void align(size_t align) noexcept;
  };

  enum class UnresolvedEntryKind : u8 {
    JMP_OR_MEM_DISP,
    JUMP_TABLE,
  };

  explicit AssemblerElfX64() = default;

  void add_unresolved_entry(Label label,
                            SecRef sec,
                            u32 off,
                            UnresolvedEntryKind kind) noexcept {
    AssemblerElfBase::reloc_sec(sec, label, static_cast<u8>(kind), off);
  }

  void handle_fixup(const TempSymbolInfo &info,
                    const TempSymbolFixup &fixup) noexcept;
};

inline void
    AssemblerElfX64::handle_fixup(const TempSymbolInfo &info,
                                  const TempSymbolFixup &fixup) noexcept {
  // TODO: emit relocations when fixup is in different section
  assert(info.section == fixup.section && "multi-text section not supported");
  u8 *dst_ptr = get_section(fixup.section).data.data() + fixup.off;

  switch (static_cast<UnresolvedEntryKind>(fixup.kind)) {
  case UnresolvedEntryKind::JMP_OR_MEM_DISP: {
    // fix the jump immediate
    u32 value = (info.off - fixup.off) - 4;
    std::memcpy(dst_ptr, &value, sizeof(u32));
    break;
  }
  case UnresolvedEntryKind::JUMP_TABLE: {
    const auto table_off = *reinterpret_cast<u32 *>(dst_ptr);
    const auto diff = (i32)info.off - (i32)table_off;
    std::memcpy(dst_ptr, &diff, sizeof(u32));
    break;
  }
  }
}

inline void AssemblerElfX64::SectionWriter::align(size_t align) noexcept {
  u32 old_off = offset();
  SectionWriterBase::align(align);
  // Pad text section with NOPs.
  if (u32 cur_off = offset(); cur_off > old_off) {
    fe64_NOP(cur_ptr() - (cur_off - old_off), cur_off - old_off);
  }
}

} // namespace tpde::x64
