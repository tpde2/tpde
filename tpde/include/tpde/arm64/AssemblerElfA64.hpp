// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/AssemblerElf.hpp"
#include "tpde/util/SegmentedVector.hpp"
#include "tpde/util/SmallVector.hpp"
#include <disarm64.h>

namespace tpde::a64 {

/// The AArch64-specific implementation for the AssemblerElf
struct AssemblerElfA64 : AssemblerElf<AssemblerElfA64> {
  using Base = AssemblerElf<AssemblerElfA64>;

  static const TargetInfoElf TARGET_INFO;

  class SectionWriter : public Base::SectionWriterBase<SectionWriter> {
  public:
    void more_space(u32 size) noexcept;

    bool try_write_inst(u32 inst) noexcept {
      if (inst == 0) {
        return false;
      }
      write(inst);
      return true;
    }

    void write_inst(u32 inst) noexcept {
      assert(inst != 0);
      write(inst);
    }

    void write_inst_unchecked(u32 inst) noexcept {
      assert(inst != 0);
      write_unchecked(inst);
    }
  };

  enum class UnresolvedEntryKind : u8 {
    BR,
    COND_BR,
    TEST_BR,
    JUMP_TABLE,
  };

  /// Information about veneers and unresolved branches for a section.
  struct VeneerInfo {
    /// Begin offsets of veneer space. A veneer always has space for all
    /// unresolved cbz/tbz branches that come after it.
    util::SmallVector<u32, 16> veneers;
    u32 unresolved_test_brs = 0, unresolved_cond_brs = 0;
  };

  util::SegmentedVector<VeneerInfo> veneer_infos;

  explicit AssemblerElfA64() = default;

private:
  VeneerInfo &get_veneer_info(DataSection &section) noexcept {
    if (!section.target_info) [[unlikely]] {
      section.target_info = &veneer_infos.emplace_back();
    }
    return *static_cast<VeneerInfo *>(section.target_info);
  }

public:
  void add_unresolved_entry(Label label,
                            SecRef sec,
                            u32 off,
                            UnresolvedEntryKind kind) noexcept {
    AssemblerElfBase::reloc_sec(sec, label, static_cast<u8>(kind), off);
    if (kind == UnresolvedEntryKind::COND_BR) {
      get_veneer_info(get_section(sec)).unresolved_cond_brs++;
    } else if (kind == UnresolvedEntryKind::TEST_BR) {
      get_veneer_info(get_section(sec)).unresolved_test_brs++;
    }
  }

  void handle_fixup(const TempSymbolInfo &info,
                    const TempSymbolFixup &fixup) noexcept;

  void reset() noexcept;
};

inline void
    AssemblerElfA64::handle_fixup(const TempSymbolInfo &info,
                                  const TempSymbolFixup &fixup) noexcept {
  // TODO: emit relocations when fixup is in different section
  assert(info.section == fixup.section && "multi-text section not supported");
  DataSection &section = get_section(fixup.section);
  VeneerInfo &vi = get_veneer_info(section);
  auto &veneers = vi.veneers;

  u8 *section_data = section.data.data();
  u32 *dst_ptr = reinterpret_cast<u32 *>(section_data + fixup.off);

  auto fix_condbr = [&](unsigned nbits) {
    i64 diff = (i64)info.off - (i64)fixup.off;
    assert(diff >= 0 && diff < 128 * 1024 * 1024);
    // lowest two bits are ignored, highest bit is sign bit
    if (diff >= (4 << (nbits - 1))) {
      auto veneer = std::lower_bound(veneers.begin(), veneers.end(), fixup.off);
      assert(veneer != veneers.end());

      // Create intermediate branch at v.begin
      auto *br = reinterpret_cast<u32 *>(section_data + *veneer);
      assert(*br == 0 && "overwriting instructions with veneer branch");
      *br = de64_B((info.off - *veneer) / 4);
      diff = *veneer - fixup.off;
      *veneer += 4;
    }
    u32 off_mask = ((1 << nbits) - 1) << 5;
    *dst_ptr = (*dst_ptr & ~off_mask) | ((diff / 4) << 5);
  };

  switch (static_cast<UnresolvedEntryKind>(fixup.kind)) {
  case UnresolvedEntryKind::BR: {
    // diff from entry to label (should be positive tho)
    i64 diff = (i64)info.off - (i64)fixup.off;
    assert(diff >= 0 && diff < 128 * 1024 * 1024);
    *dst_ptr = de64_B(diff / 4);
    break;
  }
  case UnresolvedEntryKind::COND_BR:
    if (veneers.empty() || veneers.back() < fixup.off) {
      assert(vi.unresolved_cond_brs > 0);
      vi.unresolved_cond_brs -= 1;
    }
    fix_condbr(19); // CBZ/CBNZ has 19 bits.
    break;
  case UnresolvedEntryKind::TEST_BR:
    if (veneers.empty() || veneers.back() < fixup.off) {
      assert(vi.unresolved_test_brs > 0);
      vi.unresolved_test_brs -= 1;
    }
    fix_condbr(14); // TBZ/TBNZ has 14 bits.
    break;
  case UnresolvedEntryKind::JUMP_TABLE: {
    auto table_off = *reinterpret_cast<u32 *>(section_data + fixup.off);
    *dst_ptr = (i32)info.off - (i32)table_off;
    break;
  }
  }
}

inline void AssemblerElfA64::SectionWriter::more_space(u32 size) noexcept {
  if (allocated_size() >= (128 * 1024 * 1024)) {
    // we do not support multiple text sections currently
    TPDE_FATAL("AArch64 doesn't support sections larger than 128 MiB");
  }

  // If the section has no unresolved conditional branch, veneer_info is null.
  // In that case, we don't need to do anything regarding veneers.
  VeneerInfo *vi = static_cast<VeneerInfo *>(section->target_info);
  u32 unresolved_count =
      vi ? vi->unresolved_test_brs + vi->unresolved_cond_brs : 0;
  u32 veneer_size = sizeof(u32) * unresolved_count;
  SectionWriterBase::more_space(size + veneer_size + 4);
  if (veneer_size == 0) {
    return;
  }

  // TBZ has 14 bits, CBZ has 19 bits; but the first bit is the sign bit
  u32 max_dist = vi->unresolved_test_brs ? 4 << (14 - 1) : 4 << (19 - 1);
  max_dist -= veneer_size; // must be able to reach last veneer
  // TODO: get a better approximation of the first unresolved condbr after the
  // last veneer.
  u32 first_condbr = vi->veneers.empty() ? 0 : vi->veneers.back();
  // If all condbrs can only jump inside the now-reserved memory, do nothing.
  if (first_condbr + max_dist > allocated_size()) {
    return;
  }

  u32 cur_off = offset();
  vi->veneers.push_back(cur_off + 4);
  vi->unresolved_test_brs = vi->unresolved_cond_brs = 0;

  *reinterpret_cast<u32 *>(data_begin + cur_off) = de64_B(veneer_size / 4 + 1);
  std::memset(data_begin + cur_off + 4, 0, veneer_size);
  cur_ptr() += veneer_size + 4;
}

inline void AssemblerElfA64::reset() noexcept {
  Base::reset();
  veneer_infos.clear();
}

} // namespace tpde::a64
