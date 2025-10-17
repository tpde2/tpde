// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/arm64/FunctionWriterA64.hpp"

#include <disarm64.h>

namespace tpde::a64 {

static constexpr auto get_cie_initial_instrs_a64() {
  std::array<u8, 32> data{};
  // the current frame setup does not have a constant offset from the FP
  // to the CFA so we need to encode that at the end
  // for now just encode the CFA before the first sub sp

  // def_cfa sp, 0
  unsigned len = FunctionWriterA64::write_eh_inst(
      data.data(), dwarf::DW_CFA_def_cfa, dwarf::a64::DW_reg_sp, 0);
  return std::make_pair(data, len);
}

static constexpr auto cie_instrs_a64 = get_cie_initial_instrs_a64();

const FunctionWriterA64::TargetCIEInfo FunctionWriterA64::CIEInfo{
    .instrs = {cie_instrs_a64.first.data(), cie_instrs_a64.second},
    .return_addr_register = dwarf::a64::DW_reg_lr,
    .code_alignment_factor = 4, // ULEB128 4
    .data_alignment_factor = 120, // SLEB128 -8
};

void FunctionWriterA64::more_space(u32 size) noexcept {
  if (allocated_size() >= (128 * 1024 * 1024)) {
    // we do not support multiple text sections currently
    TPDE_FATAL("AArch64 doesn't support sections larger than 128 MiB");
  }

  // If the section has no unresolved conditional branch, veneer_info is null.
  // In that case, we don't need to do anything regarding veneers.
  u32 unresolved_count = unresolved_test_brs + unresolved_cond_brs;
  u32 veneer_size = sizeof(u32) * unresolved_count;
  FunctionWriter::more_space(size + veneer_size + 4);
  if (veneer_size == 0) {
    return;
  }

  // TBZ has 14 bits, CBZ has 19 bits; but the first bit is the sign bit
  u32 max_dist = unresolved_test_brs ? 4 << (14 - 1) : 4 << (19 - 1);
  max_dist -= veneer_size; // must be able to reach last veneer
  // TODO: get a better approximation of the first unresolved condbr after the
  // last veneer.
  u32 first_condbr = veneers.empty() ? 0 : veneers.back();
  // If all condbrs can only jump inside the now-reserved memory, do nothing.
  if (first_condbr + max_dist > allocated_size()) {
    return;
  }

  u32 cur_off = offset();
  veneers.push_back(cur_off + 4);
  unresolved_test_brs = unresolved_cond_brs = 0;

  *reinterpret_cast<u32 *>(data_begin + cur_off) = de64_B(veneer_size / 4 + 1);
  std::memset(data_begin + cur_off + 4, 0, veneer_size);
  cur_ptr() += veneer_size + 4;
}

void FunctionWriterA64::handle_fixups() noexcept {
  for (const LabelFixup &fixup : label_fixups) {
    u32 label_off = label_offset(fixup.label);
    u32 *dst_ptr = reinterpret_cast<u32 *>(begin_ptr() + fixup.off);

    auto fix_condbr = [&](unsigned nbits) {
      i64 diff = i64(label_off) - i64(fixup.off);
      assert(diff >= 0 && diff < 128 * 1024 * 1024);
      // lowest two bits are ignored, highest bit is sign bit
      if (diff >= (4 << (nbits - 1))) {
        auto veneer =
            std::lower_bound(veneers.begin(), veneers.end(), fixup.off);
        assert(veneer != veneers.end());

        // Create intermediate branch at v.begin
        auto *br = reinterpret_cast<u32 *>(begin_ptr() + *veneer);
        assert(*br == 0 && "overwriting instructions with veneer branch");
        *br = de64_B((label_off - *veneer) / 4);
        diff = *veneer - fixup.off;
        *veneer += 4;
      }
      u32 off_mask = ((1 << nbits) - 1) << 5;
      *dst_ptr = (*dst_ptr & ~off_mask) | ((diff / 4) << 5);
    };

    switch (fixup.kind) {
    case LabelFixupKind::AARCH64_BR: {
      // diff from entry to label (should be positive tho)
      i64 diff = i64(label_off) - i64(fixup.off);
      assert(diff >= 0 && diff < 128 * 1024 * 1024);
      *dst_ptr = de64_B(diff / 4);
      break;
    }
    case LabelFixupKind::AARCH64_COND_BR:
      if (veneers.empty() || veneers.back() < fixup.off) {
        assert(unresolved_cond_brs > 0);
        unresolved_cond_brs -= 1;
      }
      fix_condbr(19); // CBZ/CBNZ has 19 bits.
      break;
    case LabelFixupKind::AARCH64_TEST_BR:
      if (veneers.empty() || veneers.back() < fixup.off) {
        assert(unresolved_test_brs > 0);
        unresolved_test_brs -= 1;
      }
      fix_condbr(14); // TBZ/TBNZ has 14 bits.
      break;
    case LabelFixupKind::AARCH64_JUMP_TABLE: {
      auto table_off = *dst_ptr;
      *dst_ptr = (i32)label_off - (i32)table_off;
      break;
    }
    default: TPDE_UNREACHABLE("unexpected label fixup kind");
    }
  }
}

} // end namespace tpde::a64
