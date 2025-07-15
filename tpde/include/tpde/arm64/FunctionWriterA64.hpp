// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/FunctionWriter.hpp"
#include "tpde/arm64/AssemblerElfA64.hpp"
#include <disarm64.h>

namespace tpde::a64 {

/// Helper class to write function text.
class FunctionWriterA64 : public FunctionWriter<FunctionWriterA64> {
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

inline void FunctionWriterA64::more_space(u32 size) noexcept {
  if (allocated_size() >= (128 * 1024 * 1024)) {
    // we do not support multiple text sections currently
    TPDE_FATAL("AArch64 doesn't support sections larger than 128 MiB");
  }

  // If the section has no unresolved conditional branch, veneer_info is null.
  // In that case, we don't need to do anything regarding veneers.
  auto *vi = static_cast<AssemblerElfA64::VeneerInfo *>(section->target_info);
  u32 unresolved_count =
      vi ? vi->unresolved_test_brs + vi->unresolved_cond_brs : 0;
  u32 veneer_size = sizeof(u32) * unresolved_count;
  FunctionWriter::more_space(size + veneer_size + 4);
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

} // namespace tpde::a64
