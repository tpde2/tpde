// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/FunctionWriter.hpp"

namespace tpde::a64 {

/// Helper class to write function text for AArch64.
class FunctionWriterA64 : public FunctionWriter<FunctionWriterA64> {
  friend class FunctionWriter<FunctionWriterA64>;

  util::SmallVector<u32, 16> veneers;
  u32 unresolved_cond_brs, unresolved_test_brs;

  static const TargetCIEInfo CIEInfo;

public:
  FunctionWriterA64() noexcept : FunctionWriter(CIEInfo) {}

  void begin_func(u32 align, u32 expected_size) noexcept {
    veneers.clear();
    // Must clear unresolved count here, begin_func will call more_space.
    unresolved_cond_brs = unresolved_test_brs = 0;
    FunctionWriter::begin_func(align, expected_size);
  }

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

  void label_ref(Label label, u32 off, LabelFixupKind kind) noexcept {
    FunctionWriter::label_ref(label, off, kind);
    if (kind == LabelFixupKind::AARCH64_COND_BR) {
      unresolved_cond_brs++;
    } else if (kind == LabelFixupKind::AARCH64_TEST_BR) {
      unresolved_test_brs++;
    }
  }

private:
  void handle_fixups() noexcept;
};

} // namespace tpde::a64
