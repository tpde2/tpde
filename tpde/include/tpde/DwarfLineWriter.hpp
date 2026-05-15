// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <map>
#include <vector>

#include "tpde/Assembler.hpp"
#include "tpde/base.hpp"
#include "tpde/util/SmallVector.hpp"
#include "tpde/util/VectorWriter.hpp"

namespace tpde {

/// Source location extracted from IR/compiler metadata.
struct DebugLocation {
  std::string_view file_name;
  std::string_view directory;
  u32 line;
  u16 column;
};

struct DebugFileEntry {
  std::string_view name;
  u32 dir_index;
};

/// DWARF line number program parameters (target-architecture-dependent).
struct DwarfConfig {
  u8 minimum_instruction_length;
  u8 maximum_operations_per_instruction = 1;
  bool default_is_stmt = true;
  i8 line_base = -5;
  u8 line_range = 14;
  u8 opcode_base = 13;

  explicit DwarfConfig(u8 min_inst_width)
      : minimum_instruction_length(min_inst_width) {}
};

struct PendingEntry {
  u32 raw_pc_offset;
  DebugLocation loc;
};

/// On-the-fly DWARF 5 line info writer.
///
/// File and directory tables grow dynamically as functions are compiled.
/// The line program is buffered incrementally. At finalization the header
/// (with the now-complete file/directory tables) is prepended and all three
/// DWARF sections are written to the assembler.
class DwarfLineWriter {
public:
  explicit DwarfLineWriter(const DwarfConfig &cfg);

  /// Called before each compiled instruction with the raw (pre-skew-correction)
  /// text-section offset and the corresponding source location.
  void push_location(u32 raw_pc_offset, DebugLocation loc);

  // Check if the given raw_pc_offset is the same as the last emitted one. If
  // so, it should be safe to skip emitting a new location.
  bool offset_equals_last_offset(u32 raw_pc_offset);

  /// Called after each function is fully compiled and the prologue skew is
  /// known. Applies skew to all raw pc_offsets (except the initial entry at 0),
  /// sorts and deduplicates records, and emits DWARF state-machine opcodes into
  /// the line program buffer. All functions share a single sequence;
  /// DW_LNE_end_sequence is emitted once by finalize().
  void end_function(u32 skew);

  /// Writes .debug_line, .debug_abbrev, and .debug_info sections to the
  /// assembler. Must be called after all end_function() calls.
  void finalize(Assembler &assembler);

  /// Indicates that a subprogram has been written for the current function,
  /// so the initial pending entry at offset 0 should be left unskewed.
  void set_subprogram_written();

private:
  struct LineState {
    u64 address = 0;
    u32 file = UINT32_MAX; // sentinel: force set_file on first record
    u32 line = 1;
    u32 column = 0;
    bool prologue_end = false;
    bool epilogue_begin = false;
  };

  // Configuration
  const tpde::DwarfConfig cfg;

  // Per-function temporary state
  std::vector<PendingEntry> pending;

  // File and directory tables (filled dynamically during end_function calls)
  using DirIdxMap = std::map<std::string_view, u32>;
  std::vector<std::string_view> directories;
  DirIdxMap dir_idx_map;
  using FileIdxMap = std::map<std::pair<u32, std::string_view>, u32>;
  std::vector<DebugFileEntry> files;
  FileIdxMap file_idx_map; // (dir_idx, name) → index

  // Buffered line program opcodes (all functions appended sequentially)
  util::SmallVector<u8, 1024> prog_buf;

  // DWARF state machine state (carries across functions within the single
  // sequence)
  LineState state;
  bool sequence_started = false;
  bool subprogram_written = false;

  // PC range tracking for .debug_info DW_AT_low_pc / DW_AT_high_pc
  u64 low_pc = UINT64_MAX;
  u64 high_pc = 0;

  u32 ensure_dir(const std::string_view &dir);
  u32 ensure_file(const std::string_view &name, u32 dir_idx);

  // DWARF line number program opcode emitters
  void emit_set_address(util::VectorWriter &w, u64 addr);
  void emit_end_sequence(util::VectorWriter &w);
  void emit_advance_line(util::VectorWriter &w, i32 delta);
  void emit_advance_pc(util::VectorWriter &w, u64 delta);
  void emit_set_file(util::VectorWriter &w, u32 file_idx);
  void emit_set_column(util::VectorWriter &w, u32 col);
  void emit_set_prologue_end(util::VectorWriter &w);
  void emit_set_epilogue_begin(util::VectorWriter &w);
  void emit_copy(util::VectorWriter &w);

  void emit_record(util::VectorWriter &w,
                   u32 pc_offset,
                   u32 file_idx,
                   u32 line,
                   u16 col,
                   bool prologue_end,
                   bool epilogue_begin);

  // Section writing helpers
  void write_directory_table(util::VectorWriter &w) const;
  void write_file_table(util::VectorWriter &w) const;
  void write_string(util::VectorWriter &w, const std::string_view &s) const;
  void write_debug_abbrev(Assembler &assembler) const;
  void write_debug_line(Assembler &assembler) const;
  void write_debug_info(Assembler &assembler, u64 low_pc, u64 high_pc) const;
  void copy_to_section(const u8 *data,
                       size_t size,
                       Assembler &assembler,
                       SectionKind kind) const;
};

} // namespace tpde
