// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/DwarfLineWriter.hpp"

#include "tpde/Assembler.hpp"
#include "tpde/DWARF.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace tpde {

DwarfLineWriter::DwarfLineWriter(const tpde::DwarfConfig &cfg) : cfg(cfg) {}

void DwarfLineWriter::push_location(u32 raw_pc_offset, DebugLocation loc) {
  pending.emplace_back(raw_pc_offset, std::move(loc));
}

bool DwarfLineWriter::offset_equals_last_offset(u32 raw_pc_offset) {
  return !pending.empty() && raw_pc_offset == pending.back().raw_pc_offset;
}

void DwarfLineWriter::end_function(u32 skew) {
  if (pending.empty()) {
    return;
  }

  // Apply skew to all pending entries except the initial entry for the function
  // itself if a subprogram has been written.
  auto it = subprogram_written ? pending.begin() + 1 : pending.begin();
  for (; it != pending.end(); ++it) {
    it->raw_pc_offset -= skew;
  }

  const auto comp = [](const PendingEntry &a, const PendingEntry &b) {
    return a.raw_pc_offset < b.raw_pc_offset;
  };
  std::ranges::stable_sort(pending, comp);

  // Track PC range for .debug_info.
  low_pc = std::min(low_pc, static_cast<u64>(pending.front().raw_pc_offset));
  high_pc = std::max(high_pc, static_cast<u64>(pending.back().raw_pc_offset));

  // Emit DWARF state-machine opcodes into prog_buf.
  util::VectorWriter w(prog_buf);

  if (!sequence_started) {
    emit_set_address(w, pending.front().raw_pc_offset);
    state.address = pending.front().raw_pc_offset;
    sequence_started = true;
  }

  for (const PendingEntry &e : pending) {
    const u32 file_idx =
        ensure_file(e.loc.file_name, ensure_dir(e.loc.directory));
    emit_record(
        w, e.raw_pc_offset, file_idx, e.loc.line, e.loc.column, false, false);
  }

  // Reset state for next function
  state.prologue_end = false;
  state.epilogue_begin = false;
  pending.clear();
  subprogram_written = false;
}

void DwarfLineWriter::finalize(Assembler &assembler) {
  // If the buffer is empty, we don't need to write the DWARF sections
  if (prog_buf.empty()) {
    return;
  }

  // Close the single line program sequence.
  {
    util::VectorWriter w(prog_buf);
    emit_end_sequence(w);
  }

  // --- .debug_line ---
  write_debug_line(assembler);

  // --- .debug_abbrev ---
  write_debug_abbrev(assembler);

  // --- .debug_info ---
  // high_pc is exclusive (one past the last recorded offset).
  write_debug_info(assembler, low_pc, high_pc + 1);
}

void DwarfLineWriter::set_subprogram_written() { subprogram_written = true; }

// ---- file/directory table helpers -------------------------------------------

u32 DwarfLineWriter::ensure_dir(const std::string_view &dir) {
  const auto [it, inserted] =
      dir_idx_map.try_emplace(dir, static_cast<u32>(directories.size()));
  if (inserted) {
    directories.push_back(dir);
  }
  return it->second;
}

u32 DwarfLineWriter::ensure_file(const std::string_view &name, u32 dir_idx) {
  const auto key = std::make_pair(dir_idx, name);
  const auto [it, inserted] =
      file_idx_map.try_emplace(key, static_cast<u32>(files.size()));
  if (inserted) {
    files.emplace_back(name, dir_idx);
  }
  return it->second;
}

// ---- DWARF opcode emitters --------------------------------------------------

void DwarfLineWriter::emit_set_address(util::VectorWriter &w, u64 addr) {
  w.write<u8>(dwarf::DW_LNS_extended_op);
  w.write_uleb(1 + 8);
  w.write<u8>(dwarf::DW_LNE_set_address);
  w.write<u64>(addr);
}

void DwarfLineWriter::emit_end_sequence(util::VectorWriter &w) {
  w.write<u8>(dwarf::DW_LNS_extended_op);
  w.write_uleb(1);
  w.write<u8>(dwarf::DW_LNE_end_sequence);
}

void DwarfLineWriter::emit_advance_line(util::VectorWriter &w, i32 delta) {
  w.write<u8>(dwarf::DW_LNS_advance_line);
  w.write_sleb(delta);
}

void DwarfLineWriter::emit_advance_pc(util::VectorWriter &w, u64 delta) {
  w.write<u8>(dwarf::DW_LNS_advance_pc);
  w.write_uleb(delta);
}

void DwarfLineWriter::emit_set_file(util::VectorWriter &w, u32 file_idx) {
  w.write<u8>(dwarf::DW_LNS_set_file);
  w.write_uleb(file_idx);
}

void DwarfLineWriter::emit_set_column(util::VectorWriter &w, u32 col) {
  w.write<u8>(dwarf::DW_LNS_set_column);
  w.write_uleb(col);
}

void DwarfLineWriter::emit_set_prologue_end(util::VectorWriter &w) {
  w.write<u8>(dwarf::DW_LNS_set_prologue_end);
}

void DwarfLineWriter::emit_set_epilogue_begin(util::VectorWriter &w) {
  w.write<u8>(dwarf::DW_LNS_set_epilogue_begin);
}

void DwarfLineWriter::emit_copy(util::VectorWriter &w) {
  w.write<u8>(dwarf::DW_LNS_copy);
}

void DwarfLineWriter::emit_record(util::VectorWriter &w,
                                  u32 pc_offset,
                                  u32 file_idx,
                                  u32 line,
                                  u16 col,
                                  bool prologue_end,
                                  bool epilogue_begin) {
  const u64 addr_delta = pc_offset - state.address;
  if (addr_delta != 0) {
    emit_advance_pc(w, addr_delta / cfg.minimum_instruction_length);
    state.address = pc_offset;
  }

  const i32 line_delta = static_cast<i32>(line) - static_cast<i32>(state.line);
  if (line_delta != 0) {
    emit_advance_line(w, line_delta);
    state.line = line;
  }

  if (file_idx != state.file) {
    emit_set_file(w, file_idx);
    state.file = file_idx;
  }

  if (col != state.column) {
    emit_set_column(w, col);
    state.column = col;
  }

  if (prologue_end && !state.prologue_end) {
    emit_set_prologue_end(w);
    state.prologue_end = true;
  }

  if (epilogue_begin && !state.epilogue_begin) {
    emit_set_epilogue_begin(w);
    state.epilogue_begin = true;
  }

  emit_copy(w);
}

// ---- section writing helpers ------------------------------------------------

void DwarfLineWriter::write_directory_table(util::VectorWriter &w) const {
  // DWARF 5 format: format_count, formats, entry_count, entries.
  w.write<u8>(1); // directory_entry_format_count
  w.write_uleb(dwarf::DW_LNCT_path);
  w.write_uleb(dwarf::DW_FORM_string);
  w.write_uleb(directories.size());
  for (const std::string_view &d : directories) {
    write_string(w, d);
  }
}

void DwarfLineWriter::write_file_table(util::VectorWriter &w) const {
  // DWARF 5 format: format_count, formats, entry_count, entries.
  w.write<u8>(2); // file_name_entry_format_count
  w.write_uleb(dwarf::DW_LNCT_path);
  w.write_uleb(dwarf::DW_FORM_string);
  w.write_uleb(dwarf::DW_LNCT_directory_index);
  w.write_uleb(dwarf::DW_FORM_udata);
  w.write_uleb(files.size());

  for (const DebugFileEntry &f : files) {
    write_string(w, f.name);
    w.write_uleb(f.dir_index);
  }
}

void DwarfLineWriter::write_string(util::VectorWriter &w,
                                   const std::string_view &s) const {
  const auto *data = reinterpret_cast<const u8 *>(s.data());
  w.write(std::span<const u8>(data, s.size()));
  w.write<u8>(0); // null terminator
}

void DwarfLineWriter::write_debug_abbrev(Assembler &assembler) const {
  util::SmallVector<u8, 256> buf;
  util::VectorWriter w(buf);

  w.write_uleb(1); // abbrev code
  w.write_uleb(dwarf::DW_TAG_compile_unit);
  w.write<u8>(dwarf::DW_CHILDREN_no);

  w.write_uleb(dwarf::DW_AT_stmt_list);
  w.write_uleb(dwarf::DW_FORM_sec_offset);

  w.write_uleb(dwarf::DW_AT_low_pc);
  w.write_uleb(dwarf::DW_FORM_addr);

  w.write_uleb(dwarf::DW_AT_high_pc);
  w.write_uleb(dwarf::DW_FORM_addr);

  w.write_uleb(0); // end of attributes
  w.write_uleb(0);
  w.write<u8>(0); // end of abbrev table

  w.flush();
  copy_to_section(w.data(), w.size(), assembler, SectionKind::DebugAbbrev);
}

void DwarfLineWriter::write_debug_line(Assembler &assembler) const {
  // Write the header into a temporary buffer, then copy header + prog_buf
  // into the section as one contiguous block.
  util::SmallVector<u8, 4096> hdr_buf;
  util::VectorWriter hw(hdr_buf);

  // unit_length placeholder (filled after we know total size)
  const u32 unit_len_off = static_cast<u32>(hw.size());
  hw.write<u32>(0);

  hw.write<u16>(5); // DWARF version 5
  hw.write<u8>(8);  // address_size (64-bit)
  hw.write<u8>(0);  // segment_selector_size

  // header_length placeholder (filled after header body is written)
  const u32 hdr_len_off = static_cast<u32>(hw.size());
  hw.write<u32>(0);
  const u32 hdr_body_start = static_cast<u32>(hw.size());

  hw.write<u8>(cfg.minimum_instruction_length);
  hw.write<u8>(1); // maximum_operations_per_instruction
  hw.write<u8>(cfg.default_is_stmt ? 1 : 0);
  hw.write<i8>(cfg.line_base);
  hw.write<u8>(cfg.line_range == 0 ? 14 : cfg.line_range);
  hw.write<u8>(cfg.opcode_base);

  static constexpr u8 std_opcode_lengths[] = {
      0, // extended
      0, // copy
      1, // advance_pc
      1, // advance_line
      1, // set_file
      1, // set_column
      0, // negate_stmt
      0, // set_basic_block
      0, // const_add_pc
      1, // fixed_advance_pc
      0, // set_prologue_end
      0, // set_epilogue_begin
      1, // set_isa
  };
  for (u32 i = 0; i < static_cast<u32>(cfg.opcode_base) - 1u; ++i) {
    hw.write<u8>(std_opcode_lengths[i]);
  }

  write_directory_table(hw);
  write_file_table(hw);

  // Patch header_length: bytes from hdr_body_start to end of header.
  hw.flush();
  const u32 hdr_len = static_cast<u32>(hw.size()) - hdr_body_start;
  std::memcpy(hw.data() + hdr_len_off, &hdr_len, sizeof(u32));

  // Patch unit_length: from after the unit_length field to end of section.
  const u32 unit_len = static_cast<u32>(hw.size()) - unit_len_off -
                       sizeof(u32) + static_cast<u32>(prog_buf.size());
  std::memcpy(hw.data() + unit_len_off, &unit_len, sizeof(u32));

  // Write header then line program into the section.
  SecRef sec = assembler.create_section(SectionKind::DebugLine);
  DataSection &ds = assembler.get_section(sec);
  ds.data.append(hw.data(), hw.data() + hw.size());
  ds.data.append(prog_buf.data(), prog_buf.data() + prog_buf.size());
  assert(sec.valid());
}

void DwarfLineWriter::write_debug_info(Assembler &assembler,
                                       u64 low_pc,
                                       u64 high_pc) const {
  util::SmallVector<u8, 256> buf;
  util::VectorWriter w(buf);

  const u32 unit_len_off = static_cast<u32>(w.size());
  w.write<u32>(0); // placeholder

  w.write<u16>(5);                   // DWARF version
  w.write<u8>(dwarf::DW_UT_compile); // unit_type
  w.write<u8>(8);                    // address_size (64-bit)
  w.write<u32>(0);                   // debug_abbrev_offset

  w.write_uleb(1); // abbrev code = 1 (DW_TAG_compile_unit)

  w.write<u32>(0);       // DW_AT_stmt_list = offset 0 into .debug_line
  w.write<u64>(low_pc);  // DW_AT_low_pc
  w.write<u64>(high_pc); // DW_AT_high_pc

  w.write<u8>(0); // end of DIEs

  w.flush();

  const u32 unit_len =
      static_cast<u32>(w.size()) - unit_len_off - static_cast<u32>(sizeof(u32));
  std::memcpy(w.data() + unit_len_off, &unit_len, sizeof(u32));

  copy_to_section(w.data(), w.size(), assembler, SectionKind::DebugInfo);
}

void DwarfLineWriter::copy_to_section(const u8 *data,
                                      size_t size,
                                      Assembler &assembler,
                                      SectionKind kind) const {
  SecRef sec = assembler.create_section(kind);
  DataSection &ds = assembler.get_section(sec);
  ds.data.append(data, data + size);
  assert(sec.valid());
}

} // namespace tpde
