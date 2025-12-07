// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/Assembler.hpp"
#include "tpde/DWARF.hpp"
#include "tpde/RegisterFile.hpp"
#include "tpde/base.hpp"
#include "tpde/util/BumpAllocator.hpp"
#include "tpde/util/SmallVector.hpp"
#include "tpde/util/VectorWriter.hpp"
#include <cstring>
#include <span>

namespace tpde {

enum class Label : u32 {
};

enum class LabelFixupKind : u8 {
  ArchKindBegin,

  X64_JMP_OR_MEM_DISP = ArchKindBegin,

  AARCH64_BR = ArchKindBegin,
  AARCH64_COND_BR,
  AARCH64_TEST_BR,
};

/// Architecture-independent base for helper class to write function text.
class FunctionWriterBase {
protected:
  struct TargetCIEInfo {
    /// The initial instructions for the CIE.
    std::span<const u8> instrs;
    /// The return address register for the CIE.
    u8 return_addr_register;
    /// Code alignment factor for the CIE, ULEB128, must be one byte.
    u8 code_alignment_factor;
    /// Data alignment factor for the CIE, SLEB128, must be one byte.
    u8 data_alignment_factor;
  };

  Assembler *assembler;
  const TargetCIEInfo &cie_info;

  DataSection *section = nullptr;
  u8 *data_begin = nullptr;
  u8 *data_cur = nullptr;
  u8 *data_reserve_end = nullptr;

  u32 func_begin;  ///< Begin offset of the current function.
  u32 reloc_begin; ///< Begin of relocations for current function.

  u32 label_skew; ///< Offset to subtract from all label offsets.
  /// Label offsets into section, ~0u indicates unplaced label.
  util::SmallVector<u32> label_offsets;

protected:
  struct LabelFixup {
    Label label;
    u32 off;
    LabelFixupKind kind;
  };

  /// Fixups for labels placed after their first use, processed at function end.
  util::SmallVector<LabelFixup> label_fixups;

  /// Growth size for more_space; adjusted exponentially after every grow.
  u32 growth_size = 0x10000;

public:
  struct JumpTable {
    u32 size; ///< Number of jump table entries.
    u32 off;  ///< Start offset in code, must hold sufficient space for jump.

    Reg idx; ///< Register holding the table index.
    Reg tmp; ///< Second temporary register.
    u8 misc; ///< Target-specific data.

    std::span<Label> labels() {
      return {reinterpret_cast<Label *>(this + 1), size};
    }
  };

  static_assert(std::is_trivially_destructible_v<JumpTable>);

private:
  util::BumpAllocator<> jump_table_alloc;

protected:
  util::SmallVector<JumpTable *> jump_tables;

private:
  DataSection *eh_frame_section;

public:
  util::VectorWriter eh_writer;

private:
  /// The current personality function (if any)
  SymRef cur_personality_func_addr;
  u32 eh_cur_cie_off = 0u;
  u32 fde_start;

  struct ExceptCallSiteInfo {
    /// Start offset *in section* (not inside function)
    u64 start;
    u64 len;
    Label landing_pad;
    u32 action_entry;
  };

  /// Call Sites for current function
  util::SmallVector<ExceptCallSiteInfo, 0> except_call_site_table;

  /// Temporary storage for encoding call sites
  util::SmallVector<u8, 0> except_encoded_call_sites;
  /// Action Table for current function
  util::SmallVector<u8, 0> except_action_table;
  /// The type_info table (contains the symbols which contain the pointers to
  /// the type_info)
  util::SmallVector<SymRef, 0> except_type_info_table;
  /// Table for exception specs
  util::SmallVector<u8, 0> except_spec_table;

protected:
  FunctionWriterBase(const TargetCIEInfo &cie_info) noexcept
      : cie_info(cie_info) {}

  ~FunctionWriterBase() {
    assert(data_cur == data_reserve_end &&
           "must flush section writer before destructing");
  }

public:
  /// Get the SecRef of the current section.
  SecRef get_sec_ref() const noexcept { return get_section().get_ref(); }

  /// Get the current section.
  DataSection &get_section() const noexcept {
    assert(section != nullptr);
    return *section;
  }

  /// Switch section writer to new section; must be flushed.
  void switch_section(DataSection &new_section) noexcept {
    assert(data_cur == data_reserve_end &&
           "must flush section writer before switching sections");
    section = &new_section;
    data_begin = section->data.data();
    data_cur = data_begin + section->data.size();
    data_reserve_end = data_cur;
  }

  void begin_module(Assembler &assembler) noexcept;

  void end_module() noexcept;

  void begin_func() noexcept;

  /// Get the current offset into the section.
  size_t offset() const noexcept { return data_cur - data_begin; }

  /// Get the current allocated size of the section.
  size_t allocated_size() const noexcept {
    return data_reserve_end - data_begin;
  }

  /// Pointer to beginning of section data.
  u8 *begin_ptr() noexcept { return data_begin; }

  /// Modifiable pointer to current writing position of the section. Must not
  /// be moved beyond the allocated region.
  u8 *&cur_ptr() noexcept { return data_cur; }

  void more_space(size_t size) noexcept;

  /// Record relocation at the given offset.
  void reloc(SymRef sym, u32 type, u64 off, i64 addend = 0) noexcept {
    assembler->reloc_sec(get_sec_ref(), sym, type, off, addend);
  }

  /// Remove bytes and adjust labels/relocations accordingly. The covered region
  /// must be before the first label, i.e., this function can only be used to
  /// cut out bytes from the function prologue.
  void remove_prologue_bytes(u32 start, u32 size) noexcept;

  void flush() noexcept {
    if (data_cur != data_reserve_end) {
      section->data.resize(offset());
      data_reserve_end = data_cur;
#ifndef NDEBUG
      section->locked = false;
#endif
    }
  }

  /// \name Labels
  /// @{

  /// Create a new unplaced label.
  Label label_create() noexcept {
    const Label label = Label(label_offsets.size());
    label_offsets.push_back(~0u);
    return label;
  }

  bool label_is_pending(Label label) const noexcept {
    return label_offsets[u32(label)] == ~0u;
  }

  u32 label_offset(Label label) const noexcept {
    assert(!label_is_pending(label));
    return label_offsets[u32(label)] - label_skew;
  }

  /// Place unplaced label at the specified offset inside the section.
  void label_place(Label label, u32 off) noexcept {
    assert(label_skew == 0 && "label_place called after prologue truncation");
    assert(label_is_pending(label));
    label_offsets[u32(label)] = off;
  }

  /// Reference label at given offset inside the code section.
  void label_ref(Label label, u32 off, LabelFixupKind kind) noexcept {
    // We also permit this to be called even if label is already placed to
    // simplify code at the call site. It might be preferable, however, to
    // immediately write the final value.
    label_fixups.emplace_back(LabelFixup{label, off, kind});
  }

  /// @}

protected:
  /// Allocate a jump table for the current location.
  JumpTable &alloc_jump_table(u32 size, Reg idx, Reg tmp) noexcept;

  SecRef get_jump_table_section() noexcept {
    // TODO: move jump tables to separate read-only data section if function is
    // not in the default text section (e.g., part of a section group).
    return assembler->get_default_section(SectionKind::ReadOnly);
  }

  /// \name DWARF CFI
  /// @{

public:
  static constexpr u32 write_eh_inst(u8 *dst, u8 opcode, u64 arg) noexcept {
    if (opcode & dwarf::DWARF_CFI_PRIMARY_OPCODE_MASK) {
      assert((arg & dwarf::DWARF_CFI_PRIMARY_OPCODE_MASK) == 0);
      *dst = opcode | arg;
      return 1;
    }
    *dst++ = opcode;
    return 1 + util::uleb_write(dst, arg);
  }

  static constexpr u32
      write_eh_inst(u8 *dst, u8 opcode, u64 arg1, u64 arg2) noexcept {
    u8 *base = dst;
    dst += write_eh_inst(dst, opcode, arg1);
    dst += util::uleb_write(dst, arg2);
    return dst - base;
  }

  void eh_align_frame() noexcept;
  void eh_write_inst(u8 opcode) noexcept { this->eh_writer.write<u8>(opcode); }
  void eh_write_inst(u8 opcode, u64 arg) noexcept;
  void eh_write_inst(u8 opcode, u64 first_arg, u64 second_arg) noexcept;
  /// Write CFA_advance_loc; size must be scaled by code alignment factor.
  void eh_advance_raw(u64 size_units) noexcept;
  /// Write CFA_advance_loc with code alignment factor 1.
  void eh_advance(u64 size) noexcept { eh_advance_raw(size); }

private:
  void eh_init_cie(SymRef personality_func_addr = SymRef()) noexcept;

public:
  void eh_begin_fde(SymRef personality_func_addr = SymRef()) noexcept;
  void eh_end_fde() noexcept;

  /// @}

  /// \name Itanium Exception ABI
  /// @{

  void except_encode_func() noexcept;

  /// add an entry to the call-site table
  /// must be called in strictly increasing order wrt text_off
  void except_add_call_site(u32 text_off,
                            u32 len,
                            Label landing_pad,
                            bool is_cleanup) noexcept;

  /// Add a cleanup action to the action table
  /// *MUST* be the last one
  void except_add_cleanup_action() noexcept;

  /// add an action to the action table
  /// An invalid SymRef signals a catch(...)
  void except_add_action(bool first_action, SymRef type_sym) noexcept;

  void except_add_empty_spec_action(bool first_action) noexcept;

  u32 except_type_idx_for_sym(SymRef sym) noexcept;

  /// @}
};

/// Helper class to write function text.
template <typename Derived>
class FunctionWriter : public FunctionWriterBase {
protected:
  FunctionWriter(const TargetCIEInfo &cie_info) noexcept
      : FunctionWriterBase(cie_info) {}

  Derived *derived() noexcept { return static_cast<Derived *>(this); }

public:
  void begin_func(u32 align, u32 expected_size) noexcept {
    growth_size = expected_size;
    ensure_space(align + expected_size);
    derived()->align(align);
    FunctionWriterBase::begin_func();
  }

  void finish_func() noexcept { derived()->handle_fixups(); }

  /// \name Text Writing
  /// @{

  /// Ensure that at least size bytes are available.
  void ensure_space(size_t size) noexcept {
    assert(data_reserve_end >= data_cur);
    if (size_t(data_reserve_end - data_cur) < size) [[unlikely]] {
      derived()->more_space(size);
    }
  }

  template <std::integral T>
  void write_unchecked(T t) noexcept {
    assert(size_t(data_reserve_end - data_cur) >= sizeof(T));
    std::memcpy(data_cur, &t, sizeof(T));
    data_cur += sizeof(T);
  }

  template <std::integral T>
  void write(T t) noexcept {
    ensure_space(sizeof(T));
    write_unchecked<T>(t);
  }

  void align(size_t align) noexcept {
    assert(align > 0 && (align & (align - 1)) == 0);
    ensure_space(align);
    // permit optimization when align is a constant.
    std::memset(cur_ptr(), 0, align);
    data_cur = data_begin + util::align_up(offset(), align);
    section->align = std::max(section->align, u32(align));
  }

  /// @}
};

} // namespace tpde
