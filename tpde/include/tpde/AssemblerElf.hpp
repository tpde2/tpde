// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <cassert>
#include <elf.h>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "base.hpp"
#include "tpde/Assembler.hpp"
#include "tpde/StringTable.hpp"
#include "tpde/util/VectorWriter.hpp"
#include "util/SmallVector.hpp"
#include "util/misc.hpp"

namespace tpde {

namespace dwarf {
// DWARF constants
constexpr u8 DW_CFA_nop = 0;
constexpr u8 DW_EH_PE_uleb128 = 0x01;
constexpr u8 DW_EH_PE_pcrel = 0x10;
constexpr u8 DW_EH_PE_indirect = 0x80;
constexpr u8 DW_EH_PE_sdata4 = 0x0b;
constexpr u8 DW_EH_PE_omit = 0xff;

constexpr u8 DW_CFA_offset_extended = 0x05;
constexpr u8 DW_CFA_def_cfa = 0x0c;
constexpr u8 DW_CFA_def_cfa_register = 0x0d;
constexpr u8 DW_CFA_def_cfa_offset = 0x0e;
constexpr u8 DW_CFA_offset = 0x80;
constexpr u8 DW_CFA_advance_loc = 0x40;
constexpr u8 DW_CFA_advance_loc4 = 0x04;

constexpr u8 DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;

constexpr u32 EH_FDE_FUNC_START_OFF = 0x8;

namespace x64 {
constexpr u8 DW_reg_rax = 0;
constexpr u8 DW_reg_rdx = 1;
constexpr u8 DW_reg_rcx = 2;
constexpr u8 DW_reg_rbx = 3;
constexpr u8 DW_reg_rsi = 4;
constexpr u8 DW_reg_rdi = 5;
constexpr u8 DW_reg_rbp = 6;
constexpr u8 DW_reg_rsp = 7;
constexpr u8 DW_reg_r8 = 8;
constexpr u8 DW_reg_r9 = 9;
constexpr u8 DW_reg_r10 = 10;
constexpr u8 DW_reg_r11 = 11;
constexpr u8 DW_reg_r12 = 12;
constexpr u8 DW_reg_r13 = 13;
constexpr u8 DW_reg_r14 = 14;
constexpr u8 DW_reg_r15 = 15;
constexpr u8 DW_reg_ra = 16;
} // namespace x64

namespace a64 {
constexpr u8 DW_reg_x0 = 0;
constexpr u8 DW_reg_x1 = 1;
constexpr u8 DW_reg_x2 = 2;
constexpr u8 DW_reg_x3 = 3;
constexpr u8 DW_reg_x4 = 4;
constexpr u8 DW_reg_x5 = 5;
constexpr u8 DW_reg_x6 = 6;
constexpr u8 DW_reg_x7 = 7;
constexpr u8 DW_reg_x8 = 8;
constexpr u8 DW_reg_x9 = 9;
constexpr u8 DW_reg_x10 = 10;
constexpr u8 DW_reg_x11 = 11;
constexpr u8 DW_reg_x12 = 12;
constexpr u8 DW_reg_x13 = 13;
constexpr u8 DW_reg_x14 = 14;
constexpr u8 DW_reg_x15 = 15;
constexpr u8 DW_reg_x16 = 16;
constexpr u8 DW_reg_x17 = 17;
constexpr u8 DW_reg_x18 = 18;
constexpr u8 DW_reg_x19 = 19;
constexpr u8 DW_reg_x20 = 20;
constexpr u8 DW_reg_x21 = 21;
constexpr u8 DW_reg_x22 = 22;
constexpr u8 DW_reg_x23 = 23;
constexpr u8 DW_reg_x24 = 24;
constexpr u8 DW_reg_x25 = 25;
constexpr u8 DW_reg_x26 = 26;
constexpr u8 DW_reg_x27 = 27;
constexpr u8 DW_reg_x28 = 28;
constexpr u8 DW_reg_x29 = 29;
constexpr u8 DW_reg_x30 = 30;

constexpr u8 DW_reg_fp = 29;
constexpr u8 DW_reg_lr = 30;

constexpr u8 DW_reg_v0 = 64;
constexpr u8 DW_reg_v1 = 65;
constexpr u8 DW_reg_v2 = 66;
constexpr u8 DW_reg_v3 = 67;
constexpr u8 DW_reg_v4 = 68;
constexpr u8 DW_reg_v5 = 69;
constexpr u8 DW_reg_v6 = 70;
constexpr u8 DW_reg_v7 = 71;
constexpr u8 DW_reg_v8 = 72;
constexpr u8 DW_reg_v9 = 73;
constexpr u8 DW_reg_v10 = 74;
constexpr u8 DW_reg_v11 = 75;
constexpr u8 DW_reg_v12 = 76;
constexpr u8 DW_reg_v13 = 77;
constexpr u8 DW_reg_v14 = 78;
constexpr u8 DW_reg_v15 = 79;
constexpr u8 DW_reg_v16 = 80;
constexpr u8 DW_reg_v17 = 81;
constexpr u8 DW_reg_v18 = 82;
constexpr u8 DW_reg_v19 = 83;
constexpr u8 DW_reg_v20 = 84;
constexpr u8 DW_reg_v21 = 85;
constexpr u8 DW_reg_v22 = 86;
constexpr u8 DW_reg_v23 = 87;
constexpr u8 DW_reg_v24 = 88;
constexpr u8 DW_reg_v25 = 89;
constexpr u8 DW_reg_v26 = 90;
constexpr u8 DW_reg_v27 = 91;
constexpr u8 DW_reg_v28 = 92;
constexpr u8 DW_reg_v29 = 93;
constexpr u8 DW_reg_v30 = 94;

constexpr u8 DW_reg_sp = 31;
constexpr u8 DW_reg_pc = 32;
} // namespace a64

} // namespace dwarf

class AssemblerElf : public Assembler {
  friend class ElfMapper;

protected:
  struct TargetInfoElf : Assembler::TargetInfo {
    /// The OS ABI for the ELF header.
    u8 elf_osabi;
    /// The machine for the ELF header.
    u16 elf_machine;
  };

public:
  enum class SymVisibility : u8 {
    DEFAULT = STV_DEFAULT,
    INTERNAL = STV_INTERNAL,
    HIDDEN = STV_HIDDEN,
    PROTECTED = STV_PROTECTED,
  };

private:
  std::vector<Elf64_Sym> global_symbols, local_symbols;
  /// Section indices for large section numbers
  util::SmallVector<u32, 0> global_shndx, local_shndx;

  StringTable strtab;
  /// Storage for extra user-provided section names.
  StringTable shstrtab_extra;

public:
  util::VectorWriter eh_writer;

private:
  struct ExceptCallSiteInfo {
    /// Start offset *in section* (not inside function)
    u64 start;
    u64 len;
    u32 landing_pad_label;
    u32 action_entry;
  };

  /// Exception Handling temporary storage
  /// Call Sites for current function
  std::vector<ExceptCallSiteInfo> except_call_site_table;

  /// Temporary storage for encoding call sites
  util::SmallVector<u8> except_encoded_call_sites;
  /// Action Table for current function
  util::SmallVector<u8> except_action_table;
  /// The type_info table (contains the symbols which contain the pointers to
  /// the type_info)
  std::vector<SymRef> except_type_info_table;
  /// Table for exception specs
  std::vector<u8> except_spec_table;
  /// The current personality function (if any)
  SymRef cur_personality_func_addr;
  u32 eh_cur_cie_off = 0u;
  u32 eh_first_fde_off = 0;

  /// The current function
  SymRef cur_func;

public:
  explicit AssemblerElf(const TargetInfoElf &target_info)
      : Assembler(target_info) {
    local_symbols.resize(1); // First symbol must be null.
    init_sections();
    eh_init_cie();
  }

  void reset() noexcept override;

private:
  void init_sections() noexcept;

  std::span<Relocation> get_relocs(SecRef ref) {
    return get_section(ref).relocs;
  }

  /// Allocate a new section for relocations.
  [[nodiscard]] SecRef create_rela_section(SecRef ref,
                                           unsigned flags,
                                           unsigned rela_name) noexcept;

  [[nodiscard]] SymRef create_section_symbol(SecRef ref,
                                             std::string_view name) noexcept;

public:
  SecRef create_structor_section(bool init, SecRef group = SecRef()) noexcept;

  /// Create a new section with the given name, ELF section type, and flags.
  /// Optionally, a corresponding relocation (.rela) section is also created,
  /// otherwise, the section must not have relocations.
  [[nodiscard]] SecRef create_section(std::string_view name,
                                      unsigned type,
                                      unsigned flags,
                                      bool with_rela,
                                      SecRef group = SecRef()) noexcept;

  /// Create a new group section.
  [[nodiscard]] SecRef create_group_section(SymRef signature_sym,
                                            bool is_comdat) noexcept;

  const char *sec_name(SecRef ref) const noexcept;

private:
  bool sec_is_xindex(SecRef ref) const noexcept {
    return ref.id() >= SHN_LORESERVE;
  }

public:
  // Symbols

  void sym_copy(SymRef dst, SymRef src) noexcept;

private:
  [[nodiscard]] SymRef
      sym_add(std::string_view name, SymBinding binding, u32 type) noexcept;

public:
  [[nodiscard]] SymRef sym_add_undef(std::string_view name,
                                     SymBinding binding) noexcept override {
    return sym_add(name, binding, STT_NOTYPE);
  }

  [[nodiscard]] SymRef sym_predef_func(std::string_view name,
                                       SymBinding binding) noexcept override {
    return sym_add(name, binding, STT_FUNC);
  }

  [[nodiscard]] SymRef sym_predef_data(std::string_view name,
                                       SymBinding binding) noexcept override {
    return sym_add(name, binding, STT_OBJECT);
  }

  [[nodiscard]] SymRef sym_predef_tls(std::string_view name,
                                      SymBinding binding) noexcept override {
    return sym_add(name, binding, STT_TLS);
  }

private:
  /// Set symbol sections for SHN_XINDEX.
  void sym_def_xindex(SymRef sym_ref, SecRef sec_ref) noexcept;

public:
  void sym_def(SymRef sym_ref,
               SecRef sec_ref,
               u64 pos,
               u64 size) noexcept override {
    Elf64_Sym *sym = sym_ptr(sym_ref);
    assert(sym->st_shndx == SHN_UNDEF && "cannot redefined symbol");
    sym->st_value = pos;
    sym->st_size = size;
    if (!sec_is_xindex(sec_ref)) [[likely]] {
      sym->st_shndx = sec_ref.id();
    } else {
      sym->st_shndx = SHN_XINDEX;
      sym_def_xindex(sym_ref, sec_ref);
    }
    // TODO: handle fixups?
  }

  void sym_set_visibility(SymRef sym, SymVisibility visibility) noexcept {
    sym_ptr(sym)->st_other = static_cast<u8>(visibility);
  }

  const char *sym_name(SymRef sym) const noexcept {
    return strtab.data() + sym_ptr(sym)->st_name;
  }

  SecRef sym_section(SymRef sym) const noexcept {
    Elf64_Section shndx = sym_ptr(sym)->st_shndx;
    if (shndx < SHN_LORESERVE && shndx != SHN_UNDEF) [[likely]] {
      return SecRef(shndx);
    }
    assert(shndx == SHN_XINDEX);
    const auto &shndx_tab = sym_is_local(sym) ? local_shndx : global_shndx;
    return SecRef(shndx_tab[sym_idx(sym)]);
  }

private:
  [[nodiscard]] static bool sym_is_local(const SymRef sym) noexcept {
    return (sym.id() & 0x8000'0000) == 0;
  }

  [[nodiscard]] static u32 sym_idx(const SymRef sym) noexcept {
    return sym.id() & ~0x8000'0000;
  }

  [[nodiscard]] Elf64_Sym *sym_ptr(const SymRef sym) noexcept {
    if (sym_is_local(sym)) {
      return &local_symbols[sym_idx(sym)];
    } else {
      return &global_symbols[sym_idx(sym)];
    }
  }

  [[nodiscard]] const Elf64_Sym *sym_ptr(const SymRef sym) const noexcept {
    if (sym_is_local(sym)) {
      return &local_symbols[sym_idx(sym)];
    } else {
      return &global_symbols[sym_idx(sym)];
    }
  }

  // Unwind and exception info

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
  void eh_write_inst(u8 opcode, u64 arg) noexcept;
  void eh_write_inst(u8 opcode, u64 first_arg, u64 second_arg) noexcept;

private:
  void eh_init_cie(SymRef personality_func_addr = SymRef()) noexcept;

public:
  u32 eh_begin_fde(SymRef personality_func_addr = SymRef()) noexcept;
  void eh_end_fde(u32 fde_start, SymRef func) noexcept;

  void except_begin_func() noexcept;

  void except_encode_func(SymRef func_sym, const u32 *label_offsets) noexcept;

  /// add an entry to the call-site table
  /// must be called in strictly increasing order wrt text_off
  void except_add_call_site(u32 text_off,
                            u32 len,
                            u32 landing_pad_label,
                            bool is_cleanup) noexcept;

  /// Add a cleanup action to the action table
  /// *MUST* be the last one
  void except_add_cleanup_action() noexcept;

  /// add an action to the action table
  /// An invalid SymRef signals a catch(...)
  void except_add_action(bool first_action, SymRef type_sym) noexcept;

  void except_add_empty_spec_action(bool first_action) noexcept;

  u32 except_type_idx_for_sym(SymRef sym) noexcept;

  void finalize() noexcept override;

  // Output file generation

  std::vector<u8> build_object_file() noexcept override;
};

// TODO: Remove these types, instead find a good way to specify architecture as
// enum parameter (probably contained in Assembler?) to constructor.

class AssemblerElfA64 final : public AssemblerElf {
  static const TargetInfoElf TARGET_INFO;

public:
  explicit AssemblerElfA64() noexcept : AssemblerElf(TARGET_INFO) {}
};

class AssemblerElfX64 final : public AssemblerElf {
  static const TargetInfoElf TARGET_INFO;

public:
  explicit AssemblerElfX64() noexcept : AssemblerElf(TARGET_INFO) {}
};

} // namespace tpde
