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
#include "tpde/DWARF.hpp"
#include "tpde/StringTable.hpp"
#include "tpde/util/VectorWriter.hpp"
#include "util/SmallVector.hpp"
#include "util/misc.hpp"

namespace tpde {

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

  [[nodiscard]] SymRef create_section_symbol(SecRef ref,
                                             std::string_view name) noexcept;

public:
  SecRef create_structor_section(bool init, SecRef group = SecRef()) noexcept;

  void rename_section(SecRef, std::string_view) noexcept override;

  /// Create a new group section.
  [[nodiscard]] SecRef create_group_section(SymRef signature_sym,
                                            bool is_comdat) noexcept;

  /// Add a section to a section group.
  void add_to_group(SecRef group_ref, SecRef sec_ref) noexcept;

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
