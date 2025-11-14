// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <cassert>
#include <span>
#include <string_view>
#include <vector>

#include "base.hpp"
#include "tpde/Assembler.hpp"
#include "tpde/ELF.hpp"
#include "tpde/StringTable.hpp"
#include "util/SmallVector.hpp"

namespace tpde::elf {

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
  explicit AssemblerElf(const TargetInfoElf &target_info)
      : Assembler(target_info) {
    local_symbols.resize(1); // First symbol must be null.
    init_sections();
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

  SymRef section_symbol(SecRef) noexcept override;

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
    auto shndx = sym_ptr(sym)->st_shndx;
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

public:
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

} // namespace tpde::elf
