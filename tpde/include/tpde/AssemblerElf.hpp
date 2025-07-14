// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <cassert>
#include <cstdlib>
#include <elf.h>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "base.hpp"
#include "tpde/Assembler.hpp"
#include "tpde/StringTable.hpp"
#include "tpde/util/BumpAllocator.hpp"
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

struct AssemblerElfBase {
  template <class Derived>
  friend struct AssemblerElf;
  friend class ElfMapper;

  struct TargetInfo {
    /// The OS ABI for the ELF header.
    u8 elf_osabi;
    /// The machine for the ELF header.
    u16 elf_machine;

    /// The return address register for the CIE.
    u8 cie_return_addr_register;
    /// The initial instructions for the CIE.
    std::span<const u8> cie_instrs;
    /// Code alignment factor for the CIE, ULEB128, must be one byte.
    u8 cie_code_alignment_factor;
    /// Data alignment factor for the CIE, SLEB128, must be one byte.
    u8 cie_data_alignment_factor;

    /// The relocation type for 32-bit pc-relative offsets.
    u32 reloc_pc32;
    /// The relocation type for 64-bit absolute addresses.
    u32 reloc_abs64;
  };

  enum class SymBinding : u8 {
    /// Symbol with local linkage, must be defined
    LOCAL,
    /// Weak linkage
    WEAK,
    /// Global linkage
    GLOBAL,
  };

  enum class SymVisibility : u8 {
    DEFAULT = STV_DEFAULT,
    INTERNAL = STV_INTERNAL,
    HIDDEN = STV_HIDDEN,
    PROTECTED = STV_PROTECTED,
  };

  // TODO: merge Label with SymRef when adding private symbols
  enum class Label : u32 {
  };

  template <typename Derived>
  class SectionWriterBase {
  protected:
    DataSection *section = nullptr;
    u8 *data_begin = nullptr;
    u8 *data_cur = nullptr;
    u8 *data_reserve_end = nullptr;

  public:
    /// Growth size for more_space; adjusted exponentially after every grow.
    u32 growth_size = 0x10000;

    SectionWriterBase() noexcept = default;

    ~SectionWriterBase() {
      assert(data_cur == data_reserve_end &&
             "must flush section writer before destructing");
    }

  protected:
    Derived *derived() noexcept { return static_cast<Derived *>(this); }

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

    void ensure_space(size_t size) noexcept {
      assert(data_reserve_end >= data_cur);
      if (size_t(data_reserve_end - data_cur) < size) [[unlikely]] {
        derived()->more_space(size);
      }
    }

    void more_space(size_t size) noexcept;

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

    void flush() noexcept {
      if (data_cur != data_reserve_end) {
        section->data.resize(offset());
        data_reserve_end = data_cur;
#ifndef NDEBUG
        section->locked = false;
#endif
      }
    }

    void align(size_t align) noexcept {
      assert(align > 0 && (align & (align - 1)) == 0);
      ensure_space(align);
      // permit optimization when align is a constant.
      std::memset(cur_ptr(), 0, align);
      data_cur = data_begin + util::align_up(offset(), align);
      section->align = std::max(section->align, u32(align));
    }
  };

private:
  const TargetInfo &target_info;

  util::BumpAllocator<> section_allocator;
  util::SmallVector<util::BumpAllocUniquePtr<DataSection>, 16> sections;

  std::vector<Elf64_Sym> global_symbols, local_symbols;
  /// Section indices for large section numbers
  util::SmallVector<u32, 0> global_shndx, local_shndx;

protected:
  struct TempSymbolInfo {
    /// Section, or invalid if pending
    SecRef section;
    /// Offset into section, or index into temp_symbol_fixups if pending
    union {
      u32 fixup_idx;
      u32 off;
    };
  };

  struct TempSymbolFixup {
    SecRef section;
    u32 next_list_entry;
    u32 off;
    u8 kind;
  };

private:
  std::vector<TempSymbolInfo> temp_symbols;
  std::vector<TempSymbolFixup> temp_symbol_fixups;
  u32 next_free_tsfixup = ~0u;

  StringTable strtab;
  /// Storage for extra user-provided section names.
  StringTable shstrtab_extra;

protected:
  SecRef secref_text = SecRef();
  SecRef secref_rodata = SecRef();
  SecRef secref_relro = SecRef();
  SecRef secref_data = SecRef();
  SecRef secref_bss = SecRef();
  SecRef secref_tdata = SecRef();
  SecRef secref_tbss = SecRef();

  /// Unwind Info
  SecRef secref_eh_frame = SecRef();
  SecRef secref_except_table = SecRef();

public:
  util::VectorWriter eh_writer;

private:
  struct ExceptCallSiteInfo {
    /// Start offset *in section* (not inside function)
    u64 start;
    u64 len;
    Label landing_pad;
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
  explicit AssemblerElfBase(const TargetInfo &target_info)
      : target_info(target_info) {
    local_symbols.resize(1); // First symbol must be null.
    init_sections();
    eh_init_cie();
  }

  void reset() noexcept;

  // Sections

  DataSection &get_section(SecRef ref) noexcept {
    assert(ref.valid());
    return *sections[ref.id()];
  }

  const DataSection &get_section(SecRef ref) const noexcept {
    assert(ref.valid());
    return *sections[ref.id()];
  }

private:
  void init_sections() noexcept;

  std::span<Relocation> get_relocs(SecRef ref) {
    return get_section(ref).relocs;
  }

  /// Allocate a new section.
  [[nodiscard]] SecRef
      create_section(unsigned type, unsigned flags, unsigned name) noexcept;

  /// Allocate a new section for relocations.
  [[nodiscard]] SecRef create_rela_section(SecRef ref,
                                           unsigned flags,
                                           unsigned rela_name) noexcept;

  [[nodiscard]] SymRef create_section_symbol(SecRef ref,
                                             std::string_view name) noexcept;

  DataSection &get_or_create_section(SecRef &ref,
                                     unsigned rela_name,
                                     unsigned type,
                                     unsigned flags,
                                     unsigned align,
                                     bool with_rela = true) noexcept;

public:
  SecRef get_text_section() noexcept { return secref_text; }
  SecRef get_data_section(bool rodata, bool relro = false) noexcept;
  SecRef get_bss_section() noexcept;
  SecRef get_tdata_section() noexcept;
  SecRef get_tbss_section() noexcept;
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
                                     SymBinding binding) noexcept {
    return sym_add(name, binding, STT_NOTYPE);
  }

  [[nodiscard]] SymRef sym_predef_func(std::string_view name,
                                       SymBinding binding) noexcept {
    return sym_add(name, binding, STT_FUNC);
  }

  [[nodiscard]] SymRef sym_predef_data(std::string_view name,
                                       SymBinding binding) noexcept {
    return sym_add(name, binding, STT_OBJECT);
  }

  [[nodiscard]] SymRef sym_predef_tls(std::string_view name,
                                      SymBinding binding) noexcept {
    return sym_add(name, binding, STT_TLS);
  }

  void sym_def_predef_data(SecRef sec,
                           SymRef sym,
                           std::span<const u8> data,
                           u32 align,
                           u32 *off) noexcept;

  [[nodiscard]] SymRef sym_def_data(SecRef sec,
                                    std::string_view name,
                                    std::span<const u8> data,
                                    u32 align,
                                    SymBinding binding,
                                    u32 *off = nullptr) {
    SymRef sym = sym_predef_data(name, binding);
    sym_def_predef_data(sec, sym, data, align, off);
    return sym;
  }

  void sym_def_predef_zero(SecRef sec_ref,
                           SymRef sym_ref,
                           u32 size,
                           u32 align,
                           u32 *off = nullptr) noexcept;

private:
  /// Set symbol sections for SHN_XINDEX.
  void sym_def_xindex(SymRef sym_ref, SecRef sec_ref) noexcept;

public:
  void sym_def(SymRef sym_ref, SecRef sec_ref, u64 pos, u64 size) noexcept {
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

  /// Forcefully set value of symbol, doesn't change section.
  void sym_set_value(SymRef sym, u64 value) noexcept {
    sym_ptr(sym)->st_value = value;
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

  Label label_create() noexcept {
    const Label label = static_cast<Label>(temp_symbols.size());
    temp_symbols.push_back(TempSymbolInfo{SecRef(), {.fixup_idx = ~0u}});
    return label;
  }

  // TODO: return pair of section, offset
  u32 label_is_pending(Label label) const noexcept {
    const auto &info = temp_symbols[static_cast<u32>(label)];
    return !info.section.valid();
  }

  // TODO: return pair of section, offset
  u32 label_offset(Label label) const noexcept {
    assert(!label_is_pending(label));
    const auto &info = temp_symbols[static_cast<u32>(label)];
    return info.off;
  }

protected:
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

  // Relocations

public:
  void reloc_sec(
      SecRef sec, SymRef sym, u32 type, u64 offset, i64 addend) noexcept;

  void reloc_pc32(SecRef sec, SymRef sym, u64 offset, i64 addend) noexcept {
    reloc_sec(sec, sym, target_info.reloc_pc32, offset, addend);
  }

  void reloc_abs(SecRef sec, SymRef sym, u64 offset, i64 addend) noexcept {
    reloc_sec(sec, sym, target_info.reloc_abs64, offset, addend);
  }

  void reloc_sec(SecRef sec, Label label, u8 kind, u32 offset) noexcept;

  // Unwind and exception info

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

  void except_encode_func(SymRef func_sym) noexcept;

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

  void finalize() noexcept;

  // Output file generation

  std::vector<u8> build_object_file() noexcept;
};

template <typename Derived>
void AssemblerElfBase::SectionWriterBase<Derived>::more_space(
    size_t size) noexcept {
  size_t cur_size = section->data.size();
  size_t new_size;
  if (cur_size + size <= section->data.capacity()) {
    new_size = section->data.capacity();
  } else {
    new_size = cur_size + (size <= growth_size ? growth_size : size);

    // Grow by factor 1.5
    growth_size = growth_size + (growth_size >> 1);
    // Max 16 MiB per grow.
    growth_size = growth_size < 0x1000000 ? growth_size : 0x1000000;
  }

  const size_t off = offset();
  section->data.resize_uninitialized(new_size);
#ifndef NDEBUG
  thread_local uint8_t rand = 1;
  std::memset(section->data.data() + off, rand += 2, new_size - off);
  section->locked = true;
#endif

  data_begin = section->data.data();
  data_cur = data_begin + off;
  data_reserve_end = data_begin + section->data.size();
}

/// AssemblerElf contains the architecture-independent logic to emit
/// ELF object files (currently linux-specific) which is then extended by
/// AssemblerElfX64 or AssemblerElfA64
template <typename Derived>
struct AssemblerElf : public AssemblerElfBase {
  /// The current write pointer for the text section
  explicit AssemblerElf() : AssemblerElfBase(Derived::TARGET_INFO) {
    static_assert(std::is_base_of_v<AssemblerElf, Derived>);
  }

  Derived *derived() noexcept { return static_cast<Derived *>(this); }

  void label_place(Label label, SecRef sec, u32 off) noexcept;
};

template <typename Derived>
void AssemblerElf<Derived>::label_place(Label label,
                                        SecRef sec,
                                        u32 offset) noexcept {
  assert(label_is_pending(label));
  TempSymbolInfo &info = temp_symbols[static_cast<u32>(label)];
  u32 fixup_idx = info.fixup_idx;
  info.section = sec;
  info.off = offset;

  while (fixup_idx != ~0u) {
    TempSymbolFixup &fixup = temp_symbol_fixups[fixup_idx];
    derived()->handle_fixup(info, fixup);
    auto next = fixup.next_list_entry;
    fixup.next_list_entry = next_free_tsfixup;
    next_free_tsfixup = fixup_idx;
    fixup_idx = next;
  }
}

} // namespace tpde
