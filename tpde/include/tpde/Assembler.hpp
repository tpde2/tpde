// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "tpde/base.hpp"
#include "tpde/util/BumpAllocator.hpp"
#include "tpde/util/SmallVector.hpp"
#include <cstring>
#include <span>
#include <vector>

namespace tpde {

struct SymRef {
private:
  u32 val;

public:
  /// Invalid symbol reference
  constexpr SymRef() : val(0) {}

  explicit constexpr SymRef(u32 id) : val(id) {}

  u32 id() const { return val; }
  bool valid() const { return val != 0; }

  bool operator==(const SymRef &other) const { return other.val == val; }
};

struct SecRef {
private:
  u32 val;

public:
  /// Invalid symbol reference
  constexpr SecRef() : val(0) {}

  explicit constexpr SecRef(u32 id) : val(id) {}

  u32 id() const { return val; }
  bool valid() const { return val != 0; }

  bool operator==(const SecRef &other) const { return other.val == val; }
};

struct Relocation {
  u32 offset;    ///< Offset inside section.
  SymRef symbol; ///< References symbol.
  u32 type;      ///< Relocation type. File-format-specifc.
  i32 addend;    ///< Addend.
};

/// Section kinds, lowered to file-format specific flags.
enum class SectionKind : u8 {
  Text,       ///< Text section, executable code (ELF .text)
  ReadOnly,   ///< Read-only data section (ELF .rodata)
  EHFrame,    ///< EH Frame section (ELF .eh_frame)
  LSDA,       ///< LSDA section (ELF .gcc_except_table)
  Data,       ///< Writable data section (ELF .data)
  DataRelRO,  ///< Read-only data section with relocations (ELF .data.rel.ro)
  BSS,        ///< Zero-initialized data section (ELF .bss)
  ThreadData, ///< Initialized thread-local data section (ELF .tdata)
  ThreadBSS,  ///< Zero-initialized thread-local data section (ELF .tbss)

  Max
};

struct DataSection {
  friend class Assembler;
  friend class AssemblerElf;

  /// 256 bytes inline storage is enough for 10 relocations, which is a typical
  /// number for a single function (relevant for COMDAT sections with one
  /// section per function).
  using StorageTy = util::SmallVector<u8, 256>;

  /// Section data.
  StorageTy data;

  u64 addr = 0;  ///< Address (file-format-specific).
  u64 vsize = 0; ///< Size of virtual section, otherwise data.size() is valid.
  u32 type = 0;  ///< Type (file-format-specific).
  u32 flags = 0; ///< Flags (file-format-specific).
  u32 name = 0;  ///< Name (file-format-specific, can also be index, etc.).
  u32 align = 1; ///< Alignment (bytes).

private:
  /// Section symbol, or signature symbol for SHT_GROUP sections.
  SymRef sym = {};

  SecRef sec_ref;

  util::SmallVector<Relocation, 4> relocs;

public:
  /// Generic field for target-specific data.
  void *target_info = nullptr;

  /// Whether the section is virtual, i.e. has no content.
  bool is_virtual;

private:
  /// Whether the section can have relocations. For ELF, this implies that the
  /// immediately following section ID is reserved as relocation section and
  /// that name-5..name is ".rela".
  bool has_relocs;

public:
#ifndef NDEBUG
  /// Whether the section is currently in use by a SectionWriter.
  bool locked = false;
#endif

  DataSection(SecRef ref) noexcept : sec_ref(ref) {}

  SecRef get_ref() const noexcept { return sec_ref; }

  size_t size() const { return is_virtual ? vsize : data.size(); }

  template <typename T>
  void write(const T &t) noexcept {
    assert(!locked);
    assert(!is_virtual);
    size_t off = data.size();
    data.resize_uninitialized(data.size() + sizeof(T));
    std::memcpy(data.data() + off, &t, sizeof(T));
  }
};

/// Assembler base class.
class Assembler {
public:
  enum class SymBinding : u8 {
    /// Symbol with local linkage, must be defined
    LOCAL,
    /// Weak linkage
    WEAK,
    /// Global linkage
    GLOBAL,
  };

  struct TargetInfo {
    struct SectionFlags {
      u32 type;
      u32 flags;
      u32 name;
      u8 align = 1;
      bool has_relocs = true;
      bool is_bss = false;
    };

    /// The relocation type for 32-bit pc-relative offsets.
    u32 reloc_pc32;
    /// The relocation type for 64-bit absolute addresses.
    u32 reloc_abs64;

    /// Section flags for the different section kinds.
    std::array<SectionFlags, unsigned(SectionKind::Max)> section_flags;
  };

protected:
  const TargetInfo &target_info;

  util::BumpAllocator<> section_allocator;
  util::SmallVector<util::BumpAllocUniquePtr<DataSection>, 16> sections;

  std::array<SecRef, unsigned(SectionKind::Max)> default_sections;

  Assembler(const TargetInfo &target_info) noexcept
      : target_info(target_info) {}
  virtual ~Assembler();

public:
  virtual void reset() noexcept;

  /// \name Sections
  /// @{

  DataSection &get_section(SecRef ref) noexcept {
    assert(ref.valid());
    return *sections[ref.id()];
  }

  const DataSection &get_section(SecRef ref) const noexcept {
    assert(ref.valid());
    return *sections[ref.id()];
  }

  SecRef create_section(const TargetInfo::SectionFlags &flags) noexcept;

  SecRef create_section(SectionKind kind) noexcept {
    return create_section(target_info.section_flags[unsigned(kind)]);
  }

  SecRef get_default_section(SectionKind kind) noexcept {
    SecRef &res = default_sections[unsigned(kind)];
    if (!res.valid()) {
      res = create_section(kind);
    }
    return res;
  }

  SecRef get_text_section() noexcept {
    return get_default_section(SectionKind::Text);
  }
  SecRef get_data_section(bool rodata, bool relro = false) noexcept {
    return get_default_section(!rodata ? SectionKind::Data
                               : relro ? SectionKind::DataRelRO
                                       : SectionKind::ReadOnly);
  }
  SecRef get_bss_section() noexcept {
    return get_default_section(SectionKind::BSS);
  }
  SecRef get_tdata_section() noexcept {
    return get_default_section(SectionKind::ThreadData);
  }
  SecRef get_tbss_section() noexcept {
    return get_default_section(SectionKind::ThreadBSS);
  }

  virtual void rename_section(SecRef, std::string_view name) noexcept = 0;

  virtual SymRef section_symbol(SecRef) noexcept = 0;

  /// @}

  virtual SymRef sym_add_undef(std::string_view, SymBinding) noexcept = 0;
  virtual SymRef sym_predef_func(std::string_view, SymBinding) noexcept = 0;
  virtual SymRef sym_predef_data(std::string_view, SymBinding) noexcept = 0;
  virtual SymRef sym_predef_tls(std::string_view, SymBinding) noexcept = 0;
  /// Define a symbol at the specified location.
  virtual void sym_def(SymRef, SecRef, u64 pos, u64 size) noexcept = 0;

  /// Define predefined symbol with the specified data.
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

  /// Define predefined symbol with zero; also supported for BSS sections.
  void sym_def_predef_zero(SecRef sec_ref,
                           SymRef sym_ref,
                           u32 size,
                           u32 align,
                           u32 *off = nullptr) noexcept;


  /// \name Relocations
  /// @{

  /// Add relocation. Type is file-format and target-specific.
  void reloc_sec(
      SecRef sec, SymRef sym, u32 type, u32 offset, i64 addend) noexcept {
    assert(i32(addend) == addend && "non-32-bit addends are unsupported");
    get_section(sec).relocs.emplace_back(offset, sym, type, addend);
  }

  void reloc_pc32(SecRef sec, SymRef sym, u32 offset, i64 addend) noexcept {
    reloc_sec(sec, sym, target_info.reloc_pc32, offset, addend);
  }

  void reloc_abs(SecRef sec, SymRef sym, u32 offset, i64 addend) noexcept {
    reloc_sec(sec, sym, target_info.reloc_abs64, offset, addend);
  }

  /// @}

  virtual void finalize() noexcept {}

  virtual std::vector<u8> build_object_file() noexcept = 0;
};

} // namespace tpde

#undef ARG
