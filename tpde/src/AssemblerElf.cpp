// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/AssemblerElf.hpp"
#include "tpde/Assembler.hpp"
#include "tpde/ELF.hpp"
#include "tpde/StringTable.hpp"
#include "tpde/util/misc.hpp"
#include <span>
#include <string_view>

namespace tpde::elf {

// TODO(ts): maybe just outsource this to a helper func that can live in a cpp
// file?
// TODO(ts): this is linux-specific, no?
constexpr static std::span<const char> SECTION_NAMES = {
    "\0" // first section is the null-section
    ".note.GNU-stack\0"
    ".symtab\0"
    ".strtab\0"
    ".shstrtab\0"};

// TODO(ts): this is linux-specific, no?
constexpr static std::span<const char> SHSTRTAB = {
    "\0" // first section is the null-section
    ".note.GNU-stack\0"
    ".rela.eh_frame\0"
    ".symtab\0"
    ".strtab\0"
    ".shstrtab\0"
    ".bss\0"
    ".tbss\0"
    ".rela.rodata\0"
    ".rela.text\0"
    ".rela.data.rel.ro\0"
    ".rela.data\0"
    ".rela.tdata\0"
    ".rela.gcc_except_table\0"
    ".rela.init_array\0"
    ".rela.fini_array\0"
    ".group\0"
    ".symtab_shndx\0"};

void fail_constexpr_compile(const char *) {}

consteval static u32 sec_idx(const std::string_view name) {
  // skip the first null string
  const char *data = SECTION_NAMES.data() + 1;
  u32 idx = 1;
  auto sec_name = std::string_view{data};
  while (!sec_name.empty()) {
    if (sec_name == name) {
      return idx;
    }

    ++idx;
    data += sec_name.size() + 1;
    sec_name = std::string_view{data};
  }

  fail_constexpr_compile("unknown section name");
  return ~0u;
}

consteval static u32 sec_off(const char *name) {
  std::string_view tab(SHSTRTAB.data(), SHSTRTAB.size());
  size_t pos = tab.find(name, 1, std::string_view(name).size() + 1);
  if (pos == std::string_view::npos) {
    fail_constexpr_compile("unknown section name");
  }
  return pos;
}

consteval static u32 predef_sec_count() {
  // skip the first null string
  const char *data = SECTION_NAMES.data() + 1;
  u32 idx = 1;
  auto sec_name = std::string_view{data};
  while (!sec_name.empty()) {
    ++idx;
    data += sec_name.size() + 1;
    sec_name = std::string_view{data};
  }

  return idx;
}

void AssemblerElf::reset() noexcept {
  Assembler::reset();

  global_symbols.clear();
  local_symbols.resize(1); // first symbol must be null
  strtab = StringTable();
  shstrtab_extra = StringTable();

  init_sections();
}

void AssemblerElf::rename_section(SecRef ref, std::string_view name) noexcept {
  DataSection &sec = get_section(ref);
  // This is possible, just not implemented. But maybe there's no requirement at
  // all that section symbols are named.
  assert(!sec.sym.valid() && "cannot rename after section symbol is created");
  size_t rela_name = SHSTRTAB.size();
  rela_name += shstrtab_extra.add_prefix(sec.has_relocs ? ".rela" : "", name);
  sec.name = rela_name + (sec.has_relocs ? 5 : 0);
}

SymRef AssemblerElf::section_symbol(SecRef ref) noexcept {
  SymRef &sym = get_section(ref).sym;
  if (!sym.valid()) {
    u16 shndx = sec_is_xindex(ref) ? u16(SHN_XINDEX) : ref.id();

    sym = SymRef(local_symbols.size());
    local_symbols.push_back(Elf64_Sym{
        .st_name = 0, // TODO: proper name?
        .st_info = ELF64_ST_INFO(STB_LOCAL, STT_SECTION),
        .st_other = STV_DEFAULT,
        .st_shndx = shndx,
        .st_value = 0,
        .st_size = 0,
    });
    if (sec_is_xindex(ref)) {
      sym_def_xindex(sym, ref);
    }
  }
  return sym;
}

const char *AssemblerElf::sec_name(SecRef ref) const noexcept {
  const DataSection &sec = get_section(ref);
  assert(sec.name < SHSTRTAB.size());
  return SHSTRTAB.data() + sec.name;
}

SecRef AssemblerElf::create_structor_section(bool init, SecRef group) noexcept {
  // TODO: priorities
  TargetInfo::SectionFlags flags{
      .type = u32(init ? SHT_INIT_ARRAY : SHT_FINI_ARRAY),
      .flags = SHF_ALLOC | SHF_WRITE,
      .name = 0,
      .align = 8};
  SecRef secref = create_section(flags);
  rename_section(secref, init ? ".init_array" : ".fini_array");
  if (group.valid()) {
    add_to_group(group, secref);
  }
  return secref;
}

SecRef AssemblerElf::create_group_section(SymRef signature_sym,
                                          bool is_comdat) noexcept {
  TargetInfo::SectionFlags flags{.type = SHT_GROUP,
                                 .flags = 0,
                                 .name = sec_off(".group"),
                                 .align = 4,
                                 .has_relocs = false};
  SecRef ref = create_section(flags);
  DataSection &sec = get_section(ref);
  sec.sym = signature_sym;
  // Group flags.
  sec.write<u32>(is_comdat ? u32(GRP_COMDAT) : 0);
  return ref;
}

void AssemblerElf::add_to_group(SecRef group_ref, SecRef sec_ref) noexcept {
  DataSection &sec = get_section(sec_ref);
  assert(!(sec.flags & SHF_GROUP) && "section must be in at most one group");
  sec.flags |= SHF_GROUP;
  DataSection &group = get_section(group_ref);
  assert(group.type == SHT_GROUP);
  group.write<u32>(sec_ref.id());
  if (sec.has_relocs) {
    group.write<u32>(sec_ref.id() + 1);
  }
}

void AssemblerElf::init_sections() noexcept {
  for (size_t i = 0; i < predef_sec_count(); i++) {
    sections.emplace_back(nullptr);
  }
}

void AssemblerElf::sym_copy(SymRef dst, SymRef src) noexcept {
  Elf64_Sym *src_ptr = sym_ptr(src), *dst_ptr = sym_ptr(dst);

  dst_ptr->st_shndx = src_ptr->st_shndx;
  if (src_ptr->st_shndx == SHN_XINDEX) {
    sym_def_xindex(dst, sym_section(src));
  }
  dst_ptr->st_size = src_ptr->st_size;
  dst_ptr->st_value = src_ptr->st_value;
  // Don't copy st_info.
}

SymRef AssemblerElf::sym_add(const std::string_view name,
                             SymBinding binding,
                             u32 type) noexcept {
  size_t str_off = strtab.add(name);

  u8 info;
  switch (binding) {
    using enum AssemblerElf::SymBinding;
  case LOCAL: info = ELF64_ST_INFO(STB_LOCAL, type); break;
  case WEAK: info = ELF64_ST_INFO(STB_WEAK, type); break;
  case GLOBAL: info = ELF64_ST_INFO(STB_GLOBAL, type); break;
  default: TPDE_UNREACHABLE("invalid symbol binding");
  }
  auto sym = Elf64_Sym{.st_name = static_cast<u32>(str_off),
                       .st_info = info,
                       .st_other = STV_DEFAULT,
                       .st_shndx = SHN_UNDEF,
                       .st_value = 0,
                       .st_size = 0};

  if (binding == SymBinding::LOCAL) {
    local_symbols.push_back(sym);
    assert(local_symbols.size() < 0x8000'0000);
    return SymRef(local_symbols.size() - 1);
  } else {
    global_symbols.push_back(sym);
    assert(global_symbols.size() < 0x8000'0000);
    return SymRef((global_symbols.size() - 1) | 0x8000'0000);
  }
}

void AssemblerElf::sym_def_xindex(SymRef sym_ref, SecRef sec_ref) noexcept {
  assert(sec_is_xindex(sec_ref));
  auto &shndx = sym_is_local(sym_ref) ? local_shndx : global_shndx;
  if (shndx.size() <= sym_idx(sym_ref)) {
    shndx.resize(sym_idx(sym_ref) + 1);
  }
  shndx[sym_idx(sym_ref)] = sec_ref.id();
}

std::vector<u8> AssemblerElf::build_object_file() noexcept {
  using namespace elf;

  auto target_info = static_cast<const TargetInfoElf &>(this->target_info);

  std::vector<u8> out{};

  unsigned secidx_symtax_shndx = 0;

  uint32_t sym_count = local_symbols.size() + global_symbols.size();
  uint32_t sec_count = sections.size();
  if (sec_count >= SHN_LORESERVE) {
    sec_count += 1;
    secidx_symtax_shndx = sections.size();
  } else {
    assert(local_shndx.empty() && global_shndx.empty());
  }

  u32 obj_size = sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr) * sec_count + 16;
  obj_size += sizeof(Elf64_Sym) * sym_count;
  obj_size += strtab.size();
  obj_size += SHSTRTAB.size() + shstrtab_extra.size();
  for (const auto &sec : sections) {
    if (!sec) { // skip relocation sections
      continue;
    }
    obj_size += sec->data.size() + sizeof(Elf64_Rela) * sec->relocs.size() + 16;
  }
  if (secidx_symtax_shndx != 0) {
    obj_size += sizeof(uint32_t) * sym_count;
  }
  out.reserve(obj_size);

  out.resize(sizeof(Elf64_Ehdr));

  const auto shdr_off = out.size();
  out.resize(out.size() + sizeof(Elf64_Shdr) * sec_count);

  const auto sec_hdr = [shdr_off, &out](const u32 idx) {
    return reinterpret_cast<Elf64_Shdr *>(out.data() + shdr_off) + idx;
  };

  {
    auto *hdr = reinterpret_cast<Elf64_Ehdr *>(out.data());

    hdr->e_ident[0] = ELFMAG[0];
    hdr->e_ident[1] = ELFMAG[1];
    hdr->e_ident[2] = ELFMAG[2];
    hdr->e_ident[3] = ELFMAG[3];
    hdr->e_ident[4] = ELFCLASS64;
    hdr->e_ident[5] = ELFDATA2LSB;
    hdr->e_ident[6] = EV_CURRENT;
    hdr->e_ident[7] = target_info.elf_osabi;
    hdr->e_ident[8] = 0;
    hdr->e_type = ET_REL;
    hdr->e_machine = target_info.elf_machine;
    hdr->e_version = EV_CURRENT;
    hdr->e_shoff = shdr_off;
    hdr->e_ehsize = sizeof(Elf64_Ehdr);
    hdr->e_shentsize = sizeof(Elf64_Shdr);
    if (sec_count < SHN_LORESERVE) {
      hdr->e_shnum = sec_count;
    } else {
      // If e_shnum is too small, the number of sections is stored in the size
      // field of the NULL section entry.
      hdr->e_shnum = 0;
      sec_hdr(0)->sh_size = sec_count;
    }
    hdr->e_shstrndx = sec_idx(".shstrtab");
  }

  // .note.GNU-stack
  {
    auto *hdr = sec_hdr(sec_idx(".note.GNU-stack"));
    hdr->sh_name = sec_off(".note.GNU-stack");
    hdr->sh_type = SHT_PROGBITS;
    hdr->sh_offset = out.size(); // gcc seems to give empty sections an offset
    hdr->sh_addralign = 1;
  }

  // .symtab
  {
    const size_t local_cnt = local_symbols.size();
    const size_t global_cnt = global_symbols.size();
    const auto size =
        sizeof(Elf64_Sym) * (local_symbols.size() + global_symbols.size());
    const auto sh_off = out.size();
    out.insert(out.end(),
               reinterpret_cast<uint8_t *>(local_symbols.data()),
               reinterpret_cast<uint8_t *>(local_symbols.data() + local_cnt));
    // global symbols need to come after the local symbols
    out.insert(out.end(),
               reinterpret_cast<uint8_t *>(global_symbols.data()),
               reinterpret_cast<uint8_t *>(global_symbols.data() + global_cnt));

    auto *hdr = sec_hdr(sec_idx(".symtab"));
    hdr->sh_name = sec_off(".symtab");
    hdr->sh_type = SHT_SYMTAB;
    hdr->sh_offset = sh_off;
    hdr->sh_size = size;
    hdr->sh_link = sec_idx(".strtab");
    hdr->sh_info = local_symbols.size(); // first non-local symbol idx
    hdr->sh_addralign = 8;
    hdr->sh_entsize = sizeof(Elf64_Sym);
  }

  // .strtab
  {
    const auto size = util::align_up(strtab.size(), 8);
    const auto pad = size - strtab.size();
    const auto sh_off = out.size();
    out.insert(out.end(),
               reinterpret_cast<const u8 *>(strtab.data()),
               reinterpret_cast<const u8 *>(strtab.data() + strtab.size()));
    out.resize(out.size() + pad);

    auto *hdr = sec_hdr(sec_idx(".strtab"));
    hdr->sh_name = sec_off(".strtab");
    hdr->sh_type = SHT_STRTAB;
    hdr->sh_offset = sh_off;
    hdr->sh_size = size;
    hdr->sh_addralign = 1;
  }

  // .shstrtab
  {
    const auto size = SHSTRTAB.size() + shstrtab_extra.size();
    const auto pad = util::align_up(size, 8) - size;
    const auto sh_off = out.size();
    out.insert(out.end(),
               reinterpret_cast<const u8 *>(SHSTRTAB.data()),
               reinterpret_cast<const u8 *>(SHSTRTAB.data() + SHSTRTAB.size()));
    out.insert(out.end(),
               reinterpret_cast<const u8 *>(shstrtab_extra.data()),
               reinterpret_cast<const u8 *>(shstrtab_extra.data() +
                                            shstrtab_extra.size()));
    out.resize(out.size() + pad);

    auto *hdr = sec_hdr(sec_idx(".shstrtab"));
    hdr->sh_name = sec_off(".shstrtab");
    hdr->sh_type = SHT_STRTAB;
    hdr->sh_offset = sh_off;
    hdr->sh_size = size;
    hdr->sh_addralign = 1;
  }

  for (size_t i = predef_sec_count(); i < sections.size(); ++i) {
    DataSection &sec = *sections[i];
    Elf64_Shdr *hdr = sec_hdr(i);
    hdr->sh_name = sec.name;
    hdr->sh_type = sec.type;
    hdr->sh_flags = sec.flags;
    hdr->sh_addr = 0;
    hdr->sh_offset = out.size();
    hdr->sh_size = sec.size();
    hdr->sh_link = 0;
    hdr->sh_info = 0;
    hdr->sh_addralign = sec.align;
    hdr->sh_entsize = 0;
    if (sec.type == SHT_GROUP) [[unlikely]] {
      if (sym_is_local(sec.sym)) {
        hdr->sh_info = sym_idx(sec.sym);
      } else {
        hdr->sh_info = local_symbols.size() + sym_idx(sec.sym);
      }
      hdr->sh_link = sec_idx(".symtab");
      hdr->sh_entsize = 4;
    }

    const auto pad = util::align_up(sec.data.size(), 8) - sec.data.size();
    out.insert(out.end(), sec.data.begin(), sec.data.end());
    out.resize(out.size() + pad);

    if (sec.has_relocs) {
      assert(sections[i + 1] == nullptr);
      Elf64_Shdr *rela_hdr = sec_hdr(i + 1);
      rela_hdr->sh_name = sec.name - 5;
      rela_hdr->sh_type = SHT_RELA;
      rela_hdr->sh_flags = SHF_INFO_LINK | (sec.flags & SHF_GROUP);
      rela_hdr->sh_addr = 0;
      rela_hdr->sh_offset = out.size();
      rela_hdr->sh_size = sizeof(Elf64_Rela) * sec.relocs.size();
      rela_hdr->sh_link = sec_idx(".symtab");
      rela_hdr->sh_info = i;
      rela_hdr->sh_addralign = alignof(Elf64_Rela);
      rela_hdr->sh_entsize = sizeof(Elf64_Rela);

      // Skip allocated nullptr relocation section
      i += 1;

      // Resize invalidates rela_hdr.
      size_t rela_offset = rela_hdr->sh_offset;
      out.resize(out.size() + rela_hdr->sh_size);

      // Addend to symbol id to convert global symbol to the ELF symbol.
      u32 global_symbol_fix = u32{0x8000'0000} + local_symbols.size();

      auto *rela = reinterpret_cast<Elf64_Rela *>(out.data() + rela_offset);
      [[maybe_unused]] auto *rela_begin = rela;
      for (const Relocation &reloc : sec.relocs) {
        rela->r_addend = reloc.addend;
        rela->r_offset = reloc.offset;
        u32 symbol_fix = sym_is_local(reloc.symbol) ? 0 : global_symbol_fix;
        rela->r_info = ELF64_R_INFO(reloc.symbol.id() + symbol_fix, reloc.type);
        ++rela;
      }
      assert(sec_hdr(i)->sh_size == sizeof(Elf64_Rela) * (rela - rela_begin));
    }
  }

  if (secidx_symtax_shndx != 0) {
    auto *hdr = sec_hdr(secidx_symtax_shndx);
    hdr->sh_name = sec_off(".symtab_shndx");
    hdr->sh_type = SHT_SYMTAB_SHNDX;
    hdr->sh_offset = out.size();
    hdr->sh_size = sizeof(uint32_t) * sym_count;
    hdr->sh_link = sec_idx(".symtab");
    hdr->sh_addralign = 4;
    hdr->sh_entsize = 4;

    out.insert(out.end(),
               reinterpret_cast<const uint8_t *>(local_shndx.data()),
               reinterpret_cast<const uint8_t *>(local_shndx.data() +
                                                 local_shndx.size()));
    if (uint32_t missing = local_symbols.size() - local_shndx.size()) {
      out.resize(out.size() + sizeof(uint32_t) * missing);
    }
    out.insert(out.end(),
               reinterpret_cast<const uint8_t *>(global_shndx.data()),
               reinterpret_cast<const uint8_t *>(global_shndx.data() +
                                                 global_shndx.size()));
    if (uint32_t missing = global_symbols.size() - global_shndx.size()) {
      out.resize(out.size() + sizeof(uint32_t) * missing);
    }
  }

  return out;
}

namespace {

static consteval auto get_elf_section_flags() {
  using SectionFlags = Assembler::TargetInfo::SectionFlags;
  std::array<SectionFlags, unsigned(SectionKind::Max)> section_flags;
  section_flags[u8(SectionKind::Text)] =
      SectionFlags{.type = SHT_PROGBITS,
                   .flags = SHF_ALLOC | SHF_EXECINSTR,
                   .name = sec_off(".rela.text") + 5,
                   .align = 16};
  section_flags[u8(SectionKind::ReadOnly)] =
      SectionFlags{.type = SHT_PROGBITS,
                   .flags = SHF_ALLOC,
                   .name = sec_off(".rodata"),
                   .has_relocs = false};
  section_flags[u8(SectionKind::EHFrame)] =
      SectionFlags{.type = SHT_PROGBITS, // TODO: use SHT_X86_64_UNWIND on x86
                   .flags = SHF_ALLOC,
                   .name = sec_off(".rela.eh_frame") + 5,
                   .align = 8};
  section_flags[u8(SectionKind::LSDA)] =
      SectionFlags{.type = SHT_PROGBITS,
                   .flags = SHF_ALLOC,
                   .name = sec_off(".rela.gcc_except_table") + 5,
                   .align = 8};
  section_flags[u8(SectionKind::Data)] =
      SectionFlags{.type = SHT_PROGBITS,
                   .flags = SHF_ALLOC | SHF_WRITE,
                   .name = sec_off(".rela.data") + 5};
  section_flags[u8(SectionKind::DataRelRO)] =
      SectionFlags{.type = SHT_PROGBITS,
                   .flags = SHF_ALLOC | SHF_WRITE,
                   .name = sec_off(".rela.data.rel.ro") + 5};
  section_flags[u8(SectionKind::BSS)] =
      SectionFlags{.type = SHT_NOBITS,
                   .flags = SHF_ALLOC | SHF_WRITE,
                   .name = sec_off(".bss"),
                   .has_relocs = false,
                   .is_bss = true};
  section_flags[u8(SectionKind::ThreadData)] =
      SectionFlags{.type = SHT_PROGBITS,
                   .flags = SHF_ALLOC | SHF_WRITE | SHF_TLS,
                   .name = sec_off(".rela.tdata") + 5};
  section_flags[u8(SectionKind::ThreadBSS)] =
      SectionFlags{.type = SHT_NOBITS,
                   .flags = SHF_ALLOC | SHF_WRITE | SHF_TLS,
                   .name = sec_off(".tbss"),
                   .has_relocs = false,
                   .is_bss = true};
  return section_flags;
}

static constexpr auto elf_section_flags = get_elf_section_flags();

} // namespace

// Clang Format gives random indentation.
// clang-format off
const AssemblerElf::TargetInfoElf AssemblerElfA64::TARGET_INFO{
  {
    .reloc_pc32 = R_AARCH64_PREL32,
    .reloc_abs64 = R_AARCH64_ABS64,

    .section_flags = elf_section_flags,
  },

  ELFOSABI_SYSV,
  EM_AARCH64,
};

const AssemblerElf::TargetInfoElf AssemblerElfX64::TARGET_INFO{
  {
    .reloc_pc32 = R_X86_64_PC32,
    .reloc_abs64 = R_X86_64_64,

    .section_flags = elf_section_flags,
  },

  ELFOSABI_SYSV,
  EM_X86_64,
};
// clang-format on

} // end namespace tpde::elf
