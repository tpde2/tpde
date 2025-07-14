// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/AssemblerElf.hpp"
#include "tpde/StringTable.hpp"
#include "tpde/util/VectorWriter.hpp"
#include "tpde/util/misc.hpp"

#include <algorithm>
#include <elf.h>

namespace tpde {

// TODO(ts): maybe just outsource this to a helper func that can live in a cpp
// file?
namespace elf {
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

static void fail_constexpr_compile(const char *) {
  assert(0);
  exit(1);
}

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

consteval static u32 sec_off(const std::string_view name) {
  // skip the first null string
  const char *data = SHSTRTAB.data() + 1;
  auto sec_name = std::string_view{data};
  while (!sec_name.empty()) {
    if (sec_name.ends_with(name)) {
      return sec_name.data() + sec_name.size() - name.size() - SHSTRTAB.data();
    }

    data += sec_name.size() + 1;
    sec_name = std::string_view{data};
  }

  fail_constexpr_compile("unknown section name");
  return ~0u;
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

} // namespace elf

void AssemblerElfBase::reset() noexcept {
  Assembler::reset();

  global_symbols.clear();
  local_symbols.resize(1); // first symbol must be null
  temp_symbols.clear();
  temp_symbol_fixups.clear();
  next_free_tsfixup = ~0u;
  strtab = StringTable();
  shstrtab_extra = StringTable();
  secref_text = SecRef();
  secref_rodata = SecRef();
  secref_relro = SecRef();
  secref_data = SecRef();
  secref_bss = SecRef();
  secref_tdata = SecRef();
  secref_tbss = SecRef();
  secref_eh_frame = SecRef();
  secref_except_table = SecRef();
  cur_personality_func_addr = SymRef();

  init_sections();
  eh_init_cie();
}

SecRef AssemblerElfBase::create_section(unsigned type,
                                        unsigned flags,
                                        unsigned name) noexcept {
  SecRef ref = static_cast<SecRef>(sections.size());
  auto &sec = sections.emplace_back(new (section_allocator) DataSection(ref));
  sec->type = type;
  sec->flags = flags;
  sec->name = name;
  sec->is_virtual = type == SHT_NOBITS;
  sec->has_relocs = false;
  return ref;
}

SymRef AssemblerElfBase::create_section_symbol(SecRef ref,
                                               std::string_view name) noexcept {
  const auto str_off = strtab.add(name);

  unsigned shndx = sec_is_xindex(ref) ? SHN_XINDEX : ref.id();

  SymRef sym = SymRef(local_symbols.size());
  local_symbols.push_back(Elf64_Sym{
      .st_name = static_cast<Elf64_Word>(str_off),
      .st_info = ELF64_ST_INFO(STB_LOCAL, STT_SECTION),
      .st_other = STV_DEFAULT,
      .st_shndx = static_cast<Elf64_Section>(shndx),
      .st_value = 0,
      .st_size = 0,
  });
  if (sec_is_xindex(ref)) {
    sym_def_xindex(sym, ref);
  }
  return sym;
}

DataSection &AssemblerElfBase::get_or_create_section(SecRef &ref,
                                                     unsigned rela_name,
                                                     unsigned type,
                                                     unsigned flags,
                                                     unsigned align,
                                                     bool with_rela) noexcept {
  if (!ref.valid()) [[unlikely]] {
    if (with_rela) {
      ref = create_section(type, flags, rela_name + 5);
      sections.emplace_back(nullptr); // Reserve ID for RELA section.
    } else {
      ref = create_section(type, flags, rela_name);
    }

    std::string_view name{elf::SHSTRTAB.data() + rela_name +
                          (with_rela ? 5 : 0)};

    DataSection &sec = get_section(ref);
    sec.align = align;
    sec.sym = create_section_symbol(ref, name);
    sec.has_relocs = with_rela;
  }
  return get_section(ref);
}

const char *AssemblerElfBase::sec_name(SecRef ref) const noexcept {
  const DataSection &sec = get_section(ref);
  assert(sec.name < elf::SHSTRTAB.size());
  return elf::SHSTRTAB.data() + sec.name;
}

SecRef AssemblerElfBase::get_data_section(bool rodata, bool relro) noexcept {
  SecRef &secref = !rodata ? secref_data : relro ? secref_relro : secref_rodata;
  unsigned off_r = !rodata ? elf::sec_off(".rela.data")
                   : relro ? elf::sec_off(".rela.data.rel.ro")
                           : elf::sec_off(".rela.rodata");
  unsigned flags = SHF_ALLOC | (rodata && !relro ? 0 : SHF_WRITE);
  (void)get_or_create_section(secref, off_r, SHT_PROGBITS, flags, 1);
  return secref;
}

SecRef AssemblerElfBase::get_bss_section() noexcept {
  unsigned off = elf::sec_off(".bss");
  unsigned flags = SHF_ALLOC | SHF_WRITE;
  (void)get_or_create_section(secref_bss, off, SHT_NOBITS, flags, 1, false);
  return secref_bss;
}

SecRef AssemblerElfBase::get_tdata_section() noexcept {
  unsigned off_r = elf::sec_off(".rela.tdata");
  unsigned flags = SHF_ALLOC | SHF_WRITE | SHF_TLS;
  (void)get_or_create_section(secref_tdata, off_r, SHT_PROGBITS, flags, 1);
  return secref_tdata;
}

SecRef AssemblerElfBase::get_tbss_section() noexcept {
  unsigned off = elf::sec_off(".tbss");
  unsigned flags = SHF_ALLOC | SHF_WRITE | SHF_TLS;
  (void)get_or_create_section(secref_tbss, off, SHT_NOBITS, flags, 1, false);
  return secref_tbss;
}

SecRef AssemblerElfBase::create_structor_section(bool init,
                                                 SecRef group) noexcept {
  // TODO: priorities
  std::string_view name = init ? ".init_array" : ".fini_array";
  unsigned type = init ? SHT_INIT_ARRAY : SHT_FINI_ARRAY;
  SecRef secref =
      create_section(name, type, SHF_ALLOC | SHF_WRITE, true, group);
  get_section(secref).align = 8;
  return secref;
}

SecRef AssemblerElfBase::create_section(std::string_view name,
                                        unsigned type,
                                        unsigned flags,
                                        bool with_rela,
                                        SecRef group) noexcept {
  assert(type != SHT_GROUP && "use create_group_section to create groups");

  assert(name.find('\0') == std::string_view::npos &&
         "name must not contain null-bytes");
  size_t rela_name = elf::SHSTRTAB.size();
  rela_name += shstrtab_extra.add_prefix(with_rela ? ".rela" : "", name);

  assert(!(flags & SHF_GROUP) && "SHF_GROUP is added by assembler");
  DataSection *group_sec = nullptr;
  unsigned group_flag = 0;
  if (group.valid()) {
    group_flag = SHF_GROUP;
    group_sec = &get_section(group);
    assert(group_sec->type == SHT_GROUP);
  }

  SecRef ref;
  if (with_rela) {
    ref = create_section(type, flags | group_flag, rela_name + 5);
    sections.emplace_back(nullptr); // Reserve ID for RELA section.
    if (group_sec) {
      group_sec->write<u32>(ref.id());
      group_sec->write<u32>(ref.id() + 1);
    }
  } else {
    ref = create_section(type, flags | group_flag, rela_name);
    if (group_sec) {
      group_sec->write<u32>(ref.id());
    }
  }

  get_section(ref).sym = create_section_symbol(ref, name);
  get_section(ref).has_relocs = with_rela;
  return ref;
}

SecRef AssemblerElfBase::create_group_section(SymRef signature_sym,
                                              bool is_comdat) noexcept {
  SecRef ref = create_section(SHT_GROUP, 0, elf::sec_off(".group"));
  DataSection &sec = get_section(ref);
  sec.align = 4;
  sec.sym = signature_sym;
  // Group flags.
  sec.write<u32>(is_comdat ? GRP_COMDAT : 0);
  return ref;
}

void AssemblerElfBase::init_sections() noexcept {
  for (size_t i = 0; i < elf::predef_sec_count(); i++) {
    (void)create_section(SHT_NULL, 0, 0);
  }

  unsigned off_text = elf::sec_off(".rela.text");
  (void)get_or_create_section(
      secref_text, off_text, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 16);
  unsigned off_eh_frame = elf::sec_off(".rela.eh_frame");
  DataSection &eh_frame_sec = get_or_create_section(
      secref_eh_frame, off_eh_frame, SHT_PROGBITS, SHF_ALLOC, 8);
  eh_writer = util::VectorWriter(eh_frame_sec.data);
}


void AssemblerElfBase::sym_copy(SymRef dst, SymRef src) noexcept {
  Elf64_Sym *src_ptr = sym_ptr(src), *dst_ptr = sym_ptr(dst);

  dst_ptr->st_shndx = src_ptr->st_shndx;
  if (src_ptr->st_shndx == SHN_XINDEX) {
    sym_def_xindex(dst, sym_section(src));
  }
  dst_ptr->st_size = src_ptr->st_size;
  dst_ptr->st_value = src_ptr->st_value;
  // Don't copy st_info.
}

SymRef AssemblerElfBase::sym_add(const std::string_view name,
                                 SymBinding binding,
                                 u32 type) noexcept {
  size_t str_off = strtab.add(name);

  u8 info;
  switch (binding) {
    using enum AssemblerElfBase::SymBinding;
  case LOCAL: info = ELF64_ST_INFO(STB_LOCAL, type); break;
  case WEAK: info = ELF64_ST_INFO(STB_WEAK, type); break;
  case GLOBAL: info = ELF64_ST_INFO(STB_GLOBAL, type); break;
  default: TPDE_UNREACHABLE("invalid symbol binding");
  }
  auto sym = Elf64_Sym{.st_name = static_cast<Elf64_Word>(str_off),
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

void AssemblerElfBase::sym_def_predef_data(SecRef sec_ref,
                                           SymRef sym_ref,
                                           std::span<const u8> data,
                                           const u32 align,
                                           u32 *off) noexcept {
  DataSection &sec = get_section(sec_ref);
  sec.align = std::max(sec.align, align);
  size_t pos = util::align_up(sec.size(), align);
  sym_def(sym_ref, sec_ref, pos, data.size());
  assert(!sec.is_virtual && "cannot add data to virtual section");
  sec.data.resize(pos);
  sec.data.append(data.begin(), data.end());

  if (off) {
    *off = pos;
  }
}

void AssemblerElfBase::sym_def_predef_zero(
    SecRef sec_ref, SymRef sym_ref, u32 size, u32 align, u32 *off) noexcept {
  DataSection &sec = get_section(sec_ref);
  sec.align = std::max(sec.align, align);
  size_t pos = util::align_up(sec.size(), align);
  sym_def(sym_ref, sec_ref, pos, size);
  if (sec.is_virtual) {
    sec.vsize = pos + size;
  } else {
    sec.data.resize(pos + size);
  }

  if (off) {
    *off = pos;
  }
}

void AssemblerElfBase::sym_def_xindex(SymRef sym_ref, SecRef sec_ref) noexcept {
  assert(sec_is_xindex(sec_ref));
  auto &shndx = sym_is_local(sym_ref) ? local_shndx : global_shndx;
  if (shndx.size() <= sym_idx(sym_ref)) {
    shndx.resize(sym_idx(sym_ref) + 1);
  }
  shndx[sym_idx(sym_ref)] = sec_ref.id();
}

void AssemblerElfBase::reloc_sec(const SecRef sec_ref,
                                 const SymRef sym,
                                 const u32 type,
                                 const u64 offset,
                                 const i64 addend) noexcept {
  assert(i32(addend) == addend && "non-32-bit addends are unsupported");
  get_section(sec_ref).relocs.emplace_back(offset, sym, type, addend);
}

void AssemblerElfBase::reloc_sec(const SecRef sec,
                                 const Label label,
                                 const u8 kind,
                                 const u32 offset) noexcept {
  assert(label_is_pending(label));
  u32 fixup_idx;
  if (next_free_tsfixup != ~0u) {
    fixup_idx = next_free_tsfixup;
    next_free_tsfixup = temp_symbol_fixups[fixup_idx].next_list_entry;
  } else {
    fixup_idx = temp_symbol_fixups.size();
    temp_symbol_fixups.push_back(TempSymbolFixup{});
  }

  TempSymbolInfo &info = temp_symbols[static_cast<u32>(label)];
  temp_symbol_fixups[fixup_idx] = TempSymbolFixup{
      .section = sec,
      .next_list_entry = info.fixup_idx,
      .off = offset,
      .kind = kind,
  };
  info.fixup_idx = fixup_idx;
}

void AssemblerElfBase::eh_align_frame() noexcept {
  if (unsigned count = -eh_writer.size() & 7) {
    eh_writer.reserve(8);
    // Small hack for performance: always write 8 bytes (single instruction, no
    // loop in machine code), but adjust pointer only by count bytes.
    for (unsigned i = 0; i < 8; ++i) {
      eh_writer.write_unchecked<u8>(dwarf::DW_CFA_nop);
    }
    eh_writer.unskip(8 - count);
  }
}

void AssemblerElfBase::eh_write_inst(const u8 opcode, const u64 arg) noexcept {
  if ((opcode & dwarf::DWARF_CFI_PRIMARY_OPCODE_MASK) != 0) {
    assert((arg & dwarf::DWARF_CFI_PRIMARY_OPCODE_MASK) == 0);
    eh_writer.write<u8>(opcode | arg);
  } else {
    eh_writer.reserve(11);
    eh_writer.write_unchecked<u8>(opcode);
    eh_writer.write_uleb_unchecked(arg);
  }
}

void AssemblerElfBase::eh_write_inst(const u8 opcode,
                                     const u64 first_arg,
                                     const u64 second_arg) noexcept {
  eh_write_inst(opcode, first_arg);
  eh_writer.write_uleb(second_arg);
}

void AssemblerElfBase::eh_init_cie(SymRef personality_func_addr) noexcept {
  // write out the initial CIE

  // CIE layout:
  // length: u32
  // id: u32
  // version: u8
  // augmentation: 3 or 5 bytes depending on whether the CIE has a personality
  // function
  // code_alignment_factor: uleb128 (but we only use 1 byte)
  // data_alignment_factor: sleb128 (but we only use 1 byte)
  // return_addr_register: u8
  // augmentation_data_len: uleb128 (but we only use 1 byte)
  // augmentation_data:
  //   if personality:
  //     personality_encoding: u8
  //     personality_addr: u32
  //     lsa_encoding: u8
  //   fde_ptr_encoding: u8
  // instructions: [u8]
  //
  // total: 17 bytes or 25 bytes

  const auto first = eh_writer.size() == 0;
  auto off = eh_writer.size();
  assert(off % 8 == 0 && "eh_frame section unaligned");
  eh_cur_cie_off = off;

  eh_writer.reserve(32 + target_info.cie_instrs.size());

  eh_writer.skip_unchecked(4);       // length written at the end.
  eh_writer.write_unchecked<u32>(0); // id is 0 for CIEs
  eh_writer.write_unchecked<u8>(1);  // version
  if (!personality_func_addr.valid()) {
    // augmentation is "zR" for a CIE with no personality meaning there is
    // the augmentation_data_len and ptr_size field
    eh_writer.write_unchecked<u8>('z');
    eh_writer.write_unchecked<u8>('R');
  } else {
    // with a personality function the augmentation is "zPLR" meaning there
    // is augmentation_data_len, personality_encoding, personality_addr,
    // lsa_encoding and ptr_size
    eh_writer.write_unchecked<u8>('z');
    eh_writer.write_unchecked<u8>('P');
    eh_writer.write_unchecked<u8>('L');
    eh_writer.write_unchecked<u8>('R');
  }
  eh_writer.write_unchecked<u8>(0);

  eh_writer.write_unchecked<u8>(target_info.cie_code_alignment_factor);
  eh_writer.write_unchecked<u8>(target_info.cie_data_alignment_factor);

  // return_addr_register is defined by the derived impl
  eh_writer.write_unchecked<u8>(target_info.cie_return_addr_register);

  // augmentation_data_len is 1 when no personality is present or 7 otherwise
  eh_writer.write_unchecked<u8>((!personality_func_addr.valid()) ? 1 : 7);

  if (personality_func_addr.valid()) {
    // the personality encoding is a 4-byte pc-relative address where the
    // address of the personality func is stored
    eh_writer.write_unchecked<u8>(dwarf::DW_EH_PE_pcrel |
                                  dwarf::DW_EH_PE_sdata4 |
                                  dwarf::DW_EH_PE_indirect);

    reloc_sec(secref_eh_frame,
              personality_func_addr,
              target_info.reloc_pc32,
              eh_writer.size(),
              0);
    eh_writer.write_unchecked<u32>(
        0); // relocated, zero for deterministic output

    // the lsa_encoding as a 4-byte pc-relative address since the whole
    // object should fit in 2gb
    eh_writer.write_unchecked<u8>(dwarf::DW_EH_PE_pcrel |
                                  dwarf::DW_EH_PE_sdata4);
  }

  // fde_ptr_encoding is a 4-byte signed pc-relative address
  eh_writer.write_unchecked<u8>(dwarf::DW_EH_PE_sdata4 | dwarf::DW_EH_PE_pcrel);

  eh_writer.write_unchecked(target_info.cie_instrs);

  eh_align_frame();

  // patch size of CIE (length is not counted)
  u32 size = eh_writer.size() - off - sizeof(u32);
  std::memcpy(eh_writer.data() + off, &size, sizeof(size));

  if (first) {
    eh_first_fde_off = eh_writer.size();
  }
}

u32 AssemblerElfBase::eh_begin_fde(SymRef personality_func_addr) noexcept {
  if (personality_func_addr != cur_personality_func_addr) {
    eh_init_cie(personality_func_addr);
    cur_personality_func_addr = personality_func_addr;
  }

  const auto fde_off = eh_writer.size();
  assert(fde_off % 8 == 0 && "eh_frame section unaligned");

  // FDE Layout:
  //  length: u32
  //  id: u32
  //  func_start: i32
  //  func_size: i32
  // augmentation_data_len: uleb128 (but we only use 1 byte)
  // augmentation_data:
  //   if personality:
  //     lsda_ptr: i32 (we use a 4 byte signed pc-relative pointer to an
  //     absolute address)
  // instructions: [u8]
  //
  // Total Size: 17 bytes or 21 bytes

  eh_writer.zero(!cur_personality_func_addr.valid() ? 17 : 21);
  u8 *data = eh_writer.data() + fde_off;

  // we encode length later

  // id is the offset from the current CIE to the id field
  *reinterpret_cast<u32 *>(data + 4) = fde_off - eh_cur_cie_off + sizeof(u32);

  // func_start and func_size will be relocated at the end

  // augmentation_data_len is 0 with no personality or 4 otherwise
  if (cur_personality_func_addr.valid()) {
    data[16] = 4;
  }

  return fde_off;
}

void AssemblerElfBase::eh_end_fde(u32 fde_start, SymRef func) noexcept {
  eh_align_frame();

  u8 *eh_data = eh_writer.data();
  auto *func_sym = sym_ptr(func);

  // relocate the func_start to the function
  // relocate against .text so we don't have to fix up any relocations
  // NB: ld.bfd (for a reason that needs to be investigated) doesn't accept
  // using the function symbol here.
  DataSection &func_sec = get_section(sym_section(func));
  this->reloc_sec(secref_eh_frame,
                  func_sec.sym,
                  target_info.reloc_pc32,
                  fde_start + 8,
                  func_sym->st_value);
  // Adjust func_size to the function size
  *reinterpret_cast<i32 *>(eh_data + fde_start + 12) = func_sym->st_size;

  const u32 len = eh_writer.size() - fde_start - sizeof(u32);
  *reinterpret_cast<u32 *>(eh_data + fde_start) = len;
  if (cur_personality_func_addr.valid()) {
    DataSection &except_table =
        get_or_create_section(secref_except_table,
                              elf::sec_off(".rela.gcc_except_table"),
                              SHT_PROGBITS,
                              SHF_ALLOC,
                              8);
    ;
    reloc_sec(secref_eh_frame,
              except_table.sym,
              target_info.reloc_pc32,
              fde_start + 17,
              except_table.data.size());
  }
}

void AssemblerElfBase::except_begin_func() noexcept {
  except_call_site_table.clear();
  except_action_table.clear();
  except_type_info_table.clear();
  except_spec_table.clear();
  except_action_table.resize(2); // cleanup entry
}

void AssemblerElfBase::except_encode_func(SymRef func_sym) noexcept {
  if (!cur_personality_func_addr.valid()) {
    assert(except_call_site_table.empty());
    assert(except_type_info_table.empty());
    assert(except_spec_table.empty());
    assert(except_action_table.size() == 2);
    return;
  }

  // encode the call sites first, otherwise we can't write the header
  {
    util::VectorWriter ecst_writer(except_encoded_call_sites, 0);
    ecst_writer.reserve(16 * except_call_site_table.size() + 40);

    const auto *sym = sym_ptr(func_sym);
    u64 fn_start = sym->st_value;
    u64 fn_end = fn_start + sym->st_size;
    u64 cur = fn_start;
    for (auto &info : except_call_site_table) {
      ecst_writer.reserve(80);

      if (info.start > cur) {
        // Encode padding entry
        ecst_writer.write_uleb_unchecked(cur - fn_start);
        ecst_writer.write_uleb_unchecked(info.start - cur);
        ecst_writer.write_uleb_unchecked(0);
        ecst_writer.write_uleb_unchecked(0);
      }
      ecst_writer.write_uleb_unchecked(info.start - fn_start);
      ecst_writer.write_uleb_unchecked(info.len);
      u64 fn_off = label_offset(info.landing_pad) - fn_start;
      assert(fn_off < (fn_end - fn_start));
      ecst_writer.write_uleb_unchecked(fn_off);
      ecst_writer.write_uleb_unchecked(info.action_entry);
      cur = info.start + info.len;
    }
    if (cur < fn_end) {
      // Add padding until the end of the function
      ecst_writer.write_uleb_unchecked(cur - fn_start);
      ecst_writer.write_uleb_unchecked(fn_end - cur);
      ecst_writer.write_uleb_unchecked(0);
      ecst_writer.write_uleb_unchecked(0);
    }

    // zero-terminate
    ecst_writer.write_unchecked<u8>(0);
    ecst_writer.write_unchecked<u8>(0);
  }

  {
    util::VectorWriter et_writer(get_section(secref_except_table).data);
    // write the lsda (see
    // https://github.com/llvm/llvm-project/blob/main/libcxxabi/src/cxa_personality.cpp#L60)
    et_writer.write<u8>(dwarf::DW_EH_PE_omit); // lpStartEncoding
    if (except_action_table.empty()) {
      assert(except_type_info_table.empty());
      // we don't need the type_info table if there is no action entry
      et_writer.write<u8>(dwarf::DW_EH_PE_omit); // ttypeEncoding
    } else {
      et_writer.write<u8>(dwarf::DW_EH_PE_sdata4 | dwarf::DW_EH_PE_pcrel |
                          dwarf::DW_EH_PE_indirect); // ttypeEncoding
      uint64_t classInfoOff =
          (except_type_info_table.size() + 1) * sizeof(uint32_t);
      classInfoOff += except_action_table.size();
      classInfoOff += except_encoded_call_sites.size() +
                      util::uleb_len(except_encoded_call_sites.size()) + 1;
      et_writer.write_uleb(classInfoOff);
    }

    et_writer.write<u8>(dwarf::DW_EH_PE_uleb128); // callSiteEncoding
    et_writer.write_uleb(
        except_encoded_call_sites.size()); // callSiteTableLength
    et_writer.write(except_encoded_call_sites);
    et_writer.write(except_action_table);

    if (!except_action_table.empty()) {
      // allocate space for type_info table
      et_writer.zero((except_type_info_table.size() + 1) * sizeof(u32));

      // in reverse order since indices are negative
      size_t off = et_writer.size() - sizeof(u32) * 2;
      for (auto sym : except_type_info_table) {
        reloc_sec(secref_except_table, sym, target_info.reloc_pc32, off, 0);
        off -= sizeof(u32);
      }

      et_writer.write(except_spec_table);
    }
  }
}

void AssemblerElfBase::except_add_call_site(const u32 text_off,
                                            const u32 len,
                                            const Label landing_pad,
                                            const bool is_cleanup) noexcept {
  except_call_site_table.push_back(ExceptCallSiteInfo{
      .start = text_off,
      .len = len,
      .landing_pad = landing_pad,
      .action_entry =
          (is_cleanup ? 0 : static_cast<u32>(except_action_table.size()) + 1),
  });
}

void AssemblerElfBase::except_add_cleanup_action() noexcept {
  // pop back the action offset
  except_action_table.pop_back();
  i64 offset = -static_cast<i64>(except_action_table.size());
  util::VectorWriter(except_action_table).write_sleb(offset);
}

void AssemblerElfBase::except_add_action(const bool first_action,
                                         const SymRef type_sym) noexcept {
  if (!first_action) {
    except_action_table.back() = 1;
  }

  auto idx = 0u;
  if (type_sym.valid()) {
    auto found = false;
    for (const auto &sym : except_type_info_table) {
      ++idx;
      if (sym == type_sym) {
        found = true;
        break;
      }
    }
    if (!found) {
      ++idx;
      except_type_info_table.push_back(type_sym);
    }
  }

  util::VectorWriter(except_action_table).write_sleb(idx + 1);
  except_action_table.push_back(0);
}

void AssemblerElfBase::except_add_empty_spec_action(
    const bool first_action) noexcept {
  if (!first_action) {
    except_action_table.back() = 1;
  }

  if (except_spec_table.empty()) {
    except_spec_table.resize(4);
  }

  except_action_table.push_back(127); // SLEB -1
  except_action_table.push_back(0);
}

u32 AssemblerElfBase::except_type_idx_for_sym(const SymRef sym) noexcept {
  // to explain the indexing
  // a ttypeIndex of 0 is reserved for a cleanup action so the type table
  // starts at 1 but the first entry in the type table is reserved for the 0
  // pointer used for catch(...) meaning we start at 2
  auto idx = 2u;
  for (const auto type_sym : except_type_info_table) {
    if (type_sym == sym) {
      return idx;
    }
    ++idx;
  }
  assert(0);
  return idx;
}

void AssemblerElfBase::finalize() noexcept { eh_writer.flush(); }

std::vector<u8> AssemblerElfBase::build_object_file() noexcept {
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

    hdr->e_ident[0] = ELFMAG0;
    hdr->e_ident[1] = ELFMAG1;
    hdr->e_ident[2] = ELFMAG2;
    hdr->e_ident[3] = ELFMAG3;
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

} // end namespace tpde
