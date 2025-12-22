// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/Assembler.hpp"

namespace tpde {

Assembler::~Assembler() = default;

void Assembler::reset() {
  sections.clear();
  section_allocator.reset();
  default_sections = {};
}

SecRef Assembler::create_section(const TargetInfo::SectionFlags &flags) {
  SecRef ref = static_cast<SecRef>(sections.size());
  auto &sec = sections.emplace_back(new (section_allocator) DataSection(ref));
  sec->type = flags.type;
  sec->flags = flags.flags;
  sec->name = flags.name;
  sec->align = flags.align;
  sec->is_virtual = flags.is_bss;
  sec->has_relocs = flags.has_relocs;
  if (flags.has_relocs) {
    // TODO: this is ELF only
    sections.emplace_back(nullptr);
  }
  return ref;
}

u32 Assembler::sym_def_predef_data(SecRef sec_ref,
                                   SymRef sym,
                                   u64 size,
                                   u32 align) {
  DataSection &sec = get_section(sec_ref);
  sec.align = std::max(sec.align, align);
  size_t old_size = sec.size();
  size_t pos = util::align_up(sec.size(), align);
  sym_def(sym, sec_ref, pos, size);
  assert(!sec.is_virtual && "cannot add data to virtual section");
  sec.data.resize_uninitialized(pos + size);
  if (old_size < pos) {
    // Clear padding to avoid uninitialized bytes in output.
    std::memset(sec.data.data() + old_size, 0, pos - old_size);
  }
  return pos;
}

void Assembler::sym_def_predef_data(SecRef sec_ref,
                                    SymRef sym_ref,
                                    std::span<const u8> data,
                                    const u32 align,
                                    u32 *off) {
  u32 pos = sym_def_predef_data(sec_ref, sym_ref, data.size(), align);
  std::memcpy(get_section(sec_ref).data.data() + pos, data.data(), data.size());
  if (off) {
    *off = pos;
  }
}

void Assembler::sym_def_predef_zero(
    SecRef sec_ref, SymRef sym_ref, u32 size, u32 align, u32 *off) {
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

} // namespace tpde
