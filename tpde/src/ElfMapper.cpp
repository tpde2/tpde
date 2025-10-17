// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/ElfMapper.hpp"

#include <algorithm>
#include <compare>
#include <elf.h>
#include <unistd.h>

#include "tpde/AssemblerElf.hpp"
#include "tpde/base.hpp"
#include "tpde/util/SmallVector.hpp"
#include "tpde/util/misc.hpp"

#if defined(__unix__)
  #include <disarm64.h>
  #include <fadec-enc2.h>
  #include <sys/mman.h>

extern "C" void __register_frame(void *);
extern "C" void __deregister_frame(void *);

#else
  #error "unsupported architecture/os combo"
#endif

namespace tpde {

namespace {

enum class Arch {
  Unknown,
  X86_64,
  AArch64,
};

#if defined(__x86_64__)
static constexpr Arch TargetArch = Arch::X86_64;
#elif defined(__aarch64__)
static constexpr Arch TargetArch = Arch::AArch64;
#else
static constexpr Arch TargetArch = Arch::Unknown;
#endif

} // anonymous namespace

void ElfMapper::reset() noexcept {
  if (!mapped_addr) {
    return;
  }

  if (registered_frame_off) {
    __deregister_frame(mapped_addr + registered_frame_off);
  }

  munmap(mapped_addr, mapped_size);
  mapped_addr = nullptr;
  sym_addrs.clear();
}

bool ElfMapper::map(AssemblerElf &assembler, SymbolResolver resolver) noexcept {
  // Approximate number of PLT/GOT slots.
  // TODO: better approximation
  u32 got_plt_slot_count =
      assembler.local_symbols.size() + assembler.global_symbols.size();

#ifdef __x86_64__
  // PLT+GOT slot: jmp qword ptr [rip + 2]; ud2; <address>
  constexpr size_t PLT_ENTRY_SIZE = 16;
#elif defined(__aarch64__)
  // PLT+GOT slot: ldr x16, pc+8; br x16; <address>
  constexpr size_t PLT_ENTRY_SIZE = 16;
#endif

  // Sort sections by permissions
  struct AllocSection {
    SecRef section;
    u32 sort_key;

    std::weak_ordering operator<=>(const AllocSection &other) const noexcept {
      return sort_key <=> other.sort_key;
    }
  };
  util::SmallVector<AllocSection> alloc_sections;

  for (size_t i = 0; i < assembler.sections.size(); ++i) {
    if (!assembler.sections[i]) { // skip relocation sections
      continue;
    }
    const auto &sec = *assembler.sections[i];
    if (!(sec.flags & SHF_ALLOC)) {
      continue;
    }
    u32 sort_key = 0;
    // executable before non-executable
    sort_key |= sec.flags & SHF_EXECINSTR ? 0 : (1 << 2);
    // read-only before writable
    sort_key |= !(sec.flags & SHF_WRITE) ? 0 : (1 << 1);
    // bss sections after data sections
    sort_key |= !(sec.type == SHT_NOBITS) ? 0 : (1 << 0);

    alloc_sections.emplace_back(SecRef(i), sort_key);
  }
  std::stable_sort(alloc_sections.begin(), alloc_sections.end());

  // Assign offsets and compute size of the mapped image
  size_t base_off = 0;
  u32 prev_flags = 0;
  // Permission boundaries, pair of (start addr, flags)
  util::SmallVector<std::pair<size_t, u32>> perm_boundaries;

  if (got_plt_slot_count) {
    perm_boundaries.emplace_back(base_off, SHF_EXECINSTR | SHF_ALLOC);
    base_off += got_plt_slot_count * PLT_ENTRY_SIZE;
    prev_flags = SHF_EXECINSTR | SHF_ALLOC;
  }

  size_t page_size = ::getpagesize();
  for (const auto &as : alloc_sections) {
    auto &sec = assembler.get_section(as.section);
    // mmap only hands out page-aligned regions.
    if (sec.align >= page_size) {
      TPDE_LOG_WARN("alignment ({:#x}) > PAGE_SIZE ({:#x}) will be ignored",
                    sec.align,
                    page_size);
    }
    if (prev_flags != sec.flags) {
      base_off = util::align_up(base_off, page_size);
      perm_boundaries.emplace_back(base_off, sec.flags);
      prev_flags = sec.flags;
    } else {
      base_off = util::align_up(base_off, sec.align);
    }
    sec.addr = base_off;
    size_t sec_size = sec.size();
    if (as.section == assembler.get_default_section(SectionKind::EHFrame)) {
      // Add zero-terminator to eh_frame. This is required for libgcc's
      // __register_frame, which iterates over FDEs up to the zero-terminator.
      sec_size += 4;
    }
    base_off += sec_size;

    TPDE_LOG_TRACE("allocate section {} size={:#x} to offset={:x}",
                   assembler.sec_name(as.section),
                   sec_size,
                   sec.addr);
  }
  // TODO(ts): align base_off up to page_size?
  perm_boundaries.emplace_back(base_off, 0);

  // Allocate memory
  mapped_size = base_off;
  void *mmap_res = ::mmap(nullptr,
                          mapped_size,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS,
                          -1,
                          0);
  if (mmap_res == MAP_FAILED || !mmap_res) {
    mapped_addr = nullptr;
    return false;
  }
  mapped_addr = static_cast<u8 *>(mmap_res);

  bool success = true;

  // Symbol addresses
  local_sym_count = assembler.local_symbols.size();
  sym_addrs.resize(local_sym_count + assembler.global_symbols.size());
  const auto sym_idx = [&](SymRef sym) -> size_t {
    auto idx = AssemblerElf::sym_idx(sym);
    if (!AssemblerElf::sym_is_local(sym)) {
      idx += local_sym_count;
    }
    return idx;
  };
  // Undefined symbols that are never used are permitted.
  const auto sym_addr = [&](SymRef sym) -> void * {
    size_t idx = sym_idx(sym);
    if (!sym_addrs[idx]) {
      const Elf64_Sym *elf_sym = assembler.sym_ptr(sym);
      if (elf_sym->st_shndx == SHN_UNDEF) {
        void *addr = resolver(assembler.sym_name(sym));
        sym_addrs[idx] = addr;
      } else if (elf_sym->st_shndx == SHN_ABS) {
        sym_addrs[idx] = reinterpret_cast<void *>(elf_sym->st_value);
      } else if (elf_sym->st_shndx < SHN_LORESERVE ||
                 elf_sym->st_shndx == SHN_XINDEX) {
        auto sec = assembler.sym_section(sym);
        auto off = assembler.get_section(sec).addr;
        sym_addrs[idx] = mapped_addr + off + elf_sym->st_value;
      } else {
        TPDE_LOG_ERR("unhandled section index {:x}", elf_sym->st_shndx);
        success = false;
      }
      TPDE_LOG_TRACE("resolved symbol '{}' to {}",
                     assembler.sym_name(sym),
                     sym_addrs[idx]);
    }

    return sym_addrs[idx];
  };
  // Resolve all defined global symbols.
  for (size_t i = 0; i < assembler.global_symbols.size(); ++i) {
    auto &elf_sym = assembler.global_symbols[i];
    if (elf_sym.st_shndx != SHN_UNDEF &&
        (elf_sym.st_shndx < SHN_LORESERVE || elf_sym.st_shndx == SHN_XINDEX)) {
      (void)sym_addr(SymRef(0x8000'0000 | i));
    }
  }

  // PLT/GOT slot management
  u8 *next_plt_entry = mapped_addr;
  // For every symbol index the PLT slot (offset 0) and GOT slot (offset 8).
  util::SmallVector<u8 *> got_plt_slots;
  got_plt_slots.resize(sym_addrs.size());
  const auto plt_entry = [&](size_t idx, uintptr_t addr) -> uintptr_t {
    if (!got_plt_slots[idx]) {
      if constexpr (TargetArch == Arch::X86_64) {
        fe64_JMPm(next_plt_entry, 0, FE_MEM(FE_IP, 0, FE_NOREG, 8));
        fe64_UD2(next_plt_entry + 6, 0);
        *reinterpret_cast<uintptr_t *>(next_plt_entry + sizeof(uintptr_t)) =
            addr;
      } else if constexpr (TargetArch == Arch::AArch64) {
        *reinterpret_cast<u32 *>(next_plt_entry + 0 * sizeof(u32)) =
            de64_LDRx_pcrel(DA_GP(16), 2);
        *reinterpret_cast<u32 *>(next_plt_entry + 1 * sizeof(u32)) =
            de64_BR(DA_GP(16));
        *reinterpret_cast<uintptr_t *>(next_plt_entry + sizeof(uintptr_t)) =
            addr;
      }

      assert(got_plt_slot_count-- > 0 && "insufficient PLT/GOT slots");
      got_plt_slots[idx] = next_plt_entry;
      next_plt_entry += PLT_ENTRY_SIZE;
    }
    return reinterpret_cast<uintptr_t>(got_plt_slots[idx]);
  };
  const auto got_entry = [&](size_t idx, uintptr_t addr) {
    return plt_entry(idx, addr) + 8;
  };
  (void)got_entry;

  const auto blend = [](uintptr_t pc, u32 mask, u32 data) {
    u32 *dest = reinterpret_cast<u32 *>(pc);
    *dest = (data & mask) | (*dest & ~mask);
  };
  const auto resolve_reloc = [&](u8 *sec_addr, Relocation &reloc) {
    SymRef sym_ref = reloc.symbol;
    uintptr_t sym = reinterpret_cast<uintptr_t>(sym_addr(sym_ref));
    uintptr_t syma = sym + reloc.addend;
    uintptr_t pc = reinterpret_cast<uintptr_t>(sec_addr + reloc.offset);

    if constexpr (TargetArch == Arch::X86_64) {
      switch (reloc.type) {
      case R_X86_64_64: {
        u64 v64 = syma;
        std::memcpy(reinterpret_cast<u8 *>(pc), &v64, sizeof(u64));
        break;
      }
      case R_X86_64_PC32: {
        auto v = syma - pc;
        if (util::sext(v, 32) != intptr_t(v)) {
          TPDE_LOG_ERR("R_X86_64_PC32 out of range: {:x}", v);
          success = false;
        }
        u32 v32 = v;
        std::memcpy(reinterpret_cast<u8 *>(pc), &v32, sizeof(u32));
        break;
      }
      case R_X86_64_PLT32: {
        auto v = syma - pc;
        if (util::sext(v, 32) != intptr_t(v)) {
          v = plt_entry(sym_idx(sym_ref), sym) + reloc.addend - pc;
        }
        if (util::sext(v, 32) != intptr_t(v)) {
          TPDE_LOG_ERR("R_X86_64_PLT32 out of range: {:x}", v);
          success = false;
        }
        u32 v32 = v;
        std::memcpy(reinterpret_cast<u8 *>(pc), &v32, sizeof(u32));
        break;
      }
      case R_X86_64_GOTPCREL: {
        auto got = got_entry(sym_idx(sym_ref), sym);
        auto v = got + reloc.addend - pc;
        if (util::sext(v, 32) != intptr_t(v)) {
          TPDE_LOG_ERR("R_X86_64_GOTPCREL out of range: {:x}", v);
          success = false;
        }
        u32 v32 = v;
        std::memcpy(reinterpret_cast<u8 *>(pc), &v32, sizeof(u32));
        break;
      }
      default:
        TPDE_LOG_ERR("unsupported relocation {}", reloc.type);
        success = false;
      }
    } else if constexpr (TargetArch == Arch::AArch64) {
      switch (reloc.type) {
      case R_AARCH64_ABS64: {
        u64 v64 = syma;
        std::memcpy(reinterpret_cast<u8 *>(pc), &v64, sizeof(u64));
        break;
      }
      case R_AARCH64_PREL32: {
        auto v = syma - pc;
        if (util::sext(v, 32) != intptr_t(v)) {
          TPDE_LOG_ERR("R_AARCH64_PREL32 out of range: {:x}", v);
          success = false;
        }
        u32 v32 = v;
        std::memcpy(reinterpret_cast<u8 *>(pc), &v32, sizeof(u32));
        break;
      }
      case R_AARCH64_CALL26: {
        auto v = syma - pc;
        if ((v & 3) || util::sext(v, 28) != intptr_t(v)) {
          v = plt_entry(sym_idx(sym_ref), sym) + reloc.addend - pc;
        }
        if (util::sext(v, 32) != intptr_t(v)) {
          TPDE_LOG_ERR("R_AARCH64_CALL26 out of range: {:x}", v);
          success = false;
        }
        blend(pc, 0x03ff'ffff, v >> 2);
        break;
      }
      case R_AARCH64_ADR_PREL_PG_HI21: {
        auto v = util::align_down(syma, 0x1000) - util::align_down(pc, 0x1000);
        if (util::sext(v, 33) != intptr_t(v)) {
          TPDE_LOG_ERR("R_AARCH64_ADR_PREL_PG_HI21 out of range: {:x}", v);
          success = false;
        }
        v >>= 12;
        blend(pc, 0x60ff'ffe0, (v & 3) << 29 | (((v >> 2) & 0x7'ffff) << 5));
        break;
      }
      case R_AARCH64_ADD_ABS_LO12_NC: blend(pc, 0xfff << 10, syma << 10); break;
      case R_AARCH64_ADR_GOT_PAGE: {
        auto got = got_entry(sym_idx(sym_ref), sym);
        auto v = util::align_down(got, 0x1000) - util::align_down(pc, 0x1000);
        if (util::sext(v, 33) != intptr_t(v)) {
          TPDE_LOG_ERR("R_AARCH64_ADR_GOT_PAGE out of range: {:x}", v);
          success = false;
        }
        v >>= 12;
        blend(pc, 0x60ff'ffe0, (v & 3) << 29 | (((v >> 2) & 0x7'ffff) << 5));
        break;
      }
      case R_AARCH64_LD64_GOT_LO12_NC: {
        auto got = got_entry(sym_idx(sym_ref), sym);
        blend(pc, 0xfff << 10, (got & 0xfff) << 7);
        break;
      }
      default:
        TPDE_LOG_ERR("unsupported relocation {}", reloc.type);
        success = false;
      }
    }
  };

  // Copy sections into allocation and resolve relocations
  for (const auto &as : alloc_sections) {
    auto &sec = assembler.get_section(as.section);
    // No need to zero bss, mmap zero-initializes memory.
    if (sec.type != SHT_NOBITS) {
      std::memcpy(mapped_addr + sec.addr, sec.data.data(), sec.size());
    }

    u8 *sec_addr = mapped_addr + sec.addr;
    for (auto &reloc : assembler.get_relocs(as.section)) {
      resolve_reloc(sec_addr, reloc);
    }
  }

  if (!success) {
    reset();
    return false;
  }

  // Adjust permissions
  assert(perm_boundaries.size() > 1);
  for (size_t i = 0; i < perm_boundaries.size() - 1; ++i) {
    auto [from, flags] = perm_boundaries[i];
    auto to = perm_boundaries[i + 1].first;
    int prot = PROT_READ;
    prot |= flags & SHF_EXECINSTR ? PROT_EXEC : 0;
    prot |= flags & SHF_WRITE ? PROT_WRITE : 0;
    TPDE_LOG_TRACE("mprotect: {:#x}-{:#x} {:#x}", from, to, prot);
    if (mprotect(mapped_addr + from, to - from, prot) != 0) {
      TPDE_LOG_ERR("mprotect failed");
      reset();
      return false;
    }
  }

  // Register eh_frame FDEs
  auto &eh_frame = assembler.get_section(
      assembler.get_default_section(SectionKind::EHFrame));
  registered_frame_off = eh_frame.addr;
  __register_frame(mapped_addr + registered_frame_off);

  return true;
}

void *ElfMapper::get_sym_addr(SymRef sym) noexcept {
  auto idx = AssemblerElf::sym_idx(sym);
  if (!AssemblerElf::sym_is_local(sym)) {
    idx += local_sym_count;
  }
  assert(idx < sym_addrs.size());
  return sym_addrs[idx];
}

} // namespace tpde
