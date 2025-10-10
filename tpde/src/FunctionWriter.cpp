// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/FunctionWriter.hpp"

#include "tpde/Assembler.hpp"
#include "tpde/DWARF.hpp"
#include "tpde/util/VectorWriter.hpp"

namespace tpde {

void FunctionWriterBase::begin_func() noexcept {
  func_begin = offset();

  label_offsets.clear();
  label_fixups.clear();

  except_call_site_table.clear();
  except_action_table.clear();
  except_type_info_table.clear();
  except_spec_table.clear();
  except_action_table.resize(2); // cleanup entry
}

void FunctionWriterBase::except_encode_func(Assembler &assembler) noexcept {
  if (except_call_site_table.empty()) {
    return;
  }

  // encode the call sites first, otherwise we can't write the header
  {
    util::VectorWriter ecst_writer(except_encoded_call_sites, 0);
    ecst_writer.reserve(16 * except_call_site_table.size() + 40);

    u64 fn_end = offset();
    u64 cur = func_begin;
    for (auto &info : except_call_site_table) {
      ecst_writer.reserve(80);

      if (info.start > cur) {
        // Encode padding entry
        ecst_writer.write_uleb_unchecked(cur - func_begin);
        ecst_writer.write_uleb_unchecked(info.start - cur);
        ecst_writer.write_uleb_unchecked(0);
        ecst_writer.write_uleb_unchecked(0);
      }
      ecst_writer.write_uleb_unchecked(info.start - func_begin);
      ecst_writer.write_uleb_unchecked(info.len);
      u64 fn_off = label_offset(info.landing_pad) - func_begin;
      assert(fn_off < (fn_end - func_begin));
      ecst_writer.write_uleb_unchecked(fn_off);
      ecst_writer.write_uleb_unchecked(info.action_entry);
      cur = info.start + info.len;
    }
    if (cur < fn_end) {
      // Add padding until the end of the function
      ecst_writer.write_uleb_unchecked(cur - func_begin);
      ecst_writer.write_uleb_unchecked(fn_end - cur);
      ecst_writer.write_uleb_unchecked(0);
      ecst_writer.write_uleb_unchecked(0);
    }

    // zero-terminate
    ecst_writer.write_unchecked<u8>(0);
    ecst_writer.write_unchecked<u8>(0);
  }

  {
    // TODO: if the text section is part of a section group, this should go into
    // the LSDA section of that group.
    SecRef secref_lsda = assembler.get_default_section(SectionKind::LSDA);
    util::VectorWriter et_writer(assembler.get_section(secref_lsda).data);
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
        assembler.reloc_pc32(secref_lsda, sym, off, 0);
        off -= sizeof(u32);
      }

      et_writer.write(except_spec_table);
    }
  }
}

void FunctionWriterBase::except_add_call_site(const u32 text_off,
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

void FunctionWriterBase::except_add_cleanup_action() noexcept {
  // pop back the action offset
  except_action_table.pop_back();
  i64 offset = -static_cast<i64>(except_action_table.size());
  util::VectorWriter(except_action_table).write_sleb(offset);
}

void FunctionWriterBase::except_add_action(const bool first_action,
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

void FunctionWriterBase::except_add_empty_spec_action(
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

u32 FunctionWriterBase::except_type_idx_for_sym(const SymRef sym) noexcept {
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

void FunctionWriterBase::more_space(size_t size) noexcept {
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

} // end namespace tpde
