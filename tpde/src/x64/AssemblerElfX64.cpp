// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/x64/AssemblerElfX64.hpp"

namespace tpde::x64 {

namespace {

// TODO: use static constexpr array in C++23.
static constexpr auto get_cie_initial_instrs() {
  std::array<u8, 32> data{};
  // the current frame setup does not have a constant offset from the FP
  // to the CFA so we need to encode that at the end
  // for now just encode the CFA before the first sub sp

  // we always emit a frame-setup so we can encode that in the CIE

  u8 *dst = data.data();
  // def_cfa rsp, 8
  dst += AssemblerElfX64::write_eh_inst(
      dst, dwarf::DW_CFA_def_cfa, dwarf::x64::DW_reg_rsp, 8);
  // cfa_offset ra, 8
  dst += AssemblerElfX64::write_eh_inst(
      dst, dwarf::DW_CFA_offset, dwarf::x64::DW_reg_ra, 1);
  return std::make_pair(data, dst - data.data());
}

static constexpr auto cie_instrs = get_cie_initial_instrs();

} // namespace

// Clang Format gives random indentation.
// clang-format off
const AssemblerElfBase::TargetInfoElf AssemblerElfX64::TARGET_INFO{
  {
    .cie_return_addr_register = dwarf::x64::DW_reg_ra,
    .cie_instrs = {cie_instrs.first.data(), cie_instrs.second},
    .cie_code_alignment_factor = 1,   // ULEB128 1
    .cie_data_alignment_factor = 120, // SLEB128 -8

    .reloc_pc32 = R_X86_64_PC32,
    .reloc_abs64 = R_X86_64_64,
  },

  ELFOSABI_SYSV,
  EM_X86_64,
};
// clang-format on

} // end namespace tpde::x64
