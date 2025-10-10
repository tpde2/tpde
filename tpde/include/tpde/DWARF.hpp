// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "base.hpp"

namespace tpde::dwarf {
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

} // namespace tpde::dwarf
