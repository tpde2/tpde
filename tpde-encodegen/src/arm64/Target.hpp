// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once
#include <format>
#include <string>
#include <string_view>

#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/Target/TargetMachine.h>

#include "../Target.hpp"

namespace tpde_encgen::arm64 {

struct EncodingTargetArm64 : EncodingTarget {
  explicit EncodingTargetArm64(llvm::MachineFunction *func)
      : EncodingTarget(func) {}

  std::string_view get_invalid_reg() override { return "wtf??"; }

  void get_inst_candidates(
      llvm::MachineInstr &inst,
      llvm::SmallVectorImpl<MICandidate> &candidates) override;

  std::optional<std::pair<unsigned, unsigned>>
      is_move(const llvm::MachineInstr &mi) override;

  bool reg_is_gp(const llvm::Register reg) const {
    const auto *target_reg_info = func->getSubtarget().getRegisterInfo();

    if (!reg.isPhysical()) {
      assert(0);
      exit(1);
    }
    return target_reg_info->isGeneralPurposeRegister(*func, reg);
  }

  unsigned reg_id_from_mc_reg(const llvm::MCRegister reg) override {
    return func->getSubtarget().getRegisterInfo()->getEncodingValue(reg) +
           (reg_is_gp(reg) ? 0 : 0x20);
  }

  std::string_view reg_name_lower(unsigned id) override {
    if (id <= 0x40) {
      std::string_view names[] = {
          "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",  "x8",  "x9",
          "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19",
          "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29",
          "x30", "???", "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
          "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
          "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31",
      };
      return names[id];
    } else {
      assert(0);
      exit(1);
    }
  }

  unsigned reg_bank(const unsigned id) override {
    if (id < 0x20) {
      return 0;
    } else if (id <= 0x40) {
      return 1;
    } else {
      assert(0);
      exit(1);
    }
  }

  bool reg_should_be_ignored(const llvm::MCRegister reg) override {
    const auto name =
        std::string_view{func->getSubtarget().getRegisterInfo()->getName(reg)};
    return name == "WZR" || name == "XZR" || name == "WSP" || name == "SP" ||
           name == "NZCV" || name == "FPCR";
  }

  bool reg_is_flags(const llvm::MCRegister) override {
    return false; // Flag preservation is not needed on AArch64.
  }

  void generate_copy(std::string &buf,
                     unsigned indent,
                     unsigned bank,
                     std::string_view dst,
                     std::string_view src,
                     unsigned size) override {
    if (bank == 0) {
      char size_char = size <= 4 ? 'w' : 'x';
      std::format_to(std::back_inserter(buf),
                     "{:>{}}ASMD(MOV{}, {}, {});\n",
                     "",
                     indent,
                     size_char,
                     dst,
                     src);
    } else {
      assert(bank == 1 && false && "vector copy not implemented");
    }
  }

  bool inst_should_be_skipped(llvm::MachineInstr &) override { return false; }

  std::string_view jump_code(unsigned imm) override {
    // from llvm/lib/Target/AArch64/Utils/AArch64BaseInfo.h
    std::array<const char *, 16> cond_codes = {"Jeq",
                                               "Jne",
                                               "Jhs",
                                               "Jlo",
                                               "Jmi",
                                               "Jpl",
                                               "Jvs",
                                               "Jvc",
                                               "Jhi",
                                               "Jls",
                                               "Jge",
                                               "Jlt",
                                               "Jgt",
                                               "Jle",
                                               "Jal",
                                               "Jal"};
    if (imm >= cond_codes.size()) {
      assert(0);
      return {};
    }
    return cond_codes[imm];
  }
};

} // namespace tpde_encgen::arm64
