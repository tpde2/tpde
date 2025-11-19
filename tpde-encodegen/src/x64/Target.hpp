// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once
#include <format>
#include <string>
#include <string_view>

#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/Target/TargetMachine.h>

#include "../Target.hpp"

namespace tpde_encgen::x64 {

struct EncodingTargetX64 : EncodingTarget {
  explicit EncodingTargetX64(llvm::MachineFunction *func)
      : EncodingTarget(func) {}

  std::string_view get_invalid_reg() override { return "FE_NOREG"; }

  void get_inst_candidates(
      llvm::MachineInstr &inst,
      llvm::SmallVectorImpl<MICandidate> &candidates) override;

  std::optional<std::pair<unsigned, unsigned>>
      is_move(const llvm::MachineInstr &mi) override {
    if (mi.isMoveReg() && !mi.hasImplicitDef()) {
      return std::make_pair(0, 1);
    }
    const llvm::TargetMachine &TM = mi.getMF()->getTarget();
    llvm::StringRef name = TM.getMCInstrInfo()->getName(mi.getOpcode());
    if (name == "MOVDQArr" || name == "MOVAPSrr" || name == "MOVAPDrr") {
      return std::make_pair(0, 1);
    }
    return std::nullopt;
  }

  bool reg_is_gp(const llvm::Register reg) const {
    assert(reg.isPhysical());
    const auto *target_reg_info = func->getSubtarget().getRegisterInfo();
    return target_reg_info->isGeneralPurposeRegister(*func, reg);
  }

  unsigned reg_id_from_mc_reg(const llvm::MCRegister reg) override {
    return func->getSubtarget().getRegisterInfo()->getEncodingValue(reg) +
           (reg_is_gp(reg) ? 0 : 0x20);
  }

  std::string_view reg_name_lower(unsigned id) override {
    static constexpr std::string_view gp_names[] = {
        "ax",
        "cx",
        "dx",
        "bx",
        "sp",
        "bp",
        "si",
        "di",
        "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
    };
    static constexpr std::string_view xmm_names[] = {
        "xmm0",  "xmm1",  "xmm2",  "xmm3",  "xmm4",  "xmm5",  "xmm6",  "xmm7",
        "xmm8",  "xmm9",  "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
        "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21", "xmm22", "xmm23",
        "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29", "xmm30", "xmm31",
    };

    if (id < 0x10) {
      return gp_names[id];
    } else if (id >= 0x20 && id < 0x40) {
      return xmm_names[id - 0x20];
    } else {
      llvm_unreachable("invalid register");
    }
  }

  unsigned reg_bank(const unsigned id) override {
    if (id < 0x10) {
      return 0;
    } else if (id >= 0x20 && id <= 0x40) {
      return 1;
    } else {
      llvm_unreachable("invalid register");
    }
  }

  bool reg_should_be_ignored(const llvm::MCRegister reg) override {
    if (!reg.isValid()) {
      return true;
    }
    const auto name =
        std::string_view{func->getSubtarget().getRegisterInfo()->getName(reg)};
    return (name == "EFLAGS" || name == "MXCSR" || name == "FPCW" ||
            name == "SSP" || name == "RIP");
  }

  bool reg_is_flags(const llvm::MCRegister reg) override {
    if (!reg.isValid()) {
      return false;
    }
    const auto name =
        std::string_view{func->getSubtarget().getRegisterInfo()->getName(reg)};
    return name == "EFLAGS";
  }

  void generate_copy(std::string &buf,
                     unsigned indent,
                     unsigned bank,
                     std::string_view dst,
                     std::string_view src,
                     unsigned size) override {
    if (bank == 0) {
      if (size <= 4) {
        std::format_to(std::back_inserter(buf),
                       "{:>{}}ASMD(MOV32rr, {}, {});\n",
                       "",
                       indent,
                       dst,
                       src);
      } else {
        assert(size <= 8);
        std::format_to(std::back_inserter(buf),
                       "{:>{}}ASMD(MOV64rr, {}, {});\n",
                       "",
                       indent,
                       dst,
                       src);
      }
    } else {
      assert(bank == 1);
      if (size <= 16) {
        std::format_to(
            std::back_inserter(buf), "{:>{}}if (has_avx()) {{\n", "", indent);
        std::format_to(std::back_inserter(buf),
                       "{:>{}}    ASMD(VMOVAPD128rr, {}, {});\n",
                       "",
                       indent,
                       dst,
                       src);
        std::format_to(
            std::back_inserter(buf), "{:>{}}}} else {{\n", "", indent);
        std::format_to(std::back_inserter(buf),
                       "{:>{}}    ASMD(SSE_MOVAPDrr, {}, {});\n",
                       "",
                       indent,
                       dst,
                       src);
        std::format_to(std::back_inserter(buf), "{:>{}}}}\n", "", indent);
      } else if (size <= 32) {
        std::format_to(std::back_inserter(buf),
                       "{:>{}}ASMD(VMOVAPD256rr, {}, {});\n",
                       "",
                       indent,
                       dst,
                       src);
      } else {
        assert(size <= 64);
        std::format_to(std::back_inserter(buf),
                       "{:>{}}ASMD(VMOVAPD512rr, {}, {});\n",
                       "",
                       indent,
                       dst,
                       src);
      }
    }
  }

  bool inst_should_be_skipped(llvm::MachineInstr &inst) override {
    const auto *inst_info =
        inst.getParent()->getParent()->getTarget().getMCInstrInfo();
    if (inst_info->getName(inst.getOpcode()) == "VZEROUPPER") {
      return true;
    }
    if (inst_info->getName(inst.getOpcode()) == "ENDBR64") {
      return true;
    }
    return false;
  }

  std::string_view jump_code(unsigned imm) override {
    std::array<const char *, 16> cond_codes = {"jo",
                                               "jno",
                                               "jb",
                                               "jae",
                                               "je",
                                               "jne",
                                               "jbe",
                                               "ja",
                                               "js",
                                               "jns",
                                               "jp",
                                               "jnp",
                                               "jl",
                                               "jge",
                                               "jle",
                                               "jg"};
    assert(imm < cond_codes.size() && "invalid jump condition");
    return cond_codes[imm];
  }
};

} // namespace tpde_encgen::x64
