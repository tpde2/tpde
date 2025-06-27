// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "arm64/Target.hpp"

#include <cstdint>
#include <format>
#include <string>
#include <string_view>

#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineOperand.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>


using namespace std::literals;

namespace tpde_encgen::arm64 {

namespace {

std::string format_reg(const llvm::MachineOperand &mo, std::string_view op) {
  const auto &tri =
      *mo.getParent()->getMF()->getRegInfo().getTargetRegisterInfo();
  llvm::StringRef name = tri.getName(mo.getReg());
  if (name == "WZR" || name == "XZR") {
    return "DA_ZR";
  } else if (name == "WSP" || name == "SP") {
    return "DA_SP";
  }
  return std::string(op);
}

} // end anonymous namespace

void EncodingTargetArm64::get_inst_candidates(
    llvm::MachineInstr &mi, llvm::SmallVectorImpl<MICandidate> &candidates) {
  const llvm::TargetMachine &TM = mi.getMF()->getTarget();
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCInstrDesc &MCID = MCII.get(mi.getOpcode());
  (void)MCID;

  llvm::StringRef Name = MCII.getName(mi.getOpcode());

  // From Disarm
  const auto imm_logical = [](bool sf, uint64_t Nimmsimmr) {
    unsigned N = Nimmsimmr >> 12;
    unsigned immr = Nimmsimmr >> 6 & 0x3f;
    unsigned imms = Nimmsimmr & 0x3f;
    unsigned len = 31 - __builtin_clz((N << 6) | (~imms & 0x3f));
    unsigned levels = (1 << len) - 1;
    unsigned s = imms & levels;
    unsigned r = immr & levels;
    unsigned esize = 1 << len;
    uint64_t welem = ((uint64_t)1 << (s + 1)) - 1;
    // ROR(welem, r) as bits(esize)
    welem = r ? (welem >> r) | (welem << (esize - r)) : welem;
    if (esize < 64) {
      welem &= ((uint64_t)1 << esize) - 1;
    }
    // Replicate(ROR(welem, r))
    uint64_t wmask = 0;
    for (unsigned i = 0; i < (!sf ? 32 : 64); i += esize) {
      wmask |= welem << i;
    }
    return wmask;
  };

  const auto handle_mem_imm = [&](std::string_view mnem,
                                  std::string_view mnemu,
                                  unsigned shift) {
    if (mi.getOperand(2).isImm()) {
      auto cond1 =
          std::format("{:#x}, {}", mi.getOperand(2).getImm() << shift, shift);
      candidates.emplace_back(1,
                              "encodeable_with_mem_uoff12",
                              cond1,
                              [mnem](llvm::raw_ostream &os,
                                     const llvm::MachineInstr &mi,
                                     std::span<const std::string> ops) {
                                os << "    ASMD(" << mnem << ", ";
                                if (mi.getOperand(0).isImm()) {
                                  os << "(Da64PrfOp)"
                                     << mi.getOperand(0).getImm();
                                } else {
                                  os << format_reg(mi.getOperand(0), ops[0]);
                                }
                                const auto &mem_op = ops[ops.size() - 1];
                                os << ", " << mem_op << ".first, " << mem_op
                                   << ".second);\n";
                              });
    }
    (void)mnemu;
    candidates.emplace_back([mnem, shift](llvm::raw_ostream &os,
                                          const llvm::MachineInstr &mi,
                                          std::span<const std::string> ops) {
      os << "    ASMD(" << mnem << ", ";
      unsigned op_idx = 0;
      if (mi.getOperand(0).isImm()) {
        os << "(Da64PrfOp)" << mi.getOperand(0).getImm();
      } else {
        os << format_reg(mi.getOperand(0), ops[0]);
        op_idx += 1;
      }
      os << ", " << format_reg(mi.getOperand(1), ops[op_idx]);
      if (mi.getOperand(2).isImm()) {
        os << ", " << (mi.getOperand(2).getImm() << shift) << ");\n";
      } else if (mi.getOperand(2).isCPI()) {
        os << ", 0);\n";
        os << "    derived()->reloc_text(" << ops[2] << ", R_AARCH64_LDST"
           << (8 << shift)
           << "_ABS_LO12_NC, "
              "derived()->text_writer.offset() - 4);\n";
      }
    });
  };
  const auto handle_mem_simm = [&](std::string_view mnem, bool with_update) {
    candidates.emplace_back(
        [mnem, with_update](llvm::raw_ostream &os,
                            const llvm::MachineInstr &mi,
                            std::span<const std::string> ops) {
          os << "    ASMD(" << mnem << ", ";
          unsigned op_idx = with_update ? 1 : 0;
          if (mi.getOperand(with_update).isImm()) {
            os << "(Da64PrfOp)" << mi.getOperand(with_update).getImm();
          } else {
            os << format_reg(mi.getOperand(with_update), ops[op_idx]);
            op_idx += 1;
          }
          os << ", " << format_reg(mi.getOperand(with_update + 1), ops[op_idx]);
          os << ", " << mi.getOperand(with_update + 2).getImm() << ");\n";
        });
  };
  const auto handle_shift_imm = [&](std::string_view mnem, unsigned size) {
    candidates.emplace_back(
        2,
        "encodeable_as_shiftimm",
        std::format("{}", size),
        [mnem](llvm::raw_ostream &os,
               const llvm::MachineInstr &mi,
               std::span<const std::string> ops) {
          os << "    ASMD(" << mnem;
          for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
            os << ", " << format_reg(mi.getOperand(i), ops[i]);
          }
          os << ");\n";
        });
  };
  const auto handle_arith_imm = [&](std::string_view mnem) {
    // Cannot encode ADD/SUB with immediate and zero register as second operand.
    if (format_reg(mi.getOperand(1), "") == "DA_ZR") {
      return;
    }
    candidates.emplace_back(2,
                            "encodeable_as_immarith",
                            "",
                            [mnem](llvm::raw_ostream &os,
                                   const llvm::MachineInstr &mi,
                                   std::span<const std::string> ops) {
                              auto dst = format_reg(mi.getOperand(0), ops[0]);
                              auto src = format_reg(mi.getOperand(1), ops[1]);
                              os << "    ASMD(" << mnem << ", " << dst << ", "
                                 << src << ", " << ops[2] << ");\n";
                            });
    if (!mnem.starts_with("ADD")) {
      return;
    }
    candidates.emplace_back(1,
                            "encodeable_as_immarith",
                            "",
                            [mnem](llvm::raw_ostream &os,
                                   const llvm::MachineInstr &mi,
                                   std::span<const std::string> ops) {
                              auto dst = format_reg(mi.getOperand(0), ops[0]);
                              auto src = format_reg(mi.getOperand(2), ops[2]);
                              os << "    ASMD(" << mnem << ", " << dst << ", "
                                 << src << ", " << ops[1] << ");\n";
                            });
  };
  const auto handle_logical_imm = [&](std::string_view mnem, bool inv) {
    candidates.emplace_back(2,
                            "encodeable_as_immlogical",
                            inv ? "true" : "false",
                            [mnem](llvm::raw_ostream &os,
                                   const llvm::MachineInstr &mi,
                                   std::span<const std::string> ops) {
                              auto dst = format_reg(mi.getOperand(0), ops[0]);
                              auto src = format_reg(mi.getOperand(1), ops[1]);
                              os << "    ASMD(" << mnem << ", " << dst << ", "
                                 << src << ", " << ops[2] << ");\n";
                            });
    if (inv) {
      return;
    }
    candidates.emplace_back(1,
                            "encodeable_as_immlogical",
                            "false",
                            [mnem](llvm::raw_ostream &os,
                                   const llvm::MachineInstr &mi,
                                   std::span<const std::string> ops) {
                              auto dst = format_reg(mi.getOperand(0), ops[0]);
                              auto src = format_reg(mi.getOperand(2), ops[2]);
                              os << "    ASMD(" << mnem << ", " << dst << ", "
                                 << src << ", " << ops[1] << ");\n";
                            });
  };
  const auto handle_adrp = [&](std::string_view mnem) {
    candidates.emplace_back([mnem](llvm::raw_ostream &os,
                                   const llvm::MachineInstr &mi,
                                   std::span<const std::string> ops) {
      assert(mi.getOperand(1).isCPI() && "ADRP expect constant pool imm");
      auto dst = format_reg(mi.getOperand(0), ops[0]);
      os << "    ASMD(" << mnem << ", " << dst << ", 0, 0);";
      os << "    derived()->reloc_text(" << ops[1]
         << ", R_AARCH64_ADR_PREL_PG_HI21, "
         << "derived()->text_writer.offset() - 4);\n";
    });
  };
  const auto handle_default = [&](std::string_view mnem,
                                  std::string extra_ops = "") {
    candidates.emplace_back(
        [mnem, extra_ops](llvm::raw_ostream &os,
                          const llvm::MachineInstr &mi,
                          std::span<const std::string> ops) {
          os << "    ASMD(" << mnem;
          unsigned reg_idx = 0;
          for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
            const auto &op = mi.getOperand(i);
            if (op.isReg()) {
              if (!op.isTied() || !op.isUse()) {
                os << ", " << format_reg(op, ops[reg_idx++]);
              }
            } else if (op.isImm()) {
              if (mnem.starts_with("CCMP") || mnem.starts_with("FCCMP") ||
                  mnem.starts_with("CS") || mnem.starts_with("FCSEL")) {
                std::array<std::string_view, 16> ccs = {"DA_EQ",
                                                        "DA_NE",
                                                        "DA_HS",
                                                        "DA_LO",
                                                        "DA_MI",
                                                        "DA_PL",
                                                        "DA_VS",
                                                        "DA_VC",
                                                        "DA_HI",
                                                        "DA_LS",
                                                        "DA_GE",
                                                        "DA_LT",
                                                        "DA_GT",
                                                        "DA_LE",
                                                        "DA_AL",
                                                        "DA_NV"};
                os << ", " << ccs[op.getImm()];
              } else if (op.getOperandNo() == 0 && mnem.starts_with("PRFM")) {
                os << ", (Da64PrfOp)" << op.getImm();
              } else {
                os << ", " << op.getImm();
              }
            } else {
              assert(false);
            }
          }
          os << extra_ops << ");\n";
        });
  };
  const auto handle_noimm = [&](std::string_view mnem,
                                std::string extra_ops = "") {
    std::string mnem_cpy{mnem};
    candidates.emplace_back(
        [mnem_cpy, extra_ops](llvm::raw_ostream &os,
                              const llvm::MachineInstr &mi,
                              std::span<const std::string> ops) {
          os << "    ASMD(" << mnem_cpy;
          unsigned reg_idx = 0;
          for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
            const auto &op = mi.getOperand(i);
            if (op.isReg()) {
              if (!op.isTied() || !op.isUse()) {
                os << ", " << format_reg(op, ops[reg_idx++]);
              }
            } else if (op.isImm()) {
            } else {
              assert(false);
            }
          }
          os << extra_ops << ");\n";
        });
  };

  const auto case_default = [&](std::string_view mnem_llvm,
                                std::string_view mnem_disarm) {
    if (std::string_view{Name} == mnem_llvm) {
      handle_default(mnem_disarm);
    }
  };

  case_default("DMB", "DMB");

  if (Name == "ADRP") {
    handle_adrp("ADRP");
  }

  const auto case_mov_shift = [&](std::string_view mnem_llvm,
                                  std::string_view mnem_disarm) {
    if (std::string_view{Name} == mnem_llvm) {
      unsigned imm = mi.getOperand(1).getImm();
      unsigned shift = mi.getOperand(2).getImm();
      handle_noimm(mnem_disarm, std::format(", {:#x}, {}", imm, shift / 16));
    }
  };
  case_mov_shift("MOVZWi", "MOVZw_shift");
  case_mov_shift("MOVZXi", "MOVZx_shift");
  case_mov_shift("MOVKWi", "MOVKw_shift");
  case_mov_shift("MOVKXi", "MOVKx_shift");
  case_mov_shift("MOVNWi", "MOVNw_shift");
  case_mov_shift("MOVNXi", "MOVNx_shift");

  const auto case_mem_unsigned = [&](std::string_view mnem_llvm,
                                     std::string_view mnem,
                                     std::string_view mnemu,
                                     unsigned shift) {
    if (std::string_view{Name} == mnem_llvm) {
      // This also handles the default case, which is difficult due to shift.
      handle_mem_imm(mnem, mnemu, shift);
      // TODO: If offset is zero, handle expr with base+index
    }
  };
  case_mem_unsigned("LDRBBui", "LDRBu", "LDURB", 0);
  case_mem_unsigned("LDRSBWui", "LDRSBwu", "LDURSBw", 0);
  case_mem_unsigned("LDRHHui", "LDRHu", "LDURH", 1);
  case_mem_unsigned("LDRSHWui", "LDRSHwu", "LDURSHw", 0);
  case_mem_unsigned("LDRWui", "LDRwu", "LDURw", 2);
  case_mem_unsigned("LDRXui", "LDRxu", "LDURx", 3);
  case_mem_unsigned("PRFMui", "PRFMu", "PRFUMu", 3);
  case_mem_unsigned("LDRBui", "LDRbu", "LDURb", 0);
  case_mem_unsigned("LDRHui", "LDRhu", "LDURh", 1);
  case_mem_unsigned("LDRSui", "LDRsu", "LDURs", 2);
  case_mem_unsigned("LDRDui", "LDRdu", "LDURd", 3);
  case_mem_unsigned("LDRQui", "LDRqu", "LDURq", 4);
  case_mem_unsigned("STRBBui", "STRBu", "STURB", 0);
  case_mem_unsigned("STRHHui", "STRHu", "STURH", 1);
  case_mem_unsigned("STRWui", "STRwu", "STURw", 2);
  case_mem_unsigned("STRXui", "STRxu", "STURx", 3);
  case_mem_unsigned("STRBui", "STRbu", "STURB", 0);
  case_mem_unsigned("STRHui", "STRhu", "STURh", 1);
  case_mem_unsigned("STRSui", "STRsu", "STURs", 2);
  case_mem_unsigned("STRDui", "STRdu", "STURd", 3);
  case_mem_unsigned("STRQui", "STRqu", "STURq", 4);
  if (Name == "LDRSWroW") {
    unsigned sign = mi.getOperand(3).getImm();
    unsigned shift = mi.getOperand(4).getImm();
    std::array<std::string_view, 2> mnems{"LDRSWxr_uxtw", "LDRSWxr_sxtw"};
    handle_noimm(mnems[sign], std::format(", {}", shift));
  }
  if (Name == "LDRSWroX") {
    unsigned sign = mi.getOperand(3).getImm();
    unsigned shift = mi.getOperand(4).getImm();
    std::array<std::string_view, 2> mnems{"LDRSWxr_lsl", "LDRSWxr_sxtx"};
    handle_noimm(mnems[sign], std::format(", {}", shift));
  }

  const auto case_mem_signed =
      [&](std::string_view mnem_llvm, std::string_view mnem, bool with_update) {
        if (std::string_view{Name} == mnem_llvm) {
          // TODO: encode optimization
          handle_mem_simm(mnem, with_update);
        }
      };
  case_mem_signed("STRQpre", "STRq_pre", true);
  case_mem_signed("LDRQpost", "LDRq_post", true);

  case_default("LDPWi", "LDPw");    // TODO: expr with base+off, merge offsets
  case_default("LDPXi", "LDPx");    // TODO: expr with base+off, merge offsets
  case_default("STPWi", "STPw");    // TODO: expr with base+off, merge offsets
  case_default("STPXi", "STPx");    // TODO: expr with base+off, merge offsets
  case_default("STLRB", "STLRB");   // TODO: expr with base+off, merge offsets
  case_default("UBFMWri", "UBFMw"); // TODO: fold zero-extend with load
  case_default("UBFMXri", "UBFMx"); // TODO: fold zero-extend with load
  case_default("SBFMWri", "SBFMw"); // TODO: fold sign-extend with load
  case_default("SBFMXri", "SBFMx"); // TODO: fold sign-extend with load
  case_default("BFMWri", "BFMw");
  case_default("BFMXri", "BFMx");

  const auto case_shift_var = [&](std::string_view mnem_llvm,
                                  std::string_view mnem_disarm,
                                  std::string_view mnem_disarm_imm,
                                  unsigned size) {
    if (std::string_view{Name} == mnem_llvm) {
      handle_shift_imm(mnem_disarm_imm, size);
      handle_default(mnem_disarm);
    }
  };
  case_shift_var("LSLVWr", "LSLVw", "LSLwi", 32);
  case_shift_var("LSLVXr", "LSLVx", "LSLxi", 64);
  case_shift_var("LSRVWr", "LSRVw", "LSRwi", 32);
  case_shift_var("LSRVXr", "LSRVx", "LSRxi", 64);
  case_shift_var("ASRVWr", "ASRVw", "ASRwi", 32);
  case_shift_var("ASRVXr", "ASRVx", "ASRxi", 64);
  case_shift_var("RORVWr", "RORVw", "RORwi", 32);
  case_shift_var("RORVXr", "RORVx", "RORxi", 64);

  const auto case_arith_imm = [&](std::string_view mnem_llvm,
                                  std::string_view mnem_disarm,
                                  unsigned size) {
    if (std::string_view{Name} == mnem_llvm) {
      unsigned imm = mi.getOperand(2).getImm();
      unsigned shift = mi.getOperand(3).getImm();
      handle_noimm(mnem_disarm,
                   std::format(", {}", imm << (shift & (size - 1))));
    }
  };
  case_arith_imm("ADDWri", "ADDwi", 32);
  case_arith_imm("ADDXri", "ADDxi", 64);
  case_arith_imm("ADDSWri", "ADDSwi", 32);
  case_arith_imm("ADDSXri", "ADDSxi", 64);
  case_arith_imm("SUBWri", "SUBwi", 32);
  case_arith_imm("SUBXri", "SUBxi", 64);
  case_arith_imm("SUBSWri", "SUBSwi", 32);
  case_arith_imm("SUBSXri", "SUBSxi", 64);

  const auto case_arith_shift = [&](std::string_view mnem_llvm,
                                    std::string_view mnem_disarm_base,
                                    std::string_view mnem_disarm_imm) {
    if (std::string_view{Name} == mnem_llvm) {
      // TODO: Handle expr with only a shifted index if imm==0
      static const std::array<std::string_view, 4> suffixes{
          "_lsl", "_lsr", "_asr"};
      unsigned imm = mi.getOperand(3).getImm();
      if (imm == 0) { // TODO: apply shift to immediate?
        handle_arith_imm(mnem_disarm_imm);
      }
      std::string mnem;
      llvm::raw_string_ostream(mnem) << mnem_disarm_base << suffixes[imm >> 6];
      // For 32-bit, the correct mask would be 0x1f, but it's always zero.
      handle_noimm(mnem, std::format(", {}", imm & 0x3f));
    }
  };
  case_arith_shift("ADDWrs", "ADDw", "ADDwi");
  case_arith_shift("ADDXrs", "ADDx", "ADDxi");
  case_arith_shift("ADDSWrs", "ADDSw", "ADDSwi");
  case_arith_shift("ADDSXrs", "ADDSx", "ADDSxi");
  case_arith_shift("SUBWrs", "SUBw", "SUBwi");
  case_arith_shift("SUBXrs", "SUBx", "SUBxi");
  case_arith_shift("SUBSWrs", "SUBSw", "SUBSwi");
  case_arith_shift("SUBSXrs", "SUBSx", "SUBSxi");

  const auto case_arith_ext = [&](std::string_view mnem_llvm,
                                  std::string_view mnem_disarm_base) {
    if (std::string_view{Name} == mnem_llvm) {
      // TODO: Handle arithmetic immediates
      // TODO: Handle expr with only a shifted index if imm==0
      static const std::array<std::string_view, 8> suffixes{"_uxtb",
                                                            "_uxth",
                                                            "_uxtw",
                                                            "_uxtx",
                                                            "_sxtb",
                                                            "_sxth",
                                                            "_sxtw",
                                                            "_sxtx"};
      unsigned imm = mi.getOperand(3).getImm();
      std::string mnem;
      llvm::raw_string_ostream(mnem) << mnem_disarm_base << suffixes[imm >> 3];
      handle_noimm(mnem, std::format(", {}", imm & 7));
    }
  };
  case_arith_ext("ADDWrx", "ADDw");
  case_arith_ext("ADDXrx", "ADDx");
  case_arith_ext("ADDSWrx", "ADDSw");
  case_arith_ext("ADDSXrx", "ADDSx");
  case_arith_ext("SUBWrx", "SUBw");
  case_arith_ext("SUBXrx", "SUBx");
  case_arith_ext("SUBSWrx", "SUBSw");
  case_arith_ext("SUBSXrx", "SUBSx");

  case_default("ADCWr", "ADCw");
  case_default("ADCXr", "ADCx");
  case_default("ADCSWr", "ADCSw");
  case_default("ADCSXr", "ADCSx");
  case_default("SBCWr", "SBCw");
  case_default("SBCXr", "SBCx");
  case_default("SBCSWr", "SBCSw");
  case_default("SBCSXr", "SBCSx");
  case_default("MADDWrrr", "MADDw");
  case_default("MADDXrrr", "MADDx");
  case_default("MSUBWrrr", "MSUBw");
  case_default("MSUBXrrr", "MSUBx");
  case_default("UMADDLrrr", "UMADDL");
  case_default("UMSUBLrrr", "UMSUBL");
  case_default("SMADDLrrr", "SMADDL");
  case_default("SMSUBLrrr", "SMSUBL");
  case_default("UMULHrr", "UMULH");
  case_default("SMULHrr", "SMULH");
  case_default("UDIVWr", "UDIVw");
  case_default("UDIVXr", "UDIVx");
  case_default("SDIVWr", "SDIVw");
  case_default("SDIVXr", "SDIVx");
  case_default("CSELWr", "CSELw");
  case_default("CSELXr", "CSELx");
  case_default("CSINCWr", "CSINCw");
  case_default("CSINCXr", "CSINCx");
  case_default("CSINVWr", "CSINVw");
  case_default("CSINVXr", "CSINVx");
  case_default("CSNEGWr", "CSNEGw");
  case_default("CSNEGXr", "CSNEGx");
  const auto case_logical_imm = [&](std::string_view mnem_llvm,
                                    std::string_view mnem_disarm) {
    if (std::string_view{Name} == mnem_llvm) {
      uint64_t imm = imm_logical(0, mi.getOperand(2).getImm());
      handle_noimm(mnem_disarm, std::format(", {:#x}", imm));
    }
  };
  case_logical_imm("ANDWri", "ANDwi");
  case_logical_imm("ANDXri", "ANDxi");
  case_logical_imm("ORRWri", "ORRwi");
  case_logical_imm("ORRXri", "ORRxi");
  case_logical_imm("EORWri", "EORwi");
  case_logical_imm("EORXri", "EORxi");
  case_logical_imm("ANDSWri", "ANDSwi");
  case_logical_imm("ANDSXri", "ANDSxi");

  case_default("RBITWr", "RBITw");
  case_default("RBITXr", "RBITx");
  case_default("REV16Wr", "REV16w");
  case_default("REVWr", "REV32w");
  case_default("REVXr", "REV64x");
  case_default("CLZWr", "CLZw");
  case_default("CLZXr", "CLZx");
  case_default("CLSWr", "CLSw");
  case_default("CLSXr", "CLSx");

  const auto case_logical_shift = [&](std::string_view mnem_llvm,
                                      std::string_view mnem_disarm_base,
                                      std::string_view mnem_disarm_imm,
                                      bool invert) {
    if (std::string_view{Name} == mnem_llvm) {
      // TODO: Handle expr with only a shifted index if imm==0
      static const std::array<std::string_view, 4> suffixes{
          "_lsl", "_lsr", "_asr", "_ror"};
      unsigned imm = mi.getOperand(3).getImm();
      if (imm == 0) { // TODO: apply shift to immediate?
        handle_logical_imm(mnem_disarm_imm, invert);
      }
      std::string mnem;
      llvm::raw_string_ostream(mnem) << mnem_disarm_base << suffixes[imm >> 6];
      // For 32-bit, the correct mask would be 0x1f, but it's always zero.
      handle_noimm(mnem, std::format(", {}", imm & 0x3f));
    }
  };
  case_logical_shift("ANDWrs", "ANDw", "ANDwi", false);
  case_logical_shift("ANDXrs", "ANDx", "ANDxi", false);
  case_logical_shift("ORRWrs", "ORRw", "ORRwi", false);
  case_logical_shift("ORRXrs", "ORRx", "ORRxi", false);
  case_logical_shift("EORWrs", "EORw", "EORwi", false);
  case_logical_shift("EORXrs", "EORx", "EORxi", false);
  case_logical_shift("ANDSWrs", "ANDSw", "ANDSwi", false);
  case_logical_shift("ANDSXrs", "ANDSx", "ANDSxi", false);
  case_logical_shift("BICWrs", "BICw", "ANDwi", true);
  case_logical_shift("BICXrs", "BICx", "ANDxi", true);
  case_logical_shift("ORNWrs", "ORNw", "ORRwi", true);
  case_logical_shift("ORNXrs", "ORNx", "ORRxi", true);
  case_logical_shift("EONWrs", "EONw", "EORwi", true);
  case_logical_shift("EONXrs", "EONx", "EORxi", true);
  case_logical_shift("BICSWrs", "BICSw", "ANDSwi", true);
  case_logical_shift("BICSXrs", "BICSx", "ANDSxi", true);

  case_default("CCMPWr", "CCMPw");
  case_default("CCMPXr", "CCMPx");
  case_default("CCMPWi", "CCMPwi");
  case_default("CCMPXi", "CCMPxi");

  case_default("FCSELSrrr", "FCSELs");
  case_default("FCSELDrrr", "FCSELd");
  case_default("FCMPSrr", "FCMP_s");
  case_default("FCMPDrr", "FCMP_d");
  case_default("FADDSrr", "FADDs");
  case_default("FADDDrr", "FADDd");
  case_default("FSUBSrr", "FSUBs");
  case_default("FSUBDrr", "FSUBd");
  case_default("FMULSrr", "FMULs");
  case_default("FMULDrr", "FMULd");
  case_default("FDIVSrr", "FDIVs");
  case_default("FDIVDrr", "FDIVd");
  case_default("FMINSrr", "FMINs");
  case_default("FMINDrr", "FMINd");
  case_default("FMAXSrr", "FMAXs");
  case_default("FMAXDrr", "FMAXd");
  case_default("FMINNMSrr", "FMINNMs");
  case_default("FMINNMDrr", "FMINNMd");
  case_default("FMAXNMSrr", "FMAXNMs");
  case_default("FMAXNMDrr", "FMAXNMd");
  case_default("FMOVSr", "FMOVs");
  case_default("FMOVDr", "FMOVd");
  case_default("FABSSr", "FABSs");
  case_default("FABSDr", "FABSd");
  case_default("FNEGSr", "FNEGs");
  case_default("FNEGDr", "FNEGd");
  case_default("FSQRTSr", "FSQRTs");
  case_default("FSQRTDr", "FSQRTd");
  case_default("FMADDSrrr", "FMADDs");
  case_default("FMADDDrrr", "FMADDd");
  case_default("FMSUBSrrr", "FMSUBs");
  case_default("FMSUBDrrr", "FMSUBd");
  case_default("FNMADDSrrr", "FNMADDs");
  case_default("FNMADDDrrr", "FNMADDd");
  case_default("FNMSUBSrrr", "FNMSUBs");
  case_default("FNMSUBDrrr", "FNMSUBd");

  case_default("FCVTSDr", "FCVTsd");
  case_default("FCVTDSr", "FCVTds");
  case_default("FCVTZSUWSr", "FCVTZSws"); // TODO: correct?
  case_default("FCVTZUUWSr", "FCVTZUws"); // TODO: correct?
  case_default("FCVTZSUWDr", "FCVTZSwd"); // TODO: correct?
  case_default("FCVTZUUWDr", "FCVTZUwd"); // TODO: correct?
  case_default("FCVTZSUXSr", "FCVTZSxs"); // TODO: correct?
  case_default("FCVTZUUXSr", "FCVTZUxs"); // TODO: correct?
  case_default("FCVTZSUXDr", "FCVTZSxd"); // TODO: correct?
  case_default("FCVTZUUXDr", "FCVTZUxd"); // TODO: correct?
  case_default("SCVTFUWSri", "SCVTFsw");  // TODO: correct?
  case_default("UCVTFUWSri", "UCVTFsw");  // TODO: correct?
  case_default("SCVTFUWDri", "SCVTFdw");  // TODO: correct?
  case_default("UCVTFUWDri", "UCVTFdw");  // TODO: correct?
  case_default("SCVTFUXSri", "SCVTFsx");  // TODO: correct?
  case_default("UCVTFUXSri", "UCVTFsx");  // TODO: correct?
  case_default("SCVTFUXDri", "SCVTFdx");  // TODO: correct?
  case_default("UCVTFUXDri", "UCVTFdx");  // TODO: correct?

  case_default("FADDv2f32", "FADD2s");
  case_default("FADDv4f32", "FADD4s");
  case_default("FADDv2f64", "FADD2d");
  case_default("FSUBv2f32", "FSUB2s");
  case_default("FSUBv4f32", "FSUB4s");
  case_default("FSUBv2f64", "FSUB2d");
  case_default("FMULv2f32", "FMUL2s");
  case_default("FMULv4f32", "FMUL4s");
  case_default("FMULv2f64", "FMUL2d");
  case_default("FDIVv2f32", "FDIV2s");
  case_default("FDIVv4f32", "FDIV4s");
  case_default("FDIVv2f64", "FDIV2d");
  case_default("CMLTv8i8rz", "CMLT8b_zero");
  case_default("CMLTv16i8rz", "CMLT16b_zero");
  case_default("CMLTv4i16rz", "CMLT4h_zero");
  case_default("CMLTv8i16rz", "CMLT8h_zero");
  case_default("CMLTv2i32rz", "CMLT2s_zero");
  case_default("CMLTv4i32rz", "CMLT4s_zero");
  case_default("CMLTv2i64rz", "CMLT2d_zero");
  case_default("CMEQv8i8", "CMEQ8b");
  case_default("CMEQv16i8", "CMEQ16b");
  case_default("CMEQv4i16", "CMEQ4h");
  case_default("CMEQv8i16", "CMEQ8h");
  case_default("CMEQv2i32", "CMEQ2s");
  case_default("CMEQv4i32", "CMEQ4s");
  case_default("CMEQv2i64", "CMEQ2d");
  case_default("CMHIv8i8", "CMHI8b");
  case_default("CMHIv16i8", "CMHI16b");
  case_default("CMHIv4i16", "CMHI4h");
  case_default("CMHIv8i16", "CMHI8h");
  case_default("CMHIv2i32", "CMHI2s");
  case_default("CMHIv4i32", "CMHI4s");
  case_default("CMHIv2i64", "CMHI2d");
  case_default("CMHSv8i8", "CMHS8b");
  case_default("CMHSv16i8", "CMHS16b");
  case_default("CMHSv4i16", "CMHS4h");
  case_default("CMHSv8i16", "CMHS8h");
  case_default("CMHSv2i32", "CMHS2s");
  case_default("CMHSv4i32", "CMHS4s");
  case_default("CMHSv2i64", "CMHS2d");
  case_default("CMGTv8i8", "CMGT8b");
  case_default("CMGTv16i8", "CMGT16b");
  case_default("CMGTv4i16", "CMGT4h");
  case_default("CMGTv8i16", "CMGT8h");
  case_default("CMGTv2i32", "CMGT2s");
  case_default("CMGTv4i32", "CMGT4s");
  case_default("CMGTv2i64", "CMGT2d");
  case_default("CMGEv8i8", "CMGE8b");
  case_default("CMGEv16i8", "CMGE16b");
  case_default("CMGEv4i16", "CMGE4h");
  case_default("CMGEv8i16", "CMGE8h");
  case_default("CMGEv2i32", "CMGE2s");
  case_default("CMGEv4i32", "CMGE4s");
  case_default("CMGEv2i64", "CMGE2d");
  case_default("DUPv16i8lane", "DUP16b");
  case_default("ADDv8i8", "ADD8b");
  case_default("ADDv16i8", "ADD16b");
  case_default("ADDv4i16", "ADD4h");
  case_default("ADDv8i16", "ADD8h");
  case_default("ADDv2i32", "ADD2s");
  case_default("ADDv4i32", "ADD4s");
  case_default("ADDv2i64", "ADD2d");
  case_default("SUBv8i8", "SUB8b");
  case_default("SUBv16i8", "SUB16b");
  case_default("SUBv4i16", "SUB4h");
  case_default("SUBv8i16", "SUB8h");
  case_default("SUBv2i32", "SUB2s");
  case_default("SUBv4i32", "SUB4s");
  case_default("SUBv2i64", "SUB2d");
  case_default("MULv8i8", "MUL8b");
  case_default("MULv16i8", "MUL16b");
  case_default("MULv4i16", "MUL4h");
  case_default("MULv8i16", "MUL8h");
  case_default("MULv2i32", "MUL2s");
  case_default("MULv4i32", "MUL4s");
  case_default("NEGv8i8", "NEG8b");
  case_default("NEGv16i8", "NEG16b");
  case_default("NEGv4i16", "NEG4h");
  case_default("NEGv8i16", "NEG8h");
  case_default("NEGv2i32", "NEG2s");
  case_default("NEGv4i32", "NEG4s");
  case_default("NEGv2i64", "NEG2d");
  case_default("NOTv8i8", "NOT8b");
  case_default("NOTv16i8", "NOT16b");
  case_default("NOTv4i16", "NOT4h");
  case_default("NOTv8i16", "NOT8h");
  case_default("NOTv2i32", "NOT2s");
  case_default("NOTv4i32", "NOT4s");
  case_default("NOTv2i64", "NOT2d");
  case_default("ABSv8i8", "ABS8b");
  case_default("ABSv16i8", "ABS16b");
  case_default("ABSv4i16", "ABS4h");
  case_default("ABSv8i16", "ABS8h");
  case_default("ABSv2i32", "ABS2s");
  case_default("ABSv4i32", "ABS4s");
  case_default("ABSv2i64", "ABS2d");
  case_default("CNTv8i8", "CNT8b");
  case_default("SADDLVv8i8v", "SADDLV8b");
  case_default("UADDLVv8i8v", "UADDLV8b");
  case_default("SMINVv8i8v", "SMINV8b");
  case_default("UMINVv8i8v", "UMINV8b");
  case_default("SMAXVv8i8v", "SMAXV8b");
  case_default("UMAXVv8i8v", "UMAXV8b");
  case_default("ADDVv8i8v", "ADDV8b");
  case_default("ADDVv16i8v", "ADDV16b");
  case_default("ADDVv4i16v", "ADDV4h");
  case_default("ADDVv8i16v", "ADDV8h");
  case_default("ADDVv4i32v", "ADDV4s");
  case_default("ADDPv8i8", "ADDP8b");
  case_default("ADDPv16i8", "ADDP16b");
  case_default("ADDPv4i16", "ADDP4h");
  case_default("ADDPv8i16", "ADDP8h");
  case_default("ADDPv2i32", "ADDP2s");
  case_default("ADDPv4i32", "ADDP4s");
  case_default("ADDPv2i64", "ADDP2d");
  case_default("ADDPv2i64p", "ADDPd");
  case_default("ANDv8i8", "AND8b");
  case_default("ANDv16i8", "AND16b");
  case_default("BICv8i8", "BIC8b");
  case_default("BICv16i8", "BIC16b");
  case_default("ORRv8i8", "ORR8b");
  case_default("ORRv16i8", "ORR16b");
  case_default("ORNv8i8", "ORN8b");
  case_default("ORNv16i8", "ORN16b");
  case_default("EORv8i8", "EOR8b");
  case_default("EORv16i8", "EOR16b");
  case_default("BSLv8i8", "BSL8b");
  case_default("BSLv16i8", "BSL16b");
  case_default("BITv8i8", "BIT8b");
  case_default("BITv16i8", "BIT16b");
  case_default("BIFv8i8", "BIF8b");
  case_default("BIFv16i8", "BIF16b");
  case_default("EXTv8i8", "EXT8b");
  case_default("EXTv16i8", "EXT16b");
  case_default("ZIP1v8i8", "ZIP1_8b");
  case_default("ZIP1v16i8", "ZIP1_16b");
  case_default("ZIP1v4i16", "ZIP1_4h");
  case_default("ZIP1v8i16", "ZIP1_8h");
  case_default("ZIP1v2i32", "ZIP1_2s");
  case_default("ZIP1v4i32", "ZIP1_4s");
  case_default("ZIP1v2i64", "ZIP1_2d");
  case_default("USHLv8i8", "USHL8b");
  case_default("USHLv16i8", "USHL16b");
  case_default("USHLv4i16", "USHL4h");
  case_default("USHLv8i16", "USHL8h");
  case_default("USHLv2i32", "USHL2s");
  case_default("USHLv4i32", "USHL4s");
  case_default("USHLv2i64", "USHL2d");
  case_default("SSHLv8i8", "SSHL8b");
  case_default("SSHLv16i8", "SSHL16b");
  case_default("SSHLv4i16", "SSHL4h");
  case_default("SSHLv8i16", "SSHL8h");
  case_default("SSHLv2i32", "SSHL2s");
  case_default("SSHLv4i32", "SSHL4s");
  case_default("SSHLv2i64", "SSHL2d");
  case_default("SHLv8i8_shift", "SHL8bi");
  case_default("SHLv16i8_shift", "SHL16bi");
  case_default("SHLv4i16_shift", "SHL4hi");
  case_default("SHLv8i16_shift", "SHL8hi");
  case_default("SHLv2i32_shift", "SHL2si");
  case_default("SHLv4i32_shift", "SHL4si");
  case_default("SHLv2i64_shift", "SHL2di");
  case_default("XTNv8i8", "XTN_8b");
  case_default("XTNv4i16", "XTN_4h");
  case_default("XTNv2i32", "XTN_2s");
  case_default("FNEGv2f32", "FNEG2s");
  case_default("FNEGv4f32", "FNEG4s");
  case_default("FNEGv2f64", "FNEG2d");
  if (Name == "MOVIv2d_ns") {
    uint64_t imm = 0;
    unsigned op = mi.getOperand(1).getImm();
    for (int i = 0; i < 8; i++) {
      imm |= op & (1 << i) ? uint64_t{0xff} << 8 * i : 0;
    }
    handle_noimm("MOVI2d", std::format(", {:#x}", imm));
  }
  if (Name == "MVNIv4i32") {
    uint32_t byte = mi.getOperand(1).getImm();
    uint64_t imm = ~(byte << mi.getOperand(2).getImm());
    handle_noimm("MOVI2d", std::format(", {:#x}", imm | imm << 32));
  }

  case_default("FMOVSWr", "FMOVws");
  case_default("FMOVDXr", "FMOVxd");
  case_default("FMOVWSr", "FMOVsw");
  case_default("FMOVXDr", "FMOVdx");
  case_default("UMOVvi64", "UMOVxd");
  case_default("INSvi64gpr", "INSdx");

  case_default("CASB", "CASB");
  case_default("CASH", "CASH");
  case_default("CASW", "CASw");
  case_default("CASX", "CASx");
  case_default("CASAB", "CASAB");
  case_default("CASAH", "CASAH");
  case_default("CASAW", "CASAw");
  case_default("CASAX", "CASAx");
  case_default("CASLB", "CASLB");
  case_default("CASLH", "CASLH");
  case_default("CASLW", "CASLw");
  case_default("CASLX", "CASLx");
  case_default("CASALB", "CASALB");
  case_default("CASALH", "CASALH");
  case_default("CASALW", "CASALw");
  case_default("CASALX", "CASALx");

  // atomic memory operations (LDADD) have their source and dest operands
  // swapped
  const auto case_atomic_mem_op = [&](std::string_view mnem_llvm,
                                      std::string_view mnem) {
    if (std::string_view{Name} != mnem_llvm) {
      return;
    }
    candidates.emplace_back([mnem](llvm::raw_ostream &os,
                                   const llvm::MachineInstr &mi,
                                   std::span<const std::string> ops) {
      os << "    ASMD(" << mnem;
      os << ", " << format_reg(mi.getOperand(1), ops[1]);
      os << ", " << format_reg(mi.getOperand(0), ops[0]);
      os << ", " << format_reg(mi.getOperand(2), ops[2]);
      os << ");\n";
    });
  };
  case_atomic_mem_op("SWPB", "SWPB");
  case_atomic_mem_op("SWPH", "SWPH");
  case_atomic_mem_op("SWPW", "SWPw");
  case_atomic_mem_op("SWPX", "SWPx");
  case_atomic_mem_op("SWPAB", "SWPAB");
  case_atomic_mem_op("SWPAH", "SWPAH");
  case_atomic_mem_op("SWPAW", "SWPAw");
  case_atomic_mem_op("SWPAX", "SWPAx");
  case_atomic_mem_op("SWPLB", "SWPLB");
  case_atomic_mem_op("SWPLH", "SWPLH");
  case_atomic_mem_op("SWPLW", "SWPLw");
  case_atomic_mem_op("SWPLX", "SWPLx");
  case_atomic_mem_op("SWPALB", "SWPALB");
  case_atomic_mem_op("SWPALH", "SWPALH");
  case_atomic_mem_op("SWPALW", "SWPALw");
  case_atomic_mem_op("SWPALX", "SWPALx");

  // Good that Arm no describes their architecture as "RISC".
  case_atomic_mem_op("LDADDB", "LDADDB");
  case_atomic_mem_op("LDADDH", "LDADDH");
  case_atomic_mem_op("LDADDW", "LDADDw");
  case_atomic_mem_op("LDADDX", "LDADDx");
  case_atomic_mem_op("LDADDAB", "LDADDAB");
  case_atomic_mem_op("LDADDAH", "LDADDAH");
  case_atomic_mem_op("LDADDAW", "LDADDAw");
  case_atomic_mem_op("LDADDAX", "LDADDAx");
  case_atomic_mem_op("LDADDLB", "LDADDLB");
  case_atomic_mem_op("LDADDLH", "LDADDLH");
  case_atomic_mem_op("LDADDLW", "LDADDLw");
  case_atomic_mem_op("LDADDLX", "LDADDLx");
  case_atomic_mem_op("LDADDALB", "LDADDALB");
  case_atomic_mem_op("LDADDALH", "LDADDALH");
  case_atomic_mem_op("LDADDALW", "LDADDALw");
  case_atomic_mem_op("LDADDALX", "LDADDALx");
  case_atomic_mem_op("LDCLRB", "LDCLRB");
  case_atomic_mem_op("LDCLRH", "LDCLRH");
  case_atomic_mem_op("LDCLRW", "LDCLRw");
  case_atomic_mem_op("LDCLRX", "LDCLRx");
  case_atomic_mem_op("LDCLRAB", "LDCLRAB");
  case_atomic_mem_op("LDCLRAH", "LDCLRAH");
  case_atomic_mem_op("LDCLRAW", "LDCLRAw");
  case_atomic_mem_op("LDCLRAX", "LDCLRAx");
  case_atomic_mem_op("LDCLRLB", "LDCLRLB");
  case_atomic_mem_op("LDCLRLH", "LDCLRLH");
  case_atomic_mem_op("LDCLRLW", "LDCLRLw");
  case_atomic_mem_op("LDCLRLX", "LDCLRLx");
  case_atomic_mem_op("LDCLRALB", "LDCLRALB");
  case_atomic_mem_op("LDCLRALH", "LDCLRALH");
  case_atomic_mem_op("LDCLRALW", "LDCLRALw");
  case_atomic_mem_op("LDCLRALX", "LDCLRALx");
  case_atomic_mem_op("LDEORB", "LDEORB");
  case_atomic_mem_op("LDEORH", "LDEORH");
  case_atomic_mem_op("LDEORW", "LDEORw");
  case_atomic_mem_op("LDEORX", "LDEORx");
  case_atomic_mem_op("LDEORAB", "LDEORAB");
  case_atomic_mem_op("LDEORAH", "LDEORAH");
  case_atomic_mem_op("LDEORAW", "LDEORAw");
  case_atomic_mem_op("LDEORAX", "LDEORAx");
  case_atomic_mem_op("LDEORLB", "LDEORLB");
  case_atomic_mem_op("LDEORLH", "LDEORLH");
  case_atomic_mem_op("LDEORLW", "LDEORLw");
  case_atomic_mem_op("LDEORLX", "LDEORLx");
  case_atomic_mem_op("LDEORALB", "LDEORALB");
  case_atomic_mem_op("LDEORALH", "LDEORALH");
  case_atomic_mem_op("LDEORALW", "LDEORALw");
  case_atomic_mem_op("LDEORALX", "LDEORALx");
  case_atomic_mem_op("LDSETB", "LDSETB");
  case_atomic_mem_op("LDSETH", "LDSETH");
  case_atomic_mem_op("LDSETW", "LDSETw");
  case_atomic_mem_op("LDSETX", "LDSETx");
  case_atomic_mem_op("LDSETAB", "LDSETAB");
  case_atomic_mem_op("LDSETAH", "LDSETAH");
  case_atomic_mem_op("LDSETAW", "LDSETAw");
  case_atomic_mem_op("LDSETAX", "LDSETAx");
  case_atomic_mem_op("LDSETLB", "LDSETLB");
  case_atomic_mem_op("LDSETLH", "LDSETLH");
  case_atomic_mem_op("LDSETLW", "LDSETLw");
  case_atomic_mem_op("LDSETLX", "LDSETLx");
  case_atomic_mem_op("LDSETALB", "LDSETALB");
  case_atomic_mem_op("LDSETALH", "LDSETALH");
  case_atomic_mem_op("LDSETALW", "LDSETALw");
  case_atomic_mem_op("LDSETALX", "LDSETALx");
  case_atomic_mem_op("LDSMAXB", "LDSMAXB");
  case_atomic_mem_op("LDSMAXH", "LDSMAXH");
  case_atomic_mem_op("LDSMAXW", "LDSMAXw");
  case_atomic_mem_op("LDSMAXX", "LDSMAXx");
  case_atomic_mem_op("LDSMAXAB", "LDSMAXAB");
  case_atomic_mem_op("LDSMAXAH", "LDSMAXAH");
  case_atomic_mem_op("LDSMAXAW", "LDSMAXAw");
  case_atomic_mem_op("LDSMAXAX", "LDSMAXAx");
  case_atomic_mem_op("LDSMAXLB", "LDSMAXLB");
  case_atomic_mem_op("LDSMAXLH", "LDSMAXLH");
  case_atomic_mem_op("LDSMAXLW", "LDSMAXLw");
  case_atomic_mem_op("LDSMAXLX", "LDSMAXLx");
  case_atomic_mem_op("LDSMAXALB", "LDSMAXALB");
  case_atomic_mem_op("LDSMAXALH", "LDSMAXALH");
  case_atomic_mem_op("LDSMAXALW", "LDSMAXALw");
  case_atomic_mem_op("LDSMAXALX", "LDSMAXALx");
  case_atomic_mem_op("LDSMINB", "LDSMINB");
  case_atomic_mem_op("LDSMINH", "LDSMINH");
  case_atomic_mem_op("LDSMINW", "LDSMINw");
  case_atomic_mem_op("LDSMINX", "LDSMINx");
  case_atomic_mem_op("LDSMINAB", "LDSMINAB");
  case_atomic_mem_op("LDSMINAH", "LDSMINAH");
  case_atomic_mem_op("LDSMINAW", "LDSMINAw");
  case_atomic_mem_op("LDSMINAX", "LDSMINAx");
  case_atomic_mem_op("LDSMINLB", "LDSMINLB");
  case_atomic_mem_op("LDSMINLH", "LDSMINLH");
  case_atomic_mem_op("LDSMINLW", "LDSMINLw");
  case_atomic_mem_op("LDSMINLX", "LDSMINLx");
  case_atomic_mem_op("LDSMINALB", "LDSMINALB");
  case_atomic_mem_op("LDSMINALH", "LDSMINALH");
  case_atomic_mem_op("LDSMINALW", "LDSMINALw");
  case_atomic_mem_op("LDSMINALX", "LDSMINALx");
  case_atomic_mem_op("LDUMAXB", "LDUMAXB");
  case_atomic_mem_op("LDUMAXH", "LDUMAXH");
  case_atomic_mem_op("LDUMAXW", "LDUMAXw");
  case_atomic_mem_op("LDUMAXX", "LDUMAXx");
  case_atomic_mem_op("LDUMAXAB", "LDUMAXAB");
  case_atomic_mem_op("LDUMAXAH", "LDUMAXAH");
  case_atomic_mem_op("LDUMAXAW", "LDUMAXAw");
  case_atomic_mem_op("LDUMAXAX", "LDUMAXAx");
  case_atomic_mem_op("LDUMAXLB", "LDUMAXLB");
  case_atomic_mem_op("LDUMAXLH", "LDUMAXLH");
  case_atomic_mem_op("LDUMAXLW", "LDUMAXLw");
  case_atomic_mem_op("LDUMAXLX", "LDUMAXLx");
  case_atomic_mem_op("LDUMAXALB", "LDUMAXALB");
  case_atomic_mem_op("LDUMAXALH", "LDUMAXALH");
  case_atomic_mem_op("LDUMAXALW", "LDUMAXALw");
  case_atomic_mem_op("LDUMAXALX", "LDUMAXALx");
  case_atomic_mem_op("LDUMINB", "LDUMINB");
  case_atomic_mem_op("LDUMINH", "LDUMINH");
  case_atomic_mem_op("LDUMINW", "LDUMINw");
  case_atomic_mem_op("LDUMINX", "LDUMINx");
  case_atomic_mem_op("LDUMINAB", "LDUMINAB");
  case_atomic_mem_op("LDUMINAH", "LDUMINAH");
  case_atomic_mem_op("LDUMINAW", "LDUMINAw");
  case_atomic_mem_op("LDUMINAX", "LDUMINAx");
  case_atomic_mem_op("LDUMINLB", "LDUMINLB");
  case_atomic_mem_op("LDUMINLH", "LDUMINLH");
  case_atomic_mem_op("LDUMINLW", "LDUMINLw");
  case_atomic_mem_op("LDUMINLX", "LDUMINLx");
  case_atomic_mem_op("LDUMINALB", "LDUMINALB");
  case_atomic_mem_op("LDUMINALH", "LDUMINALH");
  case_atomic_mem_op("LDUMINALW", "LDUMINALw");
  case_atomic_mem_op("LDUMINALX", "LDUMINALx");

  case_default("LDARB", "LDARB");
  case_default("LDARH", "LDARH");
  case_default("LDARW", "LDARw");
  case_default("LDARX", "LDARx");
  case_default("LADRB", "STLRB");
  case_default("STLRH", "STLRH");
  case_default("STLRW", "STLRw");
  case_default("STLRX", "STLRx");

  if (candidates.empty()) {
    llvm::errs() << "ERROR: unhandled instruction " << Name << "\n";
    assert(false);
    exit(1);
  }
}

std::optional<std::pair<unsigned, unsigned>>
    EncodingTargetArm64::is_move(const llvm::MachineInstr &mi) {
  const llvm::TargetMachine &TM = mi.getMF()->getTarget();
  llvm::StringRef name = TM.getMCInstrInfo()->getName(mi.getOpcode());
  if (name != "ORRXrs") {
    return std::nullopt;
  }
  if (mi.getOperand(3).getImm()) {
    return std::nullopt;
  }
  const auto &tri = *mi.getMF()->getRegInfo().getTargetRegisterInfo();
  if (tri.getName(mi.getOperand(1).getReg()) != "XZR"sv) {
    return std::nullopt;
  }
  if (tri.getName(mi.getOperand(2).getReg()) == "XZR"sv) {
    return std::nullopt;
  }
  return std::make_pair(0, 2);
}

} // namespace tpde_encgen::arm64
