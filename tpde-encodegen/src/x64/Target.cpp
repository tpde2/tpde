// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "x64/Target.hpp"

#include <format>
#include <string>
#include <string_view>

#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineOperand.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/Target/TargetMachine.h>


using namespace std::literals;

namespace tpde_encgen::x64 {

void EncodingTargetX64::get_inst_candidates(
    llvm::MachineInstr &mi, llvm::SmallVectorImpl<MICandidate> &candidates) {
  const llvm::TargetMachine &TM = mi.getMF()->getTarget();
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCInstrDesc &MCID = MCII.get(mi.getOpcode());
  (void)MCID;

  llvm::StringRef Name = MCII.getName(mi.getOpcode());

  const auto handle_immrepl = [&](std::string_view mnem,
                                  unsigned replop_idx,
                                  int memop_start = -1) {
    if (memop_start >= 0 && mi.getOperand(memop_start).getReg().isValid() &&
        mi.getOperand(memop_start + 3).isImm()) {
      bool has_idx = mi.getOperand(memop_start + 2).getReg().isValid();
      unsigned sc = has_idx ? mi.getOperand(memop_start + 1).getImm() : 0;
      std::string_view idx = has_idx ? "FE_AX" : "FE_NOREG";
      auto off = mi.getOperand(memop_start + 3).getImm();
      auto cond = std::format("FE_MEM(FE_NOREG, {}, {}, {})", sc, idx, off);

      candidates.emplace_back(
          MICandidate::Cond{replop_idx, "encodeable_as_imm32_sext", ""},
          MICandidate::Cond{(unsigned)memop_start, "encodeable_with", cond},
          [has_idx, mnem, memop_start, replop_idx](
              llvm::raw_ostream &os,
              const llvm::MachineInstr &mi,
              std::span<const std::string> ops) {
            os << "    ASMD(" << mnem;
            unsigned reg_idx = 0;
            for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
              const auto &op = mi.getOperand(i);
              if (i == (unsigned)memop_start) {
                if (has_idx) { // Need to replace index register
                  os << ", FE_MEM(" << ops[reg_idx] << ".base, " << ops[reg_idx]
                     << ".scale, " << ops[reg_idx + 1] << ", " << ops[reg_idx]
                     << ".off)";
                } else {
                  os << ", " << ops[reg_idx];
                }
                reg_idx += 3;
                i += 4;
              } else if (i == replop_idx) {
                assert(op.isReg());
                os << ", " << ops[reg_idx++];
              } else if (op.isReg()) {
                if (op.isTied() && op.isUse()) {
                  continue;
                }
                os << ", " << ops[reg_idx++];
              } else if (op.isImm()) {
                if (mnem.starts_with("CMOV") || mnem.starts_with("SET")) {
                  continue;
                }
                os << ", " << op.getImm();
              }
            }
            os << ");\n";
          });
    }
    // TODO: better code, less copy-paste, all cases, etc.
    // Difficult, because at condition time, we don't know the operands.
    if (memop_start >= 0 && mi.getOperand(memop_start + 0).getReg().isValid() &&
        mi.getOperand(memop_start + 2).getReg().isValid() &&
        mi.getOperand(memop_start + 1).getImm() == 1 &&
        mi.getOperand(memop_start + 3).isImm()) {
      auto off = mi.getOperand(memop_start + 3).getImm();
      auto cond = std::format("FE_MEM(FE_AX, 0, FE_NOREG, {})", off);

      candidates.emplace_back(
          MICandidate::Cond{replop_idx, "encodeable_as_imm32_sext", ""},
          MICandidate::Cond{(unsigned)memop_start + 2, "encodeable_with", cond},
          [mnem, memop_start, replop_idx](llvm::raw_ostream &os,
                                          const llvm::MachineInstr &mi,
                                          std::span<const std::string> ops) {
            os << "    ASMD(" << mnem;
            unsigned reg_idx = 0;
            for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
              const auto &op = mi.getOperand(i);
              if (i == (unsigned)memop_start) {
                os << ", FE_MEM(" << ops[reg_idx] << ", " << ops[reg_idx + 1]
                   << ".scale, " << ops[reg_idx + 1] << ".idx, "
                   << ops[reg_idx + 1] << ".off)";
                reg_idx += 3;
                i += 4;
              } else if (i == replop_idx) {
                assert(op.isReg());
                os << ", " << ops[reg_idx++];
              } else if (op.isReg()) {
                if (op.isTied() && op.isUse()) {
                  continue;
                }
                os << ", " << ops[reg_idx++];
              } else if (op.isImm()) {
                if (mnem.starts_with("CMOV") || mnem.starts_with("SET")) {
                  continue;
                }
                os << ", " << op.getImm();
              }
            }
            os << ");\n";
          });
    }
    candidates.emplace_back(
        replop_idx,
        "encodeable_as_imm32_sext",
        "",
        [mnem, replop_idx, memop_start](llvm::raw_ostream &os,
                                        const llvm::MachineInstr &mi,
                                        std::span<const std::string> ops) {
          os << "    ASMD(" << mnem;
          unsigned reg_idx = 0;
          std::string_view cp_sym = "";
          [[maybe_unused]] bool has_imm = false;
          for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
            const auto &op = mi.getOperand(i);
            if (i == (unsigned)memop_start && mi.getOperand(i + 3).isCPI()) {
              // base is rip; idx is noreg
              cp_sym = ops[reg_idx + 2];
              reg_idx += 3;
              i += 4;
              os << ", FE_MEM(FE_IP, 0, FE_NOREG, 0)";
            } else if (i == (unsigned)memop_start) {
              std::string_view base =
                  op.getReg().isValid() ? ops[reg_idx] : "FE_NOREG"sv;
              unsigned sc = mi.getOperand(i + 2).getReg().isValid()
                                ? mi.getOperand(i + 1).getImm()
                                : 0;
              std::string_view idx = mi.getOperand(i + 2).getReg().isValid()
                                         ? ops[reg_idx + 1]
                                         : "FE_NOREG"sv;
              auto off = mi.getOperand(i + 3).getImm();
              assert(!mi.getOperand(i + 4).getReg().isValid());
              reg_idx += 3;
              i += 4;
              os << ", FE_MEM(" << base << ", " << sc << ", " << idx << ", "
                 << off << ")";
            } else if (i == replop_idx) {
              assert(op.isReg());
              has_imm = true;
              os << ", " << ops[reg_idx++];
            } else if (op.isReg()) {
              if (op.isTied() && op.isUse()) {
                continue;
              }
              os << ", " << ops[reg_idx++];
            } else if (op.isImm()) {
              if (mnem.starts_with("CMOV") || mnem.starts_with("SET")) {
                continue;
              }
              has_imm = true;
              os << ", " << op.getImm();
            }
          }
          if (replop_idx >= mi.getNumExplicitOperands()) { // SHL etc. with CL
            has_imm = true;
            os << ", " << ops[reg_idx++];
          }
          os << ");\n";
          if (!cp_sym.empty()) {
            assert(!has_imm && "CPI with imm unsupported!");
            os << "        "
                  "derived()->reloc_text("
               << cp_sym
               << ", R_X86_64_PC32, "
                  "derived()->text_writer.offset() - 4, -4);\n";
          }
        });
  };
  const auto handle_memrepl = [&](std::string_view mnem,
                                  unsigned replop_idx,
                                  unsigned align = 1) {
    candidates.emplace_back(
        replop_idx,
        "encodeable_as_mem",
        std::format("{}", align),
        [mnem, replop_idx](llvm::raw_ostream &os,
                           const llvm::MachineInstr &mi,
                           std::span<const std::string> ops) {
          os << "    ASMD(" << mnem;
          unsigned reg_idx = 0;
          for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
            const auto &op = mi.getOperand(i);
            if (i == replop_idx) {
              assert(op.isReg());
              os << ", " << ops[reg_idx++];
            } else if (op.isReg()) {
              if (op.isTied() && op.isUse()) {
                continue;
              }
              os << ", " << ops[reg_idx++];
            } else if (op.isImm()) {
              if (mnem.starts_with("CMOV") || mnem.starts_with("SET")) {
                continue;
              }
              os << ", " << op.getImm();
            }
          }
          os << ");\n";
        });
  };
  const auto handle_default = [&](std::string_view mnem,
                                  int memop_start = -1,
                                  std::string_view extra_ops = ""sv) {
    if (memop_start >= 0 && mi.getOperand(memop_start).getReg().isValid() &&
        mi.getOperand(memop_start + 3).isImm()) {
      bool has_idx = mi.getOperand(memop_start + 2).getReg().isValid();
      unsigned sc = has_idx ? mi.getOperand(memop_start + 1).getImm() : 0;
      std::string_view idx = has_idx ? "FE_AX" : "FE_NOREG";
      auto off = mi.getOperand(memop_start + 3).getImm();
      auto cond = std::format("FE_MEM(FE_NOREG, {}, {}, {})", sc, idx, off);

      candidates.emplace_back(
          memop_start,
          "encodeable_with",
          cond,
          [has_idx, mnem, memop_start, extra_ops](
              llvm::raw_ostream &os,
              const llvm::MachineInstr &mi,
              std::span<const std::string> ops) {
            os << "    ASMD(" << mnem;
            unsigned reg_idx = 0;
            for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
              const auto &op = mi.getOperand(i);
              if (i == (unsigned)memop_start) {
                if (has_idx) { // Need to replace index register
                  os << ", FE_MEM(" << ops[reg_idx] << ".base, " << ops[reg_idx]
                     << ".scale, " << ops[reg_idx + 1] << ", " << ops[reg_idx]
                     << ".off)";
                } else {
                  os << ", " << ops[reg_idx];
                }
                reg_idx += 3;
                i += 4;
              } else if (op.isReg()) {
                if (op.isTied() && op.isUse()) {
                  continue;
                }
                os << ", " << ops[reg_idx++];
              } else if (op.isImm()) {
                if (mnem.starts_with("CMOV") || mnem.starts_with("SET")) {
                  continue;
                }
                os << ", " << op.getImm();
              }
            }
            os << extra_ops << ");\n";
          });
    }
    // TODO: better code, less copy-paste, all cases, etc.
    // Difficult, because at condition time, we don't know the operands.
    if (memop_start >= 0 && mi.getOperand(memop_start + 0).getReg().isValid() &&
        mi.getOperand(memop_start + 2).getReg().isValid() &&
        mi.getOperand(memop_start + 1).getImm() == 1 &&
        mi.getOperand(memop_start + 3).isImm()) {
      auto off = mi.getOperand(memop_start + 3).getImm();
      auto cond = std::format("FE_MEM(FE_AX, 0, FE_NOREG, {})", off);

      candidates.emplace_back(
          memop_start + 2,
          "encodeable_with",
          cond,
          [mnem, memop_start, extra_ops](llvm::raw_ostream &os,
                                         const llvm::MachineInstr &mi,
                                         std::span<const std::string> ops) {
            os << "    ASMD(" << mnem;
            unsigned reg_idx = 0;
            for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
              const auto &op = mi.getOperand(i);
              if (i == (unsigned)memop_start) {
                os << ", FE_MEM(" << ops[reg_idx] << ", " << ops[reg_idx + 1]
                   << ".scale, " << ops[reg_idx + 1] << ".idx, "
                   << ops[reg_idx + 1] << ".off)";
                reg_idx += 3;
                i += 4;
              } else if (op.isReg()) {
                if (op.isTied() && op.isUse()) {
                  continue;
                }
                os << ", " << ops[reg_idx++];
              } else if (op.isImm()) {
                if (mnem.starts_with("CMOV") || mnem.starts_with("SET")) {
                  continue;
                }
                os << ", " << op.getImm();
              }
            }
            os << extra_ops << ");\n";
          });
    }
    candidates.emplace_back(
        [mnem, memop_start, extra_ops](llvm::raw_ostream &os,
                                       const llvm::MachineInstr &mi,
                                       std::span<const std::string> ops) {
          os << "    ASMD(" << mnem;
          unsigned reg_idx = 0;
          std::string_view cp_sym = "";
          [[maybe_unused]] bool has_imm = false;
          for (unsigned i = 0, n = mi.getNumExplicitOperands(); i != n; i++) {
            const auto &op = mi.getOperand(i);
            if (i == (unsigned)memop_start && mi.getOperand(i + 3).isCPI()) {
              // base is rip; idx is noreg
              cp_sym = ops[reg_idx + 2];
              reg_idx += 3;
              i += 4;
              os << ", FE_MEM(FE_IP, 0, FE_NOREG, 0)";
            } else if (i == (unsigned)memop_start) {
              std::string_view base =
                  op.getReg().isValid() ? ops[reg_idx] : "FE_NOREG"sv;
              unsigned sc = mi.getOperand(i + 2).getReg().isValid()
                                ? mi.getOperand(i + 1).getImm()
                                : 0;
              std::string_view idx = mi.getOperand(i + 2).getReg().isValid()
                                         ? ops[reg_idx + 1]
                                         : "FE_NOREG"sv;
              auto off = mi.getOperand(i + 3).getImm();
              assert(!mi.getOperand(i + 4).getReg().isValid());
              reg_idx += 3;
              i += 4;
              os << ", FE_MEM(" << base << ", " << sc << ", " << idx << ", "
                 << off << ")";
            } else if (op.isReg()) {
              if (op.isTied() && op.isUse()) {
                continue;
              }
              os << ", " << ops[reg_idx++];
            } else if (op.isImm()) {
              if (mnem.starts_with("CMOV") || mnem.starts_with("SET")) {
                continue;
              }
              has_imm = true;
              os << ", " << std::format("{:#x}", op.getImm());
            }
          }
          os << extra_ops << ");\n";
          if (!cp_sym.empty()) {
            assert(!has_imm && "CPI with imm unsupported!");
            os << "        "
                  "derived()->reloc_text("
               << cp_sym
               << ", R_X86_64_PC32, "
                  "derived()->text_writer.offset() - 4, -4);\n";
          }
        });
  };
  // xchg and xadd have their operands swapped in LLVM.
  const auto handle_xchg_mem = [&](std::string_view mnem, int memop_start) {
    // TODO: address candidate
    candidates.emplace_back([mnem,
                             memop_start](llvm::raw_ostream &os,
                                          const llvm::MachineInstr &mi,
                                          std::span<const std::string> ops) {
      std::string_view base =
          mi.getOperand(memop_start).getReg().isValid() ? ops[1] : "FE_NOREG"sv;
      unsigned sc = mi.getOperand(memop_start + 2).getReg().isValid()
                        ? mi.getOperand(memop_start + 1).getImm()
                        : 0;
      std::string_view idx = mi.getOperand(memop_start + 2).getReg().isValid()
                                 ? ops[2]
                                 : "FE_NOREG"sv;
      auto off = mi.getOperand(memop_start + 3).getImm();
      assert(!mi.getOperand(memop_start + 4).getReg().isValid());

      os << "    ASMD(" << mnem << ", FE_MEM(" << base << ", " << sc << ", "
         << idx << ", " << off << "), " << ops[0] << ");\n";
    });
  };

  const auto case_default = [&](std::string_view llvm_mnem,
                                unsigned memop_start,
                                std::string_view fd_mnem) {
    if (std::string_view(Name) == llvm_mnem) {
      handle_default(fd_mnem, memop_start);
    }
  };
  const auto handle_rm = [&](std::string_view llvm_mnem_r,
                             std::string_view llvm_mnem_m,
                             unsigned memop_start,
                             std::string_view fd_mnem_r,
                             std::string_view fd_mnem_m,
                             unsigned align = 1) {
    if (std::string_view(Name) == llvm_mnem_r) {
      handle_memrepl(fd_mnem_m, memop_start, align);
      handle_default(fd_mnem_r);
    }
    if (std::string_view(Name) == llvm_mnem_m) {
      handle_default(fd_mnem_m, memop_start);
    }
  };
  const auto handle_rmi = [&](std::string_view llvm_mnem_r,
                              std::string_view llvm_mnem_m,
                              std::string_view llvm_mnem_i,
                              unsigned replop_idx,
                              std::string_view fd_mnem_r,
                              std::string_view fd_mnem_m,
                              std::string_view fd_mnem_i) {
    if (std::string_view(Name) == llvm_mnem_r) {
      handle_immrepl(fd_mnem_i, replop_idx);
      handle_memrepl(fd_mnem_m, replop_idx);
      handle_default(fd_mnem_r);
    }
    if (std::string_view(Name) == llvm_mnem_m) {
      handle_default(fd_mnem_m, replop_idx);
    }
    if (std::string_view(Name) == llvm_mnem_i) {
      handle_default(fd_mnem_i);
    }
  };
  const auto handle_ri = [&](std::string_view llvm_mnem_r,
                             std::string_view llvm_mnem_i,
                             unsigned replop_idx,
                             unsigned memop_start,
                             std::string_view fd_mnem_r,
                             std::string_view fd_mnem_i) {
    if (std::string_view(Name) == llvm_mnem_r) {
      handle_immrepl(fd_mnem_i, replop_idx, memop_start);
      handle_default(fd_mnem_r, memop_start);
    }
    if (std::string_view(Name) == llvm_mnem_i) {
      handle_default(fd_mnem_i, memop_start);
    }
  };
  const auto handle_ri_shift = [&](std::string_view llvm_mnem_r,
                                   std::string_view llvm_mnem_i,
                                   unsigned replop_idx,
                                   unsigned memop_start,
                                   std::string_view fd_mnem_r,
                                   std::string_view fd_mnem_i) {
    if (std::string_view(Name) == llvm_mnem_r) {
      handle_immrepl(fd_mnem_i, replop_idx, memop_start);
      handle_default(fd_mnem_r, memop_start, ", FE_CX");
    }
    if (std::string_view(Name) == llvm_mnem_i) {
      handle_default(fd_mnem_i, memop_start);
    }
  };

  // clang-format off
  handle_rm("MOVZX32rr8", "MOVZX32rm8", 1, "MOVZXr32r8", "MOVZXr32m8");
  handle_rm("MOVZX32rr16", "MOVZX32rm16", 1, "MOVZXr32r16", "MOVZXr32m16");
  handle_rm("MOVSX32rr8", "MOVSX32rm8", 1, "MOVSXr32r8", "MOVSXr32m8");
  handle_rm("MOVSX32rr16", "MOVSX32rm16", 1, "MOVSXr32r16", "MOVSXr32m16");
  handle_rm("MOVSX64rr8", "MOVSX64rm8", 1, "MOVSXr64r8", "MOVSXr64m8");
  handle_rm("MOVSX64rr16", "MOVSX64rm16", 1, "MOVSXr64r16", "MOVSXr64m16");
  handle_rm("MOVSX64rr32", "MOVSX64rm32", 1, "MOVSXr64r32", "MOVSXr64m32");

  handle_rmi("MOV8rr", "MOV8rm", "MOV8ri", 1, "MOV8rr", "MOV8rm", "MOV8ri");
  handle_rmi("MOV16rr", "MOV16rm", "MOV16ri", 1, "MOV16rr", "MOV16rm", "MOV16ri");
  handle_rmi("MOV32rr", "MOV32rm", "MOV32ri", 1, "MOV32rr", "MOV32rm", "MOV32ri");
  handle_rmi("MOV64rr", "MOV64rm", "MOV64ri32", 1, "MOV64rr", "MOV64rm", "MOV64ri");
  if (Name == "MOV64ri") {
    handle_default("MOV64ri");
  }
  handle_ri("MOV8mr", "MOV8mi", 5, 0, "MOV8mr", "MOV8mi");
  handle_ri("MOV16mr", "MOV16mi", 5, 0, "MOV16mr", "MOV16mi");
  handle_ri("MOV32mr", "MOV32mi", 5, 0, "MOV32mr", "MOV32mi");
  handle_ri("MOV64mr", "MOV64mi32", 5, 0, "MOV64mr", "MOV64mi");
  handle_rmi("ADD8rr", "ADD8rm", "ADD8ri", 2, "ADD8rr", "ADD8rm", "ADD8ri");
  handle_rmi("ADD16rr", "ADD16rm", "ADD16ri", 2, "ADD16rr", "ADD16rm", "ADD16ri");
  handle_rmi("ADD32rr", "ADD32rm", "ADD32ri", 2, "ADD32rr", "ADD32rm", "ADD32ri");
  handle_rmi("ADD64rr", "ADD64rm", "ADD64ri32", 2, "ADD64rr", "ADD64rm", "ADD64ri");
  handle_ri("ADD8mr", "ADD8mi", 5, 0, "ADD8mr", "ADD8mi");
  handle_ri("ADD16mr", "ADD16mi", 5, 0, "ADD16mr", "ADD16mi");
  handle_ri("ADD32mr", "ADD32mi", 5, 0, "ADD32mr", "ADD32mi");
  handle_ri("ADD64mr", "ADD64mi32", 5, 0, "ADD64mr", "ADD64mi");
  handle_rmi("OR8rr", "OR8rm", "OR8ri", 2, "OR8rr", "OR8rm", "OR8ri");
  handle_rmi("OR16rr", "OR16rm", "OR16ri", 2, "OR16rr", "OR16rm", "OR16ri");
  handle_rmi("OR32rr", "OR32rm", "OR32ri", 2, "OR32rr", "OR32rm", "OR32ri");
  handle_rmi("OR64rr", "OR64rm", "OR64ri32", 2, "OR64rr", "OR64rm", "OR64ri");
  handle_ri("OR8mr", "OR8mi", 5, 0, "OR8mr", "OR8mi");
  handle_ri("OR16mr", "OR16mi", 5, 0, "OR16mr", "OR16mi");
  handle_ri("OR32mr", "OR32mi", 5, 0, "OR32mr", "OR32mi");
  handle_ri("OR64mr", "OR64mi32", 5, 0, "OR64mr", "OR64mi");
  handle_rmi("ADC8rr", "ADC8rm", "ADC8ri", 2, "ADC8rr", "ADC8rm", "ADC8ri");
  handle_rmi("ADC16rr", "ADC16rm", "ADC16ri", 2, "ADC16rr", "ADC16rm", "ADC16ri");
  handle_rmi("ADC32rr", "ADC32rm", "ADC32ri", 2, "ADC32rr", "ADC32rm", "ADC32ri");
  handle_rmi("ADC64rr", "ADC64rm", "ADC64ri32", 2, "ADC64rr", "ADC64rm", "ADC64ri");
  handle_ri("ADC8mr", "ADC8mi", 5, 0, "ADC8mr", "ADC8mi");
  handle_ri("ADC16mr", "ADC16mi", 5, 0, "ADC16mr", "ADC16mi");
  handle_ri("ADC32mr", "ADC32mi", 5, 0, "ADC32mr", "ADC32mi");
  handle_ri("ADC64mr", "ADC64mi32", 5, 0, "ADC64mr", "ADC64mi");
  handle_rmi("SBB8rr", "SBB8rm", "SBB8ri", 2, "SBB8rr", "SBB8rm", "SBB8ri");
  handle_rmi("SBB16rr", "SBB16rm", "SBB16ri", 2, "SBB16rr", "SBB16rm", "SBB16ri");
  handle_rmi("SBB32rr", "SBB32rm", "SBB32ri", 2, "SBB32rr", "SBB32rm", "SBB32ri");
  handle_rmi("SBB64rr", "SBB64rm", "SBB64ri32", 2, "SBB64rr", "SBB64rm", "SBB64ri");
  handle_ri("SBB8mr", "SBB8mi", 5, 0, "SBB8mr", "SBB8mi");
  handle_ri("SBB16mr", "SBB16mi", 5, 0, "SBB16mr", "SBB16mi");
  handle_ri("SBB32mr", "SBB32mi", 5, 0, "SBB32mr", "SBB32mi");
  handle_ri("SBB64mr", "SBB64mi32", 5, 0, "SBB64mr", "SBB64mi");
  handle_rmi("AND8rr", "AND8rm", "AND8ri", 2, "AND8rr", "AND8rm", "AND8ri");
  handle_rmi("AND16rr", "AND16rm", "AND16ri", 2, "AND16rr", "AND16rm", "AND16ri");
  handle_rmi("AND32rr", "AND32rm", "AND32ri", 2, "AND32rr", "AND32rm", "AND32ri");
  handle_rmi("AND64rr", "AND64rm", "AND64ri32", 2, "AND64rr", "AND64rm", "AND64ri");
  handle_ri("AND8mr", "AND8mi", 5, 0, "AND8mr", "AND8mi");
  handle_ri("AND16mr", "AND16mi", 5, 0, "AND16mr", "AND16mi");
  handle_ri("AND32mr", "AND32mi", 5, 0, "AND32mr", "AND32mi");
  handle_ri("AND64mr", "AND64mi32", 5, 0, "AND64mr", "AND64mi");
  handle_rmi("SUB8rr", "SUB8rm", "SUB8ri", 2, "SUB8rr", "SUB8rm", "SUB8ri");
  handle_rmi("SUB16rr", "SUB16rm", "SUB16ri", 2, "SUB16rr", "SUB16rm", "SUB16ri");
  handle_rmi("SUB32rr", "SUB32rm", "SUB32ri", 2, "SUB32rr", "SUB32rm", "SUB32ri");
  handle_rmi("SUB64rr", "SUB64rm", "SUB64ri32", 2, "SUB64rr", "SUB64rm", "SUB64ri");
  handle_ri("SUB8mr", "SUB8mi", 5, 0, "SUB8mr", "SUB8mi");
  handle_ri("SUB16mr", "SUB16mi", 5, 0, "SUB16mr", "SUB16mi");
  handle_ri("SUB32mr", "SUB32mi", 5, 0, "SUB32mr", "SUB32mi");
  handle_ri("SUB64mr", "SUB64mi32", 5, 0, "SUB64mr", "SUB64mi");
  handle_rmi("XOR8rr", "XOR8rm", "XOR8ri", 2, "XOR8rr", "XOR8rm", "XOR8ri");
  handle_rmi("XOR16rr", "XOR16rm", "XOR16ri", 2, "XOR16rr", "XOR16rm", "XOR16ri");
  handle_rmi("XOR32rr", "XOR32rm", "XOR32ri", 2, "XOR32rr", "XOR32rm", "XOR32ri");
  handle_rmi("XOR64rr", "XOR64rm", "XOR64ri32", 2, "XOR64rr", "XOR64rm", "XOR64ri");
  handle_ri("XOR8mr", "XOR8mi", 5, 0, "XOR8mr", "XOR8mi");
  handle_ri("XOR16mr", "XOR16mi", 5, 0, "XOR16mr", "XOR16mi");
  handle_ri("XOR32mr", "XOR32mi", 5, 0, "XOR32mr", "XOR32mi");
  handle_ri("XOR64mr", "XOR64mi32", 5, 0, "XOR64mr", "XOR64mi");
  handle_rmi("CMP8rr", "CMP8rm", "CMP8ri", 1, "CMP8rr", "CMP8rm", "CMP8ri");
  handle_rmi("CMP16rr", "CMP16rm", "CMP16ri", 1, "CMP16rr", "CMP16rm", "CMP16ri");
  handle_rmi("CMP32rr", "CMP32rm", "CMP32ri", 1, "CMP32rr", "CMP32rm", "CMP32ri");
  handle_rmi("CMP64rr", "CMP64rm", "CMP64ri32", 1, "CMP64rr", "CMP64rm", "CMP64ri");
  handle_ri("CMP8mr", "CMP8mi", 5, 0, "CMP8mr", "CMP8mi");
  handle_ri("CMP16mr", "CMP16mi", 5, 0, "CMP16mr", "CMP16mi");
  handle_ri("CMP32mr", "CMP32mi", 5, 0, "CMP32mr", "CMP32mi");
  handle_ri("CMP64mr", "CMP64mi32", 5, 0, "CMP64mr", "CMP64mi");
  // TODO: TESTmr
  handle_ri("TEST8rr", "TEST8ri", 1, -1, "TEST8rr", "TEST8ri");
  handle_ri("TEST16rr", "TEST16ri", 1, -1, "TEST16rr", "TEST16ri");
  handle_ri("TEST32rr", "TEST32ri", 1, -1, "TEST32rr", "TEST32ri");
  handle_ri("TEST64rr", "TEST64ri32", 1, -1, "TEST64rr", "TEST64ri");
  handle_ri("TEST8mr", "TEST8mi", 5, 0, "TEST8mr", "TEST8mi");
  handle_ri("TEST16mr", "TEST16mi", 5, 0, "TEST16mr", "TEST16mi");
  handle_ri("TEST32mr", "TEST32mi", 5, 0, "TEST32mr", "TEST32mi");
  handle_ri("TEST64mr", "TEST64mi32", 5, 0, "TEST64mr", "TEST64mi");

  handle_ri_shift("SHR8rCL", "SHR8ri", 3, -1, "SHR8rr", "SHR8ri");
  handle_ri_shift("SHR16rCL", "SHR16ri", 3, -1, "SHR16rr", "SHR16ri");
  handle_ri_shift("SHR32rCL", "SHR32ri", 3, -1, "SHR32rr", "SHR32ri");
  handle_ri_shift("SHR64rCL", "SHR64ri", 3, -1, "SHR64rr", "SHR64ri");
  handle_ri_shift("SAR8rCL", "SAR8ri", 3, -1, "SAR8rr", "SAR8ri");
  handle_ri_shift("SAR16rCL", "SAR16ri", 3, -1, "SAR16rr", "SAR16ri");
  handle_ri_shift("SAR32rCL", "SAR32ri", 3, -1, "SAR32rr", "SAR32ri");
  handle_ri_shift("SAR64rCL", "SAR64ri", 3, -1, "SAR64rr", "SAR64ri");
  handle_ri_shift("SHL8rCL", "SHL8ri", 3, -1, "SHL8rr", "SHL8ri");
  handle_ri_shift("SHL16rCL", "SHL16ri", 3, -1, "SHL16rr", "SHL16ri");
  handle_ri_shift("SHL32rCL", "SHL32ri", 3, -1, "SHL32rr", "SHL32ri");
  handle_ri_shift("SHL64rCL", "SHL64ri", 3, -1, "SHL64rr", "SHL64ri");
  handle_ri_shift("ROR8rCL", "ROR8ri", 3, -1, "ROR8rr", "ROR8ri");
  handle_ri_shift("ROR16rCL", "ROR16ri", 3, -1, "ROR16rr", "ROR16ri");
  handle_ri_shift("ROR32rCL", "ROR32ri", 3, -1, "ROR32rr", "ROR32ri");
  handle_ri_shift("ROR64rCL", "ROR64ri", 3, -1, "ROR64rr", "ROR64ri");
  handle_ri_shift("ROL8rCL", "ROL8ri", 3, -1, "ROL8rr", "ROL8ri");
  handle_ri_shift("ROL16rCL", "ROL16ri", 3, -1, "ROL16rr", "ROL16ri");
  handle_ri_shift("ROL32rCL", "ROL32ri", 3, -1, "ROL32rr", "ROL32ri");
  handle_ri_shift("ROL64rCL", "ROL64ri", 3, -1, "ROL64rr", "ROL64ri");
  handle_ri_shift("SHRD16rrCL", "SHRD16rri", 4, -1, "SHRD16rrr", "SHRD16rri");
  handle_ri_shift("SHRD32rrCL", "SHRD32rri", 4, -1, "SHRD32rrr", "SHRD32rri");
  handle_ri_shift("SHRD64rrCL", "SHRD64rri", 4, -1, "SHRD64rrr", "SHRD64rri");
  handle_ri_shift("SHLD16rrCL", "SHLD16rri", 4, -1, "SHLD16rrr", "SHLD16rri");
  handle_ri_shift("SHLD32rrCL", "SHLD32rri", 4, -1, "SHLD32rrr", "SHLD32rri");
  handle_ri_shift("SHLD64rrCL", "SHLD64rri", 4, -1, "SHLD64rrr", "SHLD64rri");

  handle_rm("MUL8r", "MUL8m", 0, "MUL8r", "MUL8m");
  handle_rm("MUL16r", "MUL16m", 0, "MUL16r", "MUL16m");
  handle_rm("MUL32r", "MUL32m", 0, "MUL32r", "MUL32m");
  handle_rm("MUL64r", "MUL64m", 0, "MUL64r", "MUL64m");
  handle_rm("IMUL8r", "IMUL8m", 0, "IMUL8r", "IMUL8m");
  handle_rm("IMUL16r", "IMUL16m", 0, "IMUL16r", "IMUL16m");
  handle_rm("IMUL32r", "IMUL32m", 0, "IMUL32r", "IMUL32m");
  handle_rm("IMUL64r", "IMUL64m", 0, "IMUL64r", "IMUL64m");
  handle_rm("DIV8r", "DIV8m", 0, "DIV8r", "DIV8m");
  handle_rm("DIV16r", "DIV16m", 0, "DIV16r", "DIV16m");
  handle_rm("DIV32r", "DIV32m", 0, "DIV32r", "DIV32m");
  handle_rm("DIV64r", "DIV64m", 0, "DIV64r", "DIV64m");
  handle_rm("IDIV8r", "IDIV8m", 0, "IDIV8r", "IDIV8m");
  handle_rm("IDIV16r", "IDIV16m", 0, "IDIV16r", "IDIV16m");
  handle_rm("IDIV32r", "IDIV32m", 0, "IDIV32r", "IDIV32m");
  handle_rm("IDIV64r", "IDIV64m", 0, "IDIV64r", "IDIV64m");

  handle_rm("BSF16rr", "BSF16rm", 1, "BSF16rr", "BSF16rm");
  handle_rm("BSF32rr", "BSF32rm", 1, "BSF32rr", "BSF32rm");
  handle_rm("BSF64rr", "BSF64rm", 1, "BSF64rr", "BSF64rm");
  handle_rm("BSR16rr", "BSR16rm", 1, "BSR16rr", "BSR16rm");
  handle_rm("BSR32rr", "BSR32rm", 1, "BSR32rr", "BSR32rm");
  handle_rm("BSR64rr", "BSR64rm", 1, "BSR64rr", "BSR64rm");

  // TODO: memrepl for first mov operand
  handle_rm("MOVSSrr", "MOVSSrm_alt", 1, "SSE_MOVSSrr", "SSE_MOVSSrm");
  handle_rm("MOVSDrr", "MOVSDrm_alt", 1, "SSE_MOVSDrr", "SSE_MOVSDrm");
  handle_rm("MOVAPSrr", "MOVAPSrm", 1, "SSE_MOVAPSrr", "SSE_MOVAPSrm", 16);
  handle_rm("MOVAPDrr", "MOVAPDrm", 1, "SSE_MOVAPDrr", "SSE_MOVAPDrm", 16);
  handle_rm("MOVUPSrr", "MOVUPSrm", 1, "SSE_MOVUPSrr", "SSE_MOVUPSrm");
  handle_rm("MOVUPDrr", "MOVUPDrm", 1, "SSE_MOVUPDrr", "SSE_MOVUPDrm");
  handle_rm("MOVDQArr", "MOVDQArm", 1, "SSE_MOVDQArr", "SSE_MOVDQArm", 16);
  handle_rm("MOVDQUrr", "MOVDQUrm", 1, "SSE_MOVDQUrr", "SSE_MOVDQUrm");
  handle_rm("ADDSSrr", "ADDSSrm", 2, "SSE_ADDSSrr", "SSE_ADDSSrm");
  handle_rm("ADDSDrr", "ADDSDrm", 2, "SSE_ADDSDrr", "SSE_ADDSDrm");
  handle_rm("ADDPSrr", "ADDPSrm", 2, "SSE_ADDPSrr", "SSE_ADDPSrm", 16);
  handle_rm("ADDPDrr", "ADDPDrm", 2, "SSE_ADDPDrr", "SSE_ADDPDrm", 16);
  handle_rm("SUBSSrr", "SUBSSrm", 2, "SSE_SUBSSrr", "SSE_SUBSSrm");
  handle_rm("SUBSDrr", "SUBSDrm", 2, "SSE_SUBSDrr", "SSE_SUBSDrm");
  handle_rm("SUBPSrr", "SUBPSrm", 2, "SSE_SUBPSrr", "SSE_SUBPSrm", 16);
  handle_rm("SUBPDrr", "SUBPDrm", 2, "SSE_SUBPDrr", "SSE_SUBPDrm", 16);
  handle_rm("MULSSrr", "MULSSrm", 2, "SSE_MULSSrr", "SSE_MULSSrm");
  handle_rm("MULSDrr", "MULSDrm", 2, "SSE_MULSDrr", "SSE_MULSDrm");
  handle_rm("MULPSrr", "MULPSrm", 2, "SSE_MULPSrr", "SSE_MULPSrm", 16);
  handle_rm("MULPDrr", "MULPDrm", 2, "SSE_MULPDrr", "SSE_MULPDrm", 16);
  handle_rm("DIVSSrr", "DIVSSrm", 2, "SSE_DIVSSrr", "SSE_DIVSSrm");
  handle_rm("DIVSDrr", "DIVSDrm", 2, "SSE_DIVSDrr", "SSE_DIVSDrm");
  handle_rm("DIVPSrr", "DIVPSrm", 2, "SSE_DIVPSrr", "SSE_DIVPSrm", 16);
  handle_rm("DIVPDrr", "DIVPDrm", 2, "SSE_DIVPDrr", "SSE_DIVPDrm", 16);
  handle_rm("MINSSrr", "MINSSrm", 2, "SSE_MINSSrr", "SSE_MINSSrm");
  handle_rm("MINSDrr", "MINSDrm", 2, "SSE_MINSDrr", "SSE_MINSDrm");
  handle_rm("MINPSrr", "MINPSrm", 2, "SSE_MINPSrr", "SSE_MINPSrm", 16);
  handle_rm("MINPDrr", "MINPDrm", 2, "SSE_MINPDrr", "SSE_MINPDrm", 16);
  handle_rm("MAXSSrr", "MAXSSrm", 2, "SSE_MAXSSrr", "SSE_MAXSSrm");
  handle_rm("MAXSDrr", "MAXSDrm", 2, "SSE_MAXSDrr", "SSE_MAXSDrm");
  handle_rm("MAXPSrr", "MAXPSrm", 2, "SSE_MAXPSrr", "SSE_MAXPSrm", 16);
  handle_rm("MAXPDrr", "MAXPDrm", 2, "SSE_MAXPDrr", "SSE_MAXPDrm", 16);
  // commutative variants of MIN/MAX
  handle_rm("MINCSSrr", "MINCSSrm", 2, "SSE_MINSSrr", "SSE_MINSSrm");
  handle_rm("MINCSDrr", "MINCSDrm", 2, "SSE_MINSDrr", "SSE_MINSDrm");
  handle_rm("MINCPSrr", "MINCPSrm", 2, "SSE_MINPSrr", "SSE_MINPSrm", 16);
  handle_rm("MINCPDrr", "MINCPDrm", 2, "SSE_MINPDrr", "SSE_MINPDrm", 16);
  handle_rm("MAXCSSrr", "MAXCSSrm", 2, "SSE_MAXSSrr", "SSE_MAXSSrm");
  handle_rm("MAXCSDrr", "MAXCSDrm", 2, "SSE_MAXSDrr", "SSE_MAXSDrm");
  handle_rm("MAXCPSrr", "MAXCPSrm", 2, "SSE_MAXPSrr", "SSE_MAXPSrm", 16);
  handle_rm("MAXCPDrr", "MAXCPDrm", 2, "SSE_MAXPDrr", "SSE_MAXPDrm", 16);
  handle_rm("CMPSSrri", "CMPSSrmi", 2, "SSE_CMPSSrri", "SSE_CMPSSrmi");
  handle_rm("CMPSDrri", "CMPSDrmi", 2, "SSE_CMPSDrri", "SSE_CMPSDrmi");
  handle_rm("CMPPSrri", "CMPPSrmi", 2, "SSE_CMPPSrri", "SSE_CMPPSrmi", 16);
  handle_rm("CMPPDrri", "CMPPDrmi", 2, "SSE_CMPPDrri", "SSE_CMPPDrmi", 16);
  handle_rm("SQRTSSr", "SQRTSSm", 1, "SSE_SQRTSSrr", "SSE_SQRTSSrm");
  handle_rm("SQRTSDr", "SQRTSDm", 1, "SSE_SQRTSDrr", "SSE_SQRTSDrm");
  handle_rm("SQRTPSr", "SQRTPSm", 1, "SSE_SQRTPSrr", "SSE_SQRTPSrm", 16);
  handle_rm("SQRTPDr", "SQRTPDm", 1, "SSE_SQRTPDrr", "SSE_SQRTPDrm", 16);
  handle_rm("ANDPSrr", "ANDPSrm", 2, "SSE_ANDPSrr", "SSE_ANDPSrm", 16);
  handle_rm("ANDPDrr", "ANDPDrm", 2, "SSE_ANDPDrr", "SSE_ANDPDrm", 16);
  handle_rm("XORPSrr", "XORPSrm", 2, "SSE_XORPSrr", "SSE_XORPSrm", 16);
  handle_rm("XORPDrr", "XORPDrm", 2, "SSE_XORPDrr", "SSE_XORPDrm", 16);
  handle_rm("ORPSrr", "ORPSrm", 2, "SSE_ORPSrr", "SSE_ORPSrm", 16);
  handle_rm("ORPDrr", "ORPDrm", 2, "SSE_ORPDrr", "SSE_ORPDrm", 16);
  handle_rm("ANDNPSrr", "ANDNPSrm", 2, "SSE_ANDNPSrr", "SSE_ANDNPSrm", 16);
  handle_rm("ANDNPDrr", "ANDNPDrm", 2, "SSE_ANDNPDrr", "SSE_ANDNPDrm", 16);
  handle_rm("COMISSrr", "COMISSrm", 1, "SSE_COMISSrr", "SSE_COMISSrm");
  handle_rm("COMISDrr", "COMISDrm", 1, "SSE_COMISDrr", "SSE_COMISDrm");
  handle_rm("UCOMISSrr", "UCOMISSrm", 1, "SSE_UCOMISSrr", "SSE_UCOMISSrm");
  handle_rm("UCOMISDrr", "UCOMISDrm", 1, "SSE_UCOMISDrr", "SSE_UCOMISDrm");

  handle_rm("CVTTPS2DQrr", "CVTTPS2DQrm", 1, "SSE_CVTTPS2DQrr", "SSE_CVTTPS2DQrm");

  handle_rm("PANDrr", "PANDrm", 2, "SSE_PANDrr", "SSE_PANDrm", 16);
  handle_rm("PANDNrr", "PANDNrm", 2, "SSE_PANDNrr", "SSE_PANDNrm", 16);
  handle_rm("PORrr", "PORrm", 2, "SSE_PORrr", "SSE_PORrm", 16);
  handle_rm("PXORrr", "PXORrm", 2, "SSE_PXORrr", "SSE_PXORrm", 16);
  handle_rm("PADDBrr", "PADDBrm", 2, "SSE_PADDBrr", "SSE_PADDBrm", 16);
  handle_rm("PADDWrr", "PADDWrm", 2, "SSE_PADDWrr", "SSE_PADDWrm", 16);
  handle_rm("PADDDrr", "PADDDrm", 2, "SSE_PADDDrr", "SSE_PADDDrm", 16);
  handle_rm("PADDQrr", "PADDQrm", 2, "SSE_PADDQrr", "SSE_PADDQrm", 16);
  handle_rm("PSUBBrr", "PSUBBrm", 2, "SSE_PSUBBrr", "SSE_PSUBBrm", 16);
  handle_rm("PSUBWrr", "PSUBWrm", 2, "SSE_PSUBWrr", "SSE_PSUBWrm", 16);
  handle_rm("PSUBDrr", "PSUBDrm", 2, "SSE_PSUBDrr", "SSE_PSUBDrm", 16);
  handle_rm("PSUBQrr", "PSUBQrm", 2, "SSE_PSUBQrr", "SSE_PSUBQrm", 16);
  handle_rm("PSUBSBrr", "PSUBSBrm", 2, "SSE_PSUBSBrr", "SSE_PSUBSBrm", 16);
  handle_rm("PSUBSWrr", "PSUBSWrm", 2, "SSE_PSUBSWrr", "SSE_PSUBSWrm", 16);
  handle_rm("PSUBUSBrr", "PSUBUSBrm", 2, "SSE_PSUBUSBrr", "SSE_PSUBUSBrm", 16);
  handle_rm("PSUBUSWrr", "PSUBUSWrm", 2, "SSE_PSUBUSWrr", "SSE_PSUBUSWrm", 16);
  handle_rm("PMULLWrr", "PMULLWrm", 2, "SSE_PMULLWrr", "SSE_PMULLWrm", 16);
  handle_rm("PMULLDrr", "PMULLDrm", 2, "SSE_PMULLDrr", "SSE_PMULLDrm", 16);
  handle_rm("PMULDQrr", "PMULDQrm", 2, "SSE_PMULDQrr", "SSE_PMULDQrm", 16);
  handle_rm("PMULUDQrr", "PMULUDQrm", 2, "SSE_PMULUDQrr", "SSE_PMULUDQrm", 16);

  handle_rm("PCMPEQBrr", "PCMPEQBrm", 2, "SSE_PCMPEQBrr", "SSE_PCMPEQBrm", 16);
  handle_rm("PCMPEQWrr", "PCMPEQWrm", 2, "SSE_PCMPEQWrr", "SSE_PCMPEQWrm", 16);
  handle_rm("PCMPEQDrr", "PCMPEQDrm", 2, "SSE_PCMPEQDrr", "SSE_PCMPEQDrm", 16);
  handle_rm("PCMPEQQrr", "PCMPEQQrm", 2, "SSE_PCMPEQQrr", "SSE_PCMPEQQrm", 16);
  handle_rm("PCMPGTBrr", "PCMPGTBrm", 2, "SSE_PCMPGTBrr", "SSE_PCMPGTBrm", 16);
  handle_rm("PCMPGTWrr", "PCMPGTWrm", 2, "SSE_PCMPGTWrr", "SSE_PCMPGTWrm", 16);
  handle_rm("PCMPGTDrr", "PCMPGTDrm", 2, "SSE_PCMPGTDrr", "SSE_PCMPGTDrm", 16);
  handle_rm("PCMPGTQrr", "PCMPGTQrm", 2, "SSE_PCMPGTQrr", "SSE_PCMPGTQrm", 16);

  handle_rm("PMINUBrr", "PMINUBrm", 2, "SSE_PMINUBrr", "SSE_PMINUBrm", 16);
  handle_rm("PMINUWrr", "PMINUWrm", 2, "SSE_PMINUWrr", "SSE_PMINUWrm", 16);
  handle_rm("PMINUDrr", "PMINUDrm", 2, "SSE_PMINUDrr", "SSE_PMINUDrm", 16);
  handle_rm("PMINUQrr", "PMINUQrm", 2, "SSE_PMINUQrr", "SSE_PMINUQrm", 16);
  handle_rm("PMINSBrr", "PMINSBrm", 2, "SSE_PMINSBrr", "SSE_PMINSBrm", 16);
  handle_rm("PMINSWrr", "PMINSWrm", 2, "SSE_PMINSWrr", "SSE_PMINSWrm", 16);
  handle_rm("PMINSDrr", "PMINSDrm", 2, "SSE_PMINSDrr", "SSE_PMINSDrm", 16);
  handle_rm("PMINSQrr", "PMINSQrm", 2, "SSE_PMINSQrr", "SSE_PMINSQrm", 16);
  handle_rm("PMAXUBrr", "PMAXUBrm", 2, "SSE_PMAXUBrr", "SSE_PMAXUBrm", 16);
  handle_rm("PMAXUWrr", "PMAXUWrm", 2, "SSE_PMAXUWrr", "SSE_PMAXUWrm", 16);
  handle_rm("PMAXUDrr", "PMAXUDrm", 2, "SSE_PMAXUDrr", "SSE_PMAXUDrm", 16);
  handle_rm("PMAXUQrr", "PMAXUQrm", 2, "SSE_PMAXUQrr", "SSE_PMAXUQrm", 16);
  handle_rm("PMAXSBrr", "PMAXSBrm", 2, "SSE_PMAXSBrr", "SSE_PMAXSBrm", 16);
  handle_rm("PMAXSWrr", "PMAXSWrm", 2, "SSE_PMAXSWrr", "SSE_PMAXSWrm", 16);
  handle_rm("PMAXSDrr", "PMAXSDrm", 2, "SSE_PMAXSDrr", "SSE_PMAXSDrm", 16);
  handle_rm("PMAXSQrr", "PMAXSQrm", 2, "SSE_PMAXSQrr", "SSE_PMAXSQrm", 16);

  handle_rm("PSLLWrr", "PSLLWrm", 2, "SSE_PSLLWrr", "SSE_PSLLWrm", 16);
  handle_rm("PSLLDrr", "PSLLDrm", 2, "SSE_PSLLDrr", "SSE_PSLLDrm", 16);
  handle_rm("PSLLQrr", "PSLLQrm", 2, "SSE_PSLLQrr", "SSE_PSLLQrm", 16);
  handle_rm("PSRLWrr", "PSRLWrm", 2, "SSE_PSRLWrr", "SSE_PSRLWrm", 16);
  handle_rm("PSRLDrr", "PSRLDrm", 2, "SSE_PSRLDrr", "SSE_PSRLDrm", 16);
  handle_rm("PSRLQrr", "PSRLQrm", 2, "SSE_PSRLQrr", "SSE_PSRLQrm", 16);
  handle_rm("PSRAWrr", "PSRAWrm", 2, "SSE_PSRAWrr", "SSE_PSRAWrm", 16);
  handle_rm("PSRADrr", "PSRADrm", 2, "SSE_PSRADrr", "SSE_PSRADrm", 16);
  handle_rm("PSRAQrr", "PSRAQrm", 2, "SSE_PSRAQrr", "SSE_PSRAQrm", 16);
  case_default("PSLLWri", -1, "SSE_PSLLWri");
  case_default("PSLLDri", -1, "SSE_PSLLDri");
  case_default("PSLLQri", -1, "SSE_PSLLQri");
  case_default("PSRLWri", -1, "SSE_PSRLWri");
  case_default("PSRLDri", -1, "SSE_PSRLDri");
  case_default("PSRLQri", -1, "SSE_PSRLQri");
  case_default("PSRAWri", -1, "SSE_PSRAWri");
  case_default("PSRADri", -1, "SSE_PSRADri");
  case_default("PSRAQri", -1, "SSE_PSRAQri");

  handle_rm("SHUFPSrri", "SHUFPSrmi", 2, "SSE_SHUFPSrri", "SSE_SHUFPSrmi", 16);
  handle_rm("SHUFPDrri", "SHUFPDrmi", 2, "SSE_SHUFPDrri", "SSE_SHUFPDrmi", 16);
  handle_rm("PSHUFDri", "PSHUFDmi", 1, "SSE_PSHUFDrri", "SSE_PSHUFDrmi", 16);
  handle_rm("PSHUFLWri", "PSHUFLWmi", 1, "SSE_PSHUFLWrri", "SSE_PSHUFLWrmi", 16);
  handle_rm("PSHUFHWri", "PSHUFHWmi", 1, "SSE_PSHUFHWrri", "SSE_PSHUFHWrmi", 16);
  handle_rm("PACKUSWBrr", "PACKUSWBrm", 2, "SSE_PACKUSWBrr", "SSE_PACKUSWBrm", 16);
  handle_rm("PACKUSDWrr", "PACKUSDWrm", 2, "SSE_PACKUSDWrr", "SSE_PACKUSDWrm", 16);
  handle_rm("PACKSSWBrr", "PACKSSWBrm", 2, "SSE_PACKSSWBrr", "SSE_PACKSSWBrm", 16);
  handle_rm("PACKSSDWrr", "PACKSSDWrm", 2, "SSE_PACKSSDWrr", "SSE_PACKSSDWrm", 16);
  handle_rm("PUNPCKLBWrr", "PUNPCKLBWrm", 2, "SSE_PUNPCKLBWrr", "SSE_PUNPCKLBWrm", 16);
  handle_rm("PUNPCKLWDrr", "PUNPCKLWDrm", 2, "SSE_PUNPCKLWDrr", "SSE_PUNPCKLWDrm", 16);
  handle_rm("PUNPCKLDQrr", "PUNPCKLDQrm", 2, "SSE_PUNPCKLDQrr", "SSE_PUNPCKLDQrm", 16);
  handle_rm("PUNPCKLQDQrr", "PUNPCKLQDQrm", 2, "SSE_PUNPCKLQDQrr", "SSE_PUNPCKLQDQrm", 16);
  handle_rm("PUNPCKHBWrr", "PUNPCKHBWrm", 2, "SSE_PUNPCKHBWrr", "SSE_PUNPCKHBWrm", 16);
  handle_rm("PUNPCKHWDrr", "PUNPCKHWDrm", 2, "SSE_PUNPCKHWDrr", "SSE_PUNPCKHWDrm", 16);
  handle_rm("PUNPCKHDQrr", "PUNPCKHDQrm", 2, "SSE_PUNPCKHDQrr", "SSE_PUNPCKHDQrm", 16);
  handle_rm("PUNPCKHQDQrr", "PUNPCKHQDQrm", 2, "SSE_PUNPCKHQDQrr", "SSE_PUNPCKHQDQrm", 16);

  case_default("PMOVMSKBrr", -1, "SSE_PMOVMSKBrr");
  case_default("MOVMSKPSrr", -1, "SSE_MOVMSKPSrr");
  case_default("MOVMSKPDrr", -1, "SSE_MOVMSKPDrr");

  case_default("MFENCE", -1, "MFENCE");
  // clang-format on

  if (Name == "LCMPXCHG64") {
    handle_default("LOCK_CMPXCHG64mr", 0);

  } else if (Name == "VMOVAPSYrm") {
    handle_default("VMOVAPS256rm", 1);
  } else if (Name == "VMOVAPSZrm") {
    handle_default("VMOVAPS512rm", 1);
  } else if (Name == "VMOVUPSYrm") {
    handle_default("VMOVUPS256rm", 1);
  } else if (Name == "VMOVUPSZrm") {
    handle_default("VMOVUPS512rm", 1);
  } else if (Name == "MOVSSmr") {
    handle_default("SSE_MOVSSmr", 0);
  } else if (Name == "MOVSDmr") {
    handle_default("SSE_MOVSDmr", 0);
  } else if (Name == "MOVAPSmr") {
    handle_default("SSE_MOVAPSmr", 0);
  } else if (Name == "VMOVAPSYmr") {
    handle_default("VMOVAPS256mr", 0);
  } else if (Name == "VMOVAPSZmr") {
    handle_default("VMOVAPS512mr", 0);
  } else if (Name == "MOVUPSmr") {
    handle_default("SSE_MOVUPSmr", 0);
  } else if (Name == "VMOVUPSYmr") {
    handle_default("VMOVUPS256mr", 0);
  } else if (Name == "VMOVUPSZmr") {
    handle_default("VMOVUPS512mr", 0);

  } else if (Name == "CVTSD2SSrr") {
    handle_memrepl("SSE_CVTSD2SSrm", 1);
    handle_default("SSE_CVTSD2SSrr");
  } else if (Name == "CVTSS2SDrr") {
    handle_memrepl("SSE_CVTSS2SDrm", 1);
    handle_default("SSE_CVTSS2SDrr");
  } else if (Name == "CVTSI2SSrr") {
    handle_memrepl("SSE_CVTSI2SS32rm", 1);
    handle_default("SSE_CVTSI2SS32rr");
  } else if (Name == "CVTSI2SDrr") {
    handle_memrepl("SSE_CVTSI2SD32rm", 1);
    handle_default("SSE_CVTSI2SD32rr");
  } else if (Name == "CVTSI642SSrr") {
    handle_memrepl("SSE_CVTSI2SS64rm", 1);
    handle_default("SSE_CVTSI2SS64rr");
  } else if (Name == "CVTSI642SDrr") {
    handle_memrepl("SSE_CVTSI2SD64rm", 1);
    handle_default("SSE_CVTSI2SD64rr");
  } else if (Name == "CVTTSD2SIrr") {
    handle_memrepl("SSE_CVTTSD2SI32rm", 1);
    handle_default("SSE_CVTTSD2SI32rr");
  } else if (Name == "CVTTSS2SIrr") {
    handle_memrepl("SSE_CVTTSS2SI32rm", 1);
    handle_default("SSE_CVTTSS2SI32rr");
  } else if (Name == "CVTTSD2SI64rr") {
    handle_memrepl("SSE_CVTTSD2SI64rm", 1);
    handle_default("SSE_CVTTSD2SI64rr");
  } else if (Name == "CVTTSS2SI64rr") {
    handle_memrepl("SSE_CVTTSS2SI64rm", 1);
    handle_default("SSE_CVTTSS2SI64rr");
  } else if (Name == "CVTTSD2SI64rr_Int") {
    handle_memrepl("SSE_CVTTSD2SI64rm", 1);
    handle_default("SSE_CVTTSD2SI64rr");
  } else if (Name == "CVTTSS2SI64rr_Int") {
    handle_memrepl("SSE_CVTTSS2SI64rm", 1);
    handle_default("SSE_CVTTSS2SI64rr");
  } else if (Name == "MOV64toPQIrr") {
    handle_default("SSE_MOVQ_G2Xrr");
  } else if (Name == "MOV64toSDrr") {
    handle_default("SSE_MOVQ_G2Xrr");
  } else if (Name == "MOVSS2DIrr") {
    handle_default("SSE_MOVD_X2Grr");
  } else if (Name == "MOVDI2PDIrm") {
    handle_default("SSE_MOVD_G2Xrm", 1);
  } else if (Name == "MOVQI2PQIrm") {
    handle_default("SSE_MOVQ_G2Xrm", 1);
  } else if (Name == "MOVDI2SSrr") {
    handle_default("SSE_MOVD_G2Xrr");
  } else if (Name == "MOVSDto64rr") {
    handle_default("SSE_MOVQ_X2Grr");
  } else if (Name == "PUNPCKLDQrm") {
    handle_default("SSE_PUNPCKLDQrm", 2);
  } else if (Name == "UNPCKHPDrr") {
    handle_default("SSE_UNPCKHPDrr");

  } else if (Name == "INC8r") {
    handle_default("INC8r");
  } else if (Name == "INC16r") {
    handle_default("INC16r");
  } else if (Name == "INC32r") {
    handle_default("INC32r");
  } else if (Name == "INC64r") {
    handle_default("INC64r");
  } else if (Name == "DEC8r") {
    handle_default("DEC8r");
  } else if (Name == "DEC16r") {
    handle_default("DEC16r");
  } else if (Name == "DEC32r") {
    handle_default("DEC32r");
  } else if (Name == "DEC64r") {
    handle_default("DEC64r");
  } else if (Name == "BT16rr") {
    // TODO(ts): need check for imm8
    handle_default("BT16rr");
  } else if (Name == "BT32rr") {
    // TODO(ts): need check for imm8
    handle_default("BT32rr");
  } else if (Name == "BT64rr") {
    // TODO(ts): need check for imm8
    handle_default("BT64rr");
  } else if (Name == "IMUL16rr") {
    // TODO: for imm replacement, use rri encoding
    handle_memrepl("IMUL16rm", 2);
    handle_default("IMUL16rr");
  } else if (Name == "IMUL32rr") {
    // TODO: for imm replacement, use rri encoding
    handle_memrepl("IMUL32rm", 2);
    handle_default("IMUL32rr");
  } else if (Name == "IMUL64rr") {
    // TODO: for imm replacement, use rri encoding
    handle_memrepl("IMUL64rm", 2);
    handle_default("IMUL64rr");
  } else if (Name == "IMUL32rri") {
    handle_memrepl("IMUL32rmi", 1);
    handle_default("IMUL32rri");
  } else if (Name == "IMUL64rri") {
    handle_memrepl("IMUL64rmi", 1);
    handle_default("IMUL64rri");
  } else if (Name == "NEG8r") {
    handle_default("NEG8r");
  } else if (Name == "NEG16r") {
    handle_default("NEG16r");
  } else if (Name == "NEG32r") {
    handle_default("NEG32r");
  } else if (Name == "NEG64r") {
    handle_default("NEG64r");
  } else if (Name == "NOT8r") {
    handle_default("NOT8r");
  } else if (Name == "NOT16r") {
    handle_default("NOT16r");
  } else if (Name == "NOT32r") {
    handle_default("NOT32r");
  } else if (Name == "NOT64r") {
    handle_default("NOT64r");
  } else if (Name == "BSWAP32r") {
    handle_default("BSWAP32r");
  } else if (Name == "BSWAP64r") {
    handle_default("BSWAP64r");
  } else if (Name == "CWD") {
    handle_default("CWD");
  } else if (Name == "CDQ") {
    handle_default("CDQ");
  } else if (Name == "CQO") {
    handle_default("CQO");
  } else if (Name == "LEA64_32r") {
    handle_default("LEA32rm", 1);
  } else if (Name == "LEA64r") {
    handle_default("LEA64rm", 1);

  } else if (Name == "XCHG8rm") {
    handle_xchg_mem("XCHG8mr", 2);
  } else if (Name == "XCHG16rm") {
    handle_xchg_mem("XCHG16mr", 2);
  } else if (Name == "XCHG32rm") {
    handle_xchg_mem("XCHG32mr", 2);
  } else if (Name == "XCHG64rm") {
    handle_xchg_mem("XCHG64mr", 2);
  } else if (Name == "LCMPXCHG8") {
    handle_default("LOCK_CMPXCHG8mr", 0);
  } else if (Name == "LCMPXCHG16") {
    handle_default("LOCK_CMPXCHG16mr", 0);
  } else if (Name == "LCMPXCHG32") {
    handle_default("LOCK_CMPXCHG32mr", 0);
  } else if (Name == "LCMPXCHG64") {
    handle_default("LOCK_CMPXCHG64mr", 0);
  } else if (Name == "LXADD8") {
    handle_xchg_mem("LOCK_XADD8mr", 2);
  } else if (Name == "LXADD16") {
    handle_xchg_mem("LOCK_XADD16mr", 2);
  } else if (Name == "LXADD32") {
    handle_xchg_mem("LOCK_XADD32mr", 2);
  } else if (Name == "LXADD64") {
    handle_xchg_mem("LOCK_XADD64mr", 2);

  } else if (Name == "SETCCr") {
    std::array<std::string_view, 16> cond_codes = {"SETO8r",
                                                   "SETNO8r",
                                                   "SETC8r",
                                                   "SETNC8r",
                                                   "SETZ8r",
                                                   "SETNZ8r",
                                                   "SETBE8r",
                                                   "SETA8r",
                                                   "SETS8r",
                                                   "SETNS8r",
                                                   "SETP8r",
                                                   "SETNP8r",
                                                   "SETL8r",
                                                   "SETGE8r",
                                                   "SETLE8r",
                                                   "SETG8r"};
    handle_default(cond_codes[mi.getOperand(1).getImm()]);
  } else if (Name == "CMOV32rr") {
    std::array<std::string_view, 16> cond_codes = {"CMOVO32rr",
                                                   "CMOVNO32rr",
                                                   "CMOVC32rr",
                                                   "CMOVNC32rr",
                                                   "CMOVZ32rr",
                                                   "CMOVNZ32rr",
                                                   "CMOVBE32rr",
                                                   "CMOVA32rr",
                                                   "CMOVS32rr",
                                                   "CMOVNS32rr",
                                                   "CMOVP32rr",
                                                   "CMOVNP32rr",
                                                   "CMOVL32rr",
                                                   "CMOVGE32rr",
                                                   "CMOVLE32rr",
                                                   "CMOVG32rr"};
    handle_default(cond_codes[mi.getOperand(3).getImm()]);
  } else if (Name == "CMOV64rr") {
    std::array<std::string_view, 16> cond_codes = {"CMOVO64rr",
                                                   "CMOVNO64rr",
                                                   "CMOVC64rr",
                                                   "CMOVNC64rr",
                                                   "CMOVZ64rr",
                                                   "CMOVNZ64rr",
                                                   "CMOVBE64rr",
                                                   "CMOVA64rr",
                                                   "CMOVS64rr",
                                                   "CMOVNS64rr",
                                                   "CMOVP64rr",
                                                   "CMOVNP64rr",
                                                   "CMOVL64rr",
                                                   "CMOVGE64rr",
                                                   "CMOVLE64rr",
                                                   "CMOVG64rr"};
    handle_default(cond_codes[mi.getOperand(3).getImm()]);

  } else if (Name == "PREFETCHNTA") {
    handle_default("PREFETCHNTAm", 0);
  } else if (Name == "PREFETCHT2") {
    handle_default("PREFETCHT2m", 0);
  } else if (Name == "PREFETCHT1") {
    handle_default("PREFETCHT1m", 0);
  } else if (Name == "PREFETCHT0") {
    handle_default("PREFETCHT0m", 0);
  } else if (Name == "TRAP") {
    handle_default("UD2");
  }

  if (candidates.size() == 0) {
    llvm::errs() << "ERROR: unhandled instruction " << Name << "\n";
    assert(false);
    exit(1);
  }
}

} // namespace tpde_encgen::x64
