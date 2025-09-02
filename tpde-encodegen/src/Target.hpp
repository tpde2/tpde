// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once
#include <span>
#include <string>
#include <string_view>

#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/Support/raw_ostream.h>

namespace tpde_encgen {

/// Candidate for an encoding of a MachineInstr.
struct MICandidate {
  struct Cond {
    // enum Kind {
    //     NONE,
    //     IMM32S,
    //     IMM8S,
    //     MEM_X86, // can replace register with x86 memory operand, if
    //     taken, operand is expanded to four values (base, scale, index,
    //     off) ADDR_X86, // can replace register with x86 address
    //     (base+scaledidx+off) ADDR_OFF12U0, // can replace register with
    //     addr+offset where offset is 12 bits unsigned, shifted left by 0
    //     bits ADDR_OFF12U1, // can replace register with addr+offset where
    //     offset is 12 bits unsigned, shifted left by 1 bits ADDR_OFF12U2,
    //     // can replace register with addr+offset where offset is 12 bits
    //     unsigned, shifted left by 2 bits ADDR_OFF12U3, // can replace
    //     register with addr+offset where offset is 12 bits unsigned,
    //     shifted left by 3 bits
    // };
    unsigned op_idx;
    std::string func;
    std::string args;

    Cond() : op_idx(0) {}

    Cond(unsigned op_idx, std::string func, std::string args)
        : op_idx(op_idx), func(func), args(args) {}
  };

  using Generator = std::function<void(llvm::raw_ostream &os,
                                       const llvm::MachineInstr &mi,
                                       std::span<const std::string> ops)>;
  llvm::SmallVector<Cond, 2> conds;
  Generator generator;

  MICandidate(Generator generator) : conds(), generator(generator) {}

  MICandidate(unsigned op_idx,
              std::string func,
              std::string args,
              Generator generator)
      : conds({
            Cond{op_idx, func, args}
  }),
        generator(generator) {}

  MICandidate(Cond cond1, Cond cond2, Generator generator)
      : conds({cond1, cond2}), generator(generator) {}

  MICandidate(std::initializer_list<Cond> conds, Generator generator)
      : conds(conds), generator(generator) {}
};

struct EncodingTarget {
  llvm::MachineFunction *func;

  explicit EncodingTarget(llvm::MachineFunction *func) : func(func) {}

  virtual ~EncodingTarget() = default;

  virtual std::string_view get_invalid_reg() = 0;

  // Get all encoding candidates, the candidate without conditions must be
  // last
  virtual void get_inst_candidates(llvm::MachineInstr &,
                                   llvm::SmallVectorImpl<MICandidate> &) {}

  virtual std::optional<std::pair<unsigned, unsigned>>
      is_move(const llvm::MachineInstr &mi) = 0;

  virtual unsigned reg_id_from_mc_reg(llvm::MCRegister) = 0;
  virtual std::string_view reg_name_lower(unsigned id) = 0;
  virtual unsigned reg_bank(unsigned id) = 0;
  // some registers, e.g. flags should be ignored for generation purposes
  virtual bool reg_should_be_ignored(llvm::MCRegister) = 0;
  virtual bool reg_is_flags(llvm::MCRegister) = 0;
  virtual void generate_copy(std::string &buf,
                             unsigned indent,
                             unsigned bank,
                             std::string_view dst,
                             std::string_view src,
                             unsigned size) = 0;
  virtual bool inst_should_be_skipped(llvm::MachineInstr &inst) = 0;
  virtual std::string_view jump_code(unsigned imm) = 0;
};

} // namespace tpde_encgen
