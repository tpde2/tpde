// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "encode_gen.hpp"


#include <format>
#include <iostream>
#include <set>
#include <string>
#include <unordered_set>

#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/AsmPrinter.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineConstantPool.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/SectionKind.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetLoweringObjectFile.h>
#include <llvm/Target/TargetMachine.h>

#include "Target.hpp"
#include "arm64/Target.hpp"
#include "encode_gen.hpp"
#include "x64/Target.hpp"

using namespace tpde_encgen;

namespace {

static unsigned reg_size_bytes(llvm::MachineFunction *func,
                               llvm::Register reg) {
  const auto &target_reg_info = func->getSubtarget().getRegisterInfo();
  const auto &mach_reg_info = func->getRegInfo();
  return target_reg_info->getRegSizeInBits(reg, mach_reg_info) / 8;
}

struct GenerationState {
  llvm::MachineFunction *func;
  EncodingTarget *target;
  std::vector<std::string> param_names{};
  std::set<unsigned> used_regs{};
  llvm::DenseMap<unsigned, std::string> asm_operand_refs{};
  std::unordered_map<std::string, unsigned> operand_ref_counts{};
  /// Mapping from register id to the last instruction defining the return
  /// register *iff the register is unused otherwise*. Empty for functions with
  /// multiple blocks. Used for eliminating useless moves at function end.
  llvm::DenseMap<unsigned, const llvm::MachineInstr *> ret_defs{};
  unsigned cur_inst_id = 0;
  int num_ret_regs = -1;
  std::vector<unsigned> return_regs = {};
  unsigned &sym_count;
  std::unordered_map<unsigned, unsigned> const_pool_indices_used = {};

  // Conditions on which fixed registers depend
  std::vector<std::string> asm_operand_early_conds;
  unsigned num_conds = 0;

  llvm::DenseMap<unsigned, std::vector<std::string>> fixed_reg_conds{};
  // TODO(ts): we could precompute the conditions to get even better
  // knowledge about this
  std::unordered_set<unsigned> maybe_fixed_regs{};

  template <typename... T>
  void fmt_line(std::string &buf,
                unsigned indent,
                std::format_string<T...> fmt,
                T &&...args) {
    std::format_to(std::back_inserter(buf), "{:>{}}", "", indent);
    std::format_to(std::back_inserter(buf), fmt, std::forward<T>(args)...);
    buf += '\n';
  }

  /// Remove asm operand references to the register.
  void kill_reg(llvm::raw_ostream &os, unsigned reg_id);

  bool can_salvage_operand(const llvm::MachineOperand &op);

  void generate_cp_entry_sym(llvm::raw_ostream &os,
                             std::string_view sym_name,
                             unsigned cp_idx);

  void handle_terminator(llvm::raw_ostream &os, llvm::MachineInstr *inst);

  void handle_end_of_block(llvm::raw_ostream &os, llvm::MachineBasicBlock *);
};

bool const_to_bytes(const llvm::DataLayout &dl,
                    const llvm::Constant *constant,
                    std::vector<uint8_t> &bytes,
                    const unsigned off) {
  const auto alloc_size = dl.getTypeAllocSize(constant->getType());
  // can't do this since for floats -0 is special
  // if (constant->isZeroValue()) {
  //    return true;
  //}

  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(constant); CI) {
    // TODO: endianness?
    llvm::StoreIntToMemory(CI->getValue(), bytes.data() + off, alloc_size);
    return true;
  }
  if (auto *CF = llvm::dyn_cast<llvm::ConstantFP>(constant); CF) {
    // TODO: endianness?
    llvm::StoreIntToMemory(
        CF->getValue().bitcastToAPInt(), bytes.data() + off, alloc_size);
    return true;
  }
  if (auto *CDS = llvm::dyn_cast<llvm::ConstantDataSequential>(constant); CDS) {
    auto d = CDS->getRawDataValues();
    assert(d.size() <= alloc_size);
    std::copy(d.bytes_begin(), d.bytes_end(), bytes.begin() + off);
    return true;
  }
  if (auto *CA = llvm::dyn_cast<llvm::ConstantArray>(constant); CA) {
    const auto num_elements = CA->getType()->getNumElements();
    const auto element_size =
        dl.getTypeAllocSize(CA->getType()->getElementType());
    for (auto i = 0u; i < num_elements; ++i) {
      const_to_bytes(
          dl, CA->getAggregateElement(i), bytes, off + i * element_size);
    }
    return true;
  }
  if (auto *CA = llvm::dyn_cast<llvm::ConstantAggregate>(constant); CA) {
    const auto num_elements = CA->getNumOperands();
    auto &ctx = constant->getContext();

    auto *ty = CA->getType();
    auto c0 = llvm::ConstantInt::get(ctx, llvm::APInt(32, 0, false));

    for (auto i = 0u; i < num_elements; ++i) {
      auto idx = llvm::ConstantInt::get(ctx, llvm::APInt(32, i, false));
      auto agg_off = dl.getIndexedOffsetInType(ty, {c0, idx});
      const_to_bytes(dl, CA->getAggregateElement(i), bytes, off + agg_off);
    }
    return true;
  }
  if (auto *ud = llvm::dyn_cast<llvm::UndefValue>(constant); ud) {
    // just leave it at 0
    return true;
  }

  std::cerr << "ERROR: Encountered unsupported constant in constant pool\n";
  return false;
}

void GenerationState::generate_cp_entry_sym(llvm::raw_ostream &os,
                                            std::string_view sym_name,
                                            unsigned cp_idx) {
  auto [it, inserted] = const_pool_indices_used.try_emplace(cp_idx, sym_count);
  if (inserted) {
    sym_count += 1;
  }

  const llvm::MachineConstantPoolEntry &cp_entry =
      func->getConstantPool()->getConstants()[cp_idx];

  if (cp_entry.isMachineConstantPoolEntry()) {
    std::cerr << "ERROR: encountered MachineConstantPoolEntry\n";
    exit(1);
  }

  if (cp_entry.needsRelocation()) {
    std::cerr
        << "ERROR: encountered constant pool entry that needs relocation\n";
    exit(1);
  }

  const auto *constant = cp_entry.Val.ConstVal;
  const auto size = cp_entry.getSizeInBytes(func->getDataLayout());
  // const auto sec_kind =
  // cp_entry.getSectionKind(&func->getDataLayout());
  const auto align = cp_entry.getAlign().value();

  std::vector<uint8_t> bytes{};
  bytes.resize(size);
  if (!const_to_bytes(func->getDataLayout(), constant, bytes, 0)) {
    std::cerr << "ERROR: could not convert constant to bytes\n";
    exit(1);
  }

  os << "    auto &" << sym_name << " = symbols[" << it->second << "];\n";
  os << "    if (!" << sym_name << ".valid()) [[unlikely]] {\n";
  os << "      const std::array<u8, " << bytes.size() << "> data = {{";
  for (uint8_t byte : bytes) {
    os << llvm::format_hex(byte, 2) << ',';
  }
  os << "}};\n";
  os << "      auto sec = derived()->assembler.get_data_section(true);\n";
  os << "      " << sym_name << " = derived()->assembler.sym_def_data(sec, "
     << "\"\", data, " << align << ", Assembler::SymBinding::LOCAL);\n";
  os << "    }\n";
}

bool generate_inst(std::string &buf,
                   GenerationState &state,
                   llvm::MachineInstr *inst) {
  (void)buf;
  (void)state;

  // check if this is a side-effect-free move of an asm operand
  if (auto move_ops = state.target->is_move(*inst)) {
    auto &src_op = inst->getOperand(move_ops->second);
    auto &dst_op = inst->getOperand(move_ops->first);
    auto src_id = state.target->reg_id_from_mc_reg(src_op.getReg());
    auto dst_id = state.target->reg_id_from_mc_reg(dst_op.getReg());
    if (auto it = state.asm_operand_refs.find(src_id);
        it != state.asm_operand_refs.end()) {
      state.operand_ref_counts[it->second]++;
      state.asm_operand_refs[dst_id] = it->second;
      if (src_op.isKill()) {
        llvm::raw_string_ostream os(buf);
        state.kill_reg(os, src_id);
      }
      return true;
    }
    if (src_op.isKill() && state.ret_defs.lookup(dst_id) == inst) {
      state.fmt_line(buf, 4, "// optimized move to returned register");
      state.fmt_line(buf,
                     4,
                     "scratch_{} = std::move(scratch_{});",
                     state.target->reg_name_lower(dst_id),
                     state.target->reg_name_lower(src_id));

      llvm::raw_string_ostream os(buf);
      state.kill_reg(os, dst_id);
      return true;
    }
  }

  if (inst->isPseudo()) {
    const llvm::TargetMachine &TM = inst->getMF()->getTarget();
    const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
    llvm::StringRef Name = MCII.getName(inst->getOpcode());

    if (inst->isKill()) {
      assert(inst->getOperand(0).isReg() && inst->getOperand(1).isReg());
      if (state.target->reg_id_from_mc_reg(inst->getOperand(0).getReg()) !=
          state.target->reg_id_from_mc_reg(inst->getOperand(1).getReg())) {
        std::cerr << "ERROR: Found KILL instruction with different "
                     "source and destination register\n";
        assert(0);
        return false;
      }
      state.fmt_line(buf, 4, "// KILL is a no-op");
      return true;
    } else if (Name == "MEMBARRIER") {
      state.fmt_line(buf, 4, "// MEMBARRIER is a no-op");
      return true;
    } else {
      llvm::errs() << "ERROR: unhandled pseudo: " << *inst << "\n";
      return false;
    }
  }

  if (state.target->inst_should_be_skipped(*inst)) {
    state.fmt_line(buf, 4, "// Skipping");
    return true;
  }

  // we need to check all definitions for existing aliases so
  // that those do not get overwritten since they refer to the old value
  const llvm::MCInstrDesc &llvm_inst_desc =
      state.func->getTarget().getMCInstrInfo()->get(inst->getOpcode());

  llvm::SmallVector<MICandidate> candidates;
  state.target->get_inst_candidates(*inst, candidates);
  assert(candidates.size() > 0);

  if (candidates.size() > 1) {
    state.fmt_line(buf, 4, "do {{");
  }
  std::string parent_conds = "0";
  for (const auto &cand : candidates) {
    std::string if_cond = "";
    llvm::SmallVector<std::pair<unsigned, std::string>, 2> repl_ops{};
    llvm::SmallVector<std::string> use_ops;
    llvm::SmallVector<std::string> ops;

    auto wrote_initial_bracket = false;
    if (!cand.conds.empty()) {
      auto skip = false;
      for (auto &cond_entry : cand.conds) {
        llvm::MachineOperand &cond_op = inst->getOperand(cond_entry.op_idx);
        auto reg_id = state.target->reg_id_from_mc_reg(cond_op.getReg());
        auto value_map_it = state.asm_operand_refs.find(reg_id);
        if (value_map_it == state.asm_operand_refs.end()) {
          skip = true;
          break;
        }
        const auto &op_name = value_map_it->second;

        std::string opt = std::format("{}({}{}{})",
                                      cond_entry.func,
                                      op_name,
                                      cond_entry.args.size() ? ", " : "",
                                      cond_entry.args);
        auto cond = std::format("cond{}", state.num_conds++);
        std::string cond_decl = std::format("auto {} = {};", cond, opt);

        if (!wrote_initial_bracket) {
          state.fmt_line(buf, 4, "{{");
          wrote_initial_bracket = true;
        }

        // Extremely ugly. We want to execute checks which avoid
        // requiring fixed registers at the top. For now pretend this is
        // only relevant when a condition allows replacing an implicit
        // operands. So essentially, this works exactly for shifts on
        // x86 and nothing else.
        // TODO: proper solution? Or explicitly special case x86 shifts?
        if (cond_entry.op_idx >= inst->getNumExplicitOperands()) {
          state.asm_operand_early_conds.push_back(std::move(cond_decl));
          parent_conds += "||" + cond;
        } else {
          state.fmt_line(buf, 4, "{}", cond_decl);
        }
        if (if_cond.empty()) {
          if_cond = cond;
        } else {
          if_cond += std::format(" && {}", cond);
        }
        repl_ops.push_back(std::make_pair(cond_entry.op_idx, cond));
      }
      if (skip) {
        continue;
      }
    } else {
      state.fmt_line(buf, 4, "{{");
    }

    // allocate mapping for destinations
    std::vector<std::pair<bool, const llvm::MachineOperand *>> defs_allocated{};
    for (const auto &def : inst->all_defs()) {
      const auto reg = def.getReg();
      if (state.target->reg_should_be_ignored(reg)) {
        continue;
      }

      defs_allocated.emplace_back(def.isImplicit(), &def);
      state.used_regs.insert(state.target->reg_id_from_mc_reg(reg));
    }

    const auto def_idx = [inst, &state](const llvm::MachineOperand *op) {
      unsigned idx = 0;
      for (const auto &def : inst->all_defs()) {
        assert(def.isReg() && def.getReg().isPhysical());
        if (state.target->reg_should_be_ignored(def.getReg())) {
          continue;
        }
        if (&def == op) {
          return idx;
        }
        ++idx;
      }
      assert(0);
      exit(1);
    };

    const auto inst_id = state.cur_inst_id;

    // Map from register ID their operand to prevent loading the same
    // register twice.
    llvm::DenseMap<unsigned, std::string> allocated_regs;

    unsigned implicit_uses = 0; //, explicit_uses = 0;

    state.fmt_line(buf, 4, "if ({}) {{", !if_cond.empty() ? if_cond : "1");
    unsigned op_cnt = 0;
    for (const llvm::MachineOperand &use : inst->operands()) {
      // llvm::errs() << use.isCPI() << " " << use << "\n";
      unsigned op_idx = op_cnt++;
      if (use.isReg() && use.isUse() && use.isImplicit()) {
        if (implicit_uses >= llvm_inst_desc.NumImplicitUses) {
          continue;
        }
        implicit_uses++;
      }
      if (use.isCPI()) {
        assert(use.getOffset() == 0);
        auto var_name = std::format("op{}_sym", use.getOperandNo());
        use_ops.push_back(var_name);
        llvm::raw_string_ostream os(buf);
        state.generate_cp_entry_sym(os, var_name, use.getIndex());
        continue;
      }

      if (!use.isReg() || !use.isUse()) {
        continue;
      }
      if (auto it = std::find_if(repl_ops.begin(),
                                 repl_ops.end(),
                                 [&](const auto &entry) {
                                   return entry.first == use.getOperandNo();
                                 });
          it != repl_ops.end()) {
        use_ops.push_back(std::format("(*{})", it->second));
        continue;
      }

      if (state.target->reg_should_be_ignored(use.getReg())) {
        if (!use.isImplicit()) {
          use_ops.push_back("");
        }
        continue;
      }

      assert(!use.isTied() ||
             (!use.isImplicit() && "implicit tied use not implemented"));

      auto reg_id = state.target->reg_id_from_mc_reg(use.getReg());
      state.used_regs.insert(reg_id);

      // Don't allocate the same register twice
      auto it = allocated_regs.find(reg_id);
      if (it != allocated_regs.end()) {
        if (!it->second.empty()) {
          use_ops.push_back(it->second);
        }
        continue;
      }

      if (use.isUndef()) {
        if (use.isTied()) {
          state.fmt_line(buf, 8, "// undef tied");
        } else {
          state.fmt_line(buf, 8, "// undef allocate scratch");
          state.fmt_line(buf,
                         8,
                         "AsmReg inst{}_op{} = "
                         "scratch_{}.alloc(RegBank({}));\n",
                         inst_id,
                         op_idx,
                         state.target->reg_name_lower(reg_id),
                         state.target->reg_bank(reg_id));
          use_ops.push_back(std::format("inst{}_op{}", inst_id, op_idx));
          allocated_regs[reg_id] = use_ops.back();
        }
        continue;
      }

      auto asm_op_name = state.asm_operand_refs.lookup(reg_id);

      // TODO(ts): need to think about this
      assert(!use.isEarlyClobber());

      if (use.isImplicit()) {
        // notify the outer code that we need the register
        // argument as a fixed register
        state.fmt_line(
            buf, 8, "// fixing reg {} for cond ({})", reg_id, if_cond);
        state.fixed_reg_conds[reg_id].emplace_back(std::format(
            "!({})&&{}", parent_conds, if_cond.empty() ? "1" : if_cond));

        if (!asm_op_name.empty()) {
          auto dst = std::format("scratch_{}.cur_reg()",
                                 state.target->reg_name_lower(reg_id));
          // TODO(ts): try to salvage the register if possible
          // register should be allocated if we come into here
          std::string copy_src = std::format("op{}_tmp", use.getOperandNo());
          state.fmt_line(buf,
                         8,
                         "AsmReg op{}_tmp = derived()->gval_as_reg({});",
                         use.getOperandNo(),
                         asm_op_name);
          state.target->generate_copy(buf,
                                      8,
                                      state.target->reg_bank(reg_id),
                                      dst,
                                      copy_src,
                                      reg_size_bytes(state.func, use.getReg()));
        }

        continue;
      }

      if (!use.isTied()) {
        auto op_name = std::format("op{}", use.getOperandNo());
        if (asm_op_name.empty()) {
          // TODO(ts): if this reg is also used as a def-reg
          // should we mark it as allocated here?
          state.fmt_line(buf,
                         8,
                         "AsmReg {} = scratch_{}.cur_reg();",
                         op_name,
                         state.target->reg_name_lower(reg_id));
          use_ops.push_back(std::move(op_name));
          allocated_regs[reg_id] = use_ops.back();
          continue;
        }

        auto handled = false;
        if (state.can_salvage_operand(use)) {
          const auto op_bank = state.target->reg_bank(reg_id);
          for (auto &[allocated, def] : defs_allocated) {
            if (allocated) {
              continue;
            }
            const auto def_reg_id =
                state.target->reg_id_from_mc_reg(def->getReg());

            const auto def_bank = state.target->reg_bank(def_reg_id);

            if (def_bank != op_bank) {
              continue;
            }

            const auto def_reg_name = state.target->reg_name_lower(def_reg_id);
            state.fmt_line(buf,
                           8,
                           "AsmReg {} = derived()->gval_as_reg_reuse({}, "
                           "scratch_{});",
                           op_name,
                           asm_op_name,
                           def_reg_name);
            handled = true;
            break;
          }
        }
        if (!handled) {
          state.fmt_line(buf,
                         8,
                         "AsmReg {} = derived()->gval_as_reg({});",
                         op_name,
                         asm_op_name);
        }

        use_ops.push_back(std::move(op_name));
        allocated_regs[reg_id] = use_ops.back();
        continue;
      }

      // Only explicit tied-def are left here, they need no use_ops entry
      const auto &def_reg =
          inst->getOperand(inst->findTiedOperandIdx(use.getOperandNo()));
      defs_allocated[def_idx(&def_reg)].first = true;
      if (!asm_op_name.empty()) {
        // asm operand => allocate register and move source operand there.
        // Try to reuse the asm operand.
        // TODO(ts): if the destination is used as a fixed
        // reg at some point we cannot simply salvage into
        // it i think so we need some extra helpers for that
        if (state.can_salvage_operand(use)) {
          std::format_to(std::back_inserter(buf),
                         "{:>{}}try_salvage_or_materialize({}, "
                         "scratch_{}, {}, {});\n",
                         "",
                         8,
                         asm_op_name,
                         state.target->reg_name_lower(reg_id),
                         state.target->reg_bank(reg_id),
                         reg_size_bytes(state.func, use.getReg()));
        } else {
          std::format_to(std::back_inserter(buf),
                         "{:>{}}AsmReg inst{}_op{} = "
                         "scratch_{}.alloc(RegBank({}));\n",
                         "",
                         8,
                         inst_id,
                         op_idx,
                         state.target->reg_name_lower(reg_id),
                         state.target->reg_bank(reg_id));
          std::format_to(std::back_inserter(buf),
                         "{:>{}}AsmReg inst{}_op{}_tmp = "
                         "derived()->gval_as_reg({});\n",
                         "",
                         8,
                         inst_id,
                         op_idx,
                         asm_op_name);
          state.target->generate_copy(
              buf,
              8,
              state.target->reg_bank(reg_id),
              std::format("inst{}_op{}", inst_id, op_idx),
              std::format("inst{}_op{}_tmp", inst_id, op_idx),
              reg_size_bytes(state.func, use.getReg()));
        }
      }
    }

    // allocate destinations if it did not yet happen
    unsigned imp_defs = 0;
    for (const llvm::MachineOperand &def : inst->all_defs()) {
      assert(def.isReg() && def.getReg().isPhysical());
      const auto reg = def.getReg();

      // Ignore exceeding implicit defs
      if (def.isImplicit() && imp_defs++ >= llvm_inst_desc.NumImplicitDefs) {
        continue;
      }
      if (state.target->reg_should_be_ignored(reg)) {
        if (!def.isImplicit()) {
          ops.push_back("");
        }
        continue;
      }

      const auto reg_id = state.target->reg_id_from_mc_reg(reg);
      if (def.isImplicit()) {
        // notify the outer code that the register is needed as a
        // fixed register
        state.fixed_reg_conds[reg_id].emplace_back(if_cond);
      }

      if (!defs_allocated[def_idx(&def)].first) {
        std::format_to(std::back_inserter(buf),
                       "        // def {} has not been allocated yet\n",
                       state.target->reg_name_lower(reg_id));
        std::format_to(std::back_inserter(buf),
                       "        scratch_{}.alloc(RegBank({}));\n",
                       state.target->reg_name_lower(reg_id),
                       state.target->reg_bank(reg_id));
      }

      if (def.isImplicit()) {
        // implicit defs do not show up in the argument list
        continue;
      }
      ops.push_back(std::format("scratch_{}.cur_reg()",
                                state.target->reg_name_lower(reg_id)));
    }

    ops.append(use_ops);

    llvm::raw_string_ostream os(buf);
    cand.generator(os, *inst, ops);
    if (candidates.size() > 1) {
      state.fmt_line(buf, 8, "break;");
    }
    state.fmt_line(buf, 4, "}}");
    state.fmt_line(buf, 4, "}}");
  }
  if (candidates.size() > 1) {
    state.fmt_line(buf, 4, "}} while (false);");
  }

  // TODO(ts): check killed operands
  for (auto &use : inst->all_uses()) {
    if (!use.isReg() || !use.isKill()) {
      continue;
    }

    const auto reg = use.getReg();
    if (state.target->reg_should_be_ignored(reg)) {
      continue;
    }

    const auto reg_id = state.target->reg_id_from_mc_reg(reg);
    llvm::raw_string_ostream os(buf);
    state.kill_reg(os, reg_id);
  }

  for (auto &def : inst->all_defs()) {
    assert(def.isReg() && def.getReg().isPhysical());
    const auto reg = def.getReg();
    if (state.target->reg_should_be_ignored(reg)) {
      continue;
    }

    // after an instruction all registers are in ScratchRegs
    const auto reg_id = state.target->reg_id_from_mc_reg(reg);
    llvm::raw_string_ostream os(buf);
    state.kill_reg(os, reg_id);
  }

  return true;
}

void GenerationState::kill_reg(llvm::raw_ostream &os, unsigned reg_id) {
  auto asm_op_ref_it = asm_operand_refs.find(reg_id);
  if (asm_op_ref_it == asm_operand_refs.end()) {
    return;
  }
  assert(operand_ref_counts[asm_op_ref_it->second] > 0);
  operand_ref_counts[asm_op_ref_it->second] -= 1;
  if (operand_ref_counts[asm_op_ref_it->second] == 0) {
    os << "    " << asm_op_ref_it->second << ".reset();\n";
  }
  asm_operand_refs.erase(asm_op_ref_it);
}

bool GenerationState::can_salvage_operand(const llvm::MachineOperand &op) {
  if (!op.isReg() || !op.isKill()) {
    return false;
  }

  const auto reg_id = target->reg_id_from_mc_reg(op.getReg());
  auto asm_op_ref_it = asm_operand_refs.find(reg_id);
  assert(asm_op_ref_it != asm_operand_refs.end());
  assert(operand_ref_counts.contains(asm_op_ref_it->second));
  // can overwrite if there are no further aliases
  return operand_ref_counts[asm_op_ref_it->second] <= 1;
}

void GenerationState::handle_end_of_block(llvm::raw_ostream &os,
                                          llvm::MachineBasicBlock *block) {
  if (block->isEntryBlock()) {
    // collect all liveouts with size information <reg, size>
    // ordered map for deterministic output
    std::map<unsigned, unsigned> func_used_regs;
    for (auto &reg : block->liveouts()) {
      if (target->reg_should_be_ignored(reg.PhysReg)) {
        continue;
      }

      const auto reg_id = target->reg_id_from_mc_reg(reg.PhysReg);
      const auto reg_size = reg_size_bytes(func, reg.PhysReg);
      auto [it, inserted] = func_used_regs.try_emplace(reg_id, reg_size);
      if (!inserted && it->second < reg_size) {
        it->second = reg_size;
      }
    }

    // need to collect all registers that may be used in the function
    // and allocate them
    for (auto it = ++block->getIterator(); it != func->end(); ++it) {
      for (const auto &inst : *it) {
        for (const auto &op : inst.operands()) {
          if (!op.isReg() || op.isImplicit()) {
            // implicit defs/uses should be handled through the
            // fixed allocation
            continue;
          }

          if (target->reg_should_be_ignored(op.getReg())) {
            continue;
          }

          const auto reg_id = target->reg_id_from_mc_reg(op.getReg());
          func_used_regs.try_emplace(reg_id, 0);
          used_regs.insert(reg_id);
        }
      }
    }

    // allocate all used registers and resolve aliases if needed
    os << "  // allocate registers used later on in the function\n";
    for (auto [reg, size] : func_used_regs) {
      auto name = target->reg_name_lower(reg);
      auto bank = target->reg_bank(reg);
      auto asm_op_ref_it = asm_operand_refs.find(reg);
      if (asm_op_ref_it == asm_operand_refs.end()) {
        os << "  scratch_" << name << ".alloc(RegBank(" << bank << "));\n";
      } else {
        auto param = asm_op_ref_it->second;
        os << "  try_salvage_or_materialize(" << param << ", scratch_" << name
           << ", " << bank << ", " << size << ");\n";
      }
    }

    asm_operand_refs.clear();
  }
}

void GenerationState::handle_terminator(llvm::raw_ostream &os,
                                        llvm::MachineInstr *inst) {
  if (!inst->isBranch() || inst->isIndirectBranch()) {
    llvm::errs() << "ERROR: unhandled terminator: " << inst << "\n";
    exit(1);
  }

  handle_end_of_block(os, inst->getParent());

  // TODO: this code is inherently target-specific and does not belong here.
  // TODO: reuse generate_inst and just handle block operands specially.
  // Terminators can use registers etc. Delegate jump generation to the target.

  os << "  // Preparing jump to other block\n";
  std::string jump_code;

  if (inst->isUnconditionalBranch()) {
    jump_code = "jmp";
  } else {
    // just assume the first immediate is the condition code
    for (const auto &op : inst->explicit_uses()) {
      if (!op.isImm()) {
        continue;
      }
      assert(op.getImm() >= 0);
      jump_code = target->jump_code(op.getImm());
    }

    if (jump_code.empty()) {
      llvm::errs() << "ERROR: encountered jump without known condition code\n";
      exit(1);
    }
  }
  llvm::MachineBasicBlock *target = nullptr;
  for (const auto &op : inst->explicit_uses()) {
    if (op.isMBB()) {
      if (target) {
        llvm::errs() << "ERROR: multiple block targets for branch\n";
        exit(1);
      }
      target = op.getMBB();
    }
  }
  if (!target) {
    llvm::errs() << "ERROR: could not find block target for branch\n";
    exit(1);
  }

  os << "  derived()->generate_raw_jump(Derived::Jump::" << jump_code
     << ", block" << target->getNumber() << "_label);\n\n";
}

bool encode_prepass(llvm::MachineFunction *func, GenerationState &state) {
  for (auto bb_it = func->begin(); bb_it != func->end(); ++bb_it) {
    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      llvm::MachineInstr &inst = *inst_it;
      if (inst.isDebugInstr() || inst.isCFIInstruction()) {
        continue;
      }

      if (inst.isReturn()) {
        if (state.num_ret_regs == -1) {
          for (const auto &mo : inst.all_uses()) {
            if (!mo.isReg() || mo.isUndef()) { // undef => skip LR on AArch64
              continue;
            }
            auto reg_id = state.target->reg_id_from_mc_reg(mo.getReg());
            state.return_regs.push_back(reg_id);
          }
          state.num_ret_regs = state.return_regs.size();
        }
      }

      if (inst.isTerminator() || inst.isPseudo() ||
          state.target->inst_should_be_skipped(inst)) {
        continue;
      }

      const llvm::MCInstrDesc &MCID =
          state.func->getTarget().getMCInstrInfo()->get(inst.getOpcode());
      for (auto reg : MCID.implicit_defs()) {
        if (state.target->reg_should_be_ignored(reg)) {
          continue;
        }
        state.maybe_fixed_regs.insert(state.target->reg_id_from_mc_reg(reg));
      }
      for (auto reg : MCID.implicit_uses()) {
        if (state.target->reg_should_be_ignored(reg)) {
          continue;
        }
        state.maybe_fixed_regs.insert(state.target->reg_id_from_mc_reg(reg));
      }
    }
  }

  if (func->size() == 1 && func->begin()->back().isReturn()) {
    // For single-block functions, try to identify moves to result registers
    const auto &mbb = *func->begin();
    unsigned missing_defs = state.num_ret_regs;
    for (auto ret_reg : state.return_regs) {
      state.ret_defs[ret_reg] = nullptr;
    }

    // Find last def that is unused
    for (const llvm::MachineInstr &mi : llvm::reverse(mbb)) {
      if (mi.isReturn()) {
        continue;
      }
      for (const auto &mo : mi.operands()) {
        if (!mo.isReg() || !mo.getReg().isValid()) {
          continue;
        }
        auto reg_id = state.target->reg_id_from_mc_reg(mo.getReg());
        auto it = state.ret_defs.find(reg_id);
        if (it == state.ret_defs.end() || it->second != nullptr) {
          continue;
        }
        // Within the same instruction, we visit the defs first, so for an instr
        // that defines and uses the same reg we will only process the def.
        if (mo.isUse()) {
          state.ret_defs.erase(it);
        } else if (mo.isDef()) {
          it->second = &mi;
        }
        missing_defs -= 1;
      }
      if (!missing_defs) {
        break;
      }
    }
  }

  return true;
}

} // namespace

namespace tpde_encgen {
bool create_encode_function(llvm::MachineFunction *func,
                            std::string_view name,
                            std::string &decl_lines,
                            unsigned &sym_count,
                            std::string &impl_lines) {
  std::string write_buf{};

  // update dead/kill flags since they might not be very accurate anymore
  // NOTE: we assume that the MBB liveins are accurate though
  for (auto &MBB : *func) {
    llvm::recomputeLivenessFlags(MBB);
  }

  // write MachineIR as comment at the start of the function
  {
    std::string tmp;
    llvm::raw_string_ostream os(tmp);
    func->print(os);

    auto view = std::string_view{tmp};
    while (!view.empty()) {
      write_buf += "    // ";

      const auto pos = view.find('\n');
      if (pos == std::string_view::npos) {
        write_buf += view;
      } else {
        write_buf += view.substr(0, pos);
        view = view.substr(pos + 1);
      }
      write_buf += '\n';
    }
    write_buf += '\n';
  }

  // TODO(ts): check for the target feature flags of the function
  // and if they are not default, emit a check at the beginning for it

  // TODO(ts): parameterize
  std::unique_ptr<EncodingTarget> target;
  switch (func->getTarget().getTargetTriple().getArch()) {
  case llvm::Triple::x86_64:
    target = std::make_unique<x64::EncodingTargetX64>(func);
    break;
  case llvm::Triple::aarch64:
    target = std::make_unique<arm64::EncodingTargetArm64>(func);
    break;
  default: assert(false && "unsupported target"); abort();
  }
  GenerationState state{
      .func = func, .target = target.get(), .sym_count = sym_count};

  if (!encode_prepass(func, state)) {
    std::cerr << "Prepass failed\n";
    return false;
  }

  auto &mach_reg_info = func->getRegInfo();
  // const auto &target_reg_info = func->getSubtarget().getRegisterInfo();


  // map inputs
  {
    unsigned idx = 0;
    for (auto it = mach_reg_info.livein_begin(),
              end = mach_reg_info.livein_end();
         it != end;
         ++it) {
      auto name = std::format("param_{}", state.param_names.size());
      state.param_names.push_back(std::move(name));

      const auto reg_id = target->reg_id_from_mc_reg(it->first);
      assert(!state.used_regs.contains(reg_id));
      state.used_regs.insert(reg_id);
      state.asm_operand_refs[reg_id] = state.param_names[idx++];
      state.operand_ref_counts[state.param_names.back()] = 1;

      std::format_to(std::back_inserter(write_buf),
                     "    // Mapping {} to {}\n",
                     target->reg_name_lower(reg_id),
                     state.param_names[idx - 1]);
    }
  }

  std::string write_buf_inner{};
  llvm::raw_string_ostream os(write_buf_inner);

  const auto multiple_bbs = func->size() > 1;

  if (multiple_bbs) {
    write_buf += "\n    // Creating label for convergence point at the "
                 "end of the function\n";
    write_buf += "    tpde::Label ret_converge_label = "
                 "derived()->text_writer.label_create();\n";
    write_buf += "    // Creating labels for blocks that are jump targets\n";
  }

  for (auto bb_it = func->begin(); bb_it != func->end(); ++bb_it) {
    if (multiple_bbs && !bb_it->pred_empty()) {
      std::format_to(std::back_inserter(write_buf),
                     "    tpde::Label block{}_label = "
                     "derived()->text_writer.label_create();\n",
                     bb_it->getNumber());

      os << "  // Start of block " << bb_it->getNumber() << "\n";
      os << "  derived()->label_place(block" << bb_it->getNumber()
         << "_label);\n";
    }

    for (auto inst_it = bb_it->begin(); inst_it != bb_it->end(); ++inst_it) {
      llvm::MachineInstr *inst = &(*inst_it);
      if (inst->isDebugInstr() || inst->isCFIInstruction()) {
        continue;
      }


      os << "\n\n    // ";
      inst->print(os, true, false, true, true);
      os << "\n";

      if (inst->isReturn()) {
        if (func->size() > 1 && inst->getParent()->getNextNode()) {
          os << "  derived()->generate_raw_jump(Derived::Jump::jmp, "
                "ret_converge_label);\n";
        }
      } else if (inst->isTerminator()) {
        state.handle_terminator(os, inst);
      } else {
        if (!generate_inst(write_buf_inner, state, inst)) {
          llvm::errs() << "ERROR: failed for instruction: " << *inst << "\n";
          return false;
        }
      }
      ++state.cur_inst_id;
    }
    if (bb_it->empty() || !bb_it->back().isTerminator()) {
      state.handle_end_of_block(os, &*bb_it);
    }
  }

  if (multiple_bbs) {
    // separate block labels from ScratchRegs
    write_buf += '\n';

    os << "  derived()->label_place(ret_converge_label);\n";
  }

  // create ScratchRegs
  for (const auto reg : state.used_regs) {
    std::format_to(std::back_inserter(write_buf),
                   "    ScratchReg scratch_{}{{derived()}};\n",
                   state.target->reg_name_lower(reg));
  }

  // create param conditions
  for (const auto &cond : state.asm_operand_early_conds) {
    std::format_to(std::back_inserter(write_buf), "    {};\n", cond);
  }

  // handle fixed registers
  std::string fixed_reg_param_names{};
  for (const auto &name : state.param_names) {
    if (!fixed_reg_param_names.empty()) {
      fixed_reg_param_names += ", ";
    }
    fixed_reg_param_names += '&';
    fixed_reg_param_names += name;
  }

  for (const auto &[reg, fix_conds] : state.fixed_reg_conds) {
    const auto reg_name_lower = state.target->reg_name_lower(reg);
    auto reg_name_upper = std::string{reg_name_lower};
    std::transform(reg_name_upper.begin(),
                   reg_name_upper.end(),
                   reg_name_upper.begin(),
                   toupper);

    state.fmt_line(write_buf,
                   4,
                   "FixedRegBackup reg_backup_{} = {{.scratch = "
                   "ScratchReg{{derived()}}}};",
                   reg_name_lower);

    auto if_written = false;
    if (!std::any_of(fix_conds.begin(), fix_conds.end(), [](const auto &e) {
          return e.empty();
        })) {
      // no use is unconditional, so we emit an if to not always fix the
      // register
      if_written = true;
      auto if_conds = std::string{};
      for (auto i = 0u; i < fix_conds.size(); ++i) {
        if (std::find(fix_conds.begin(), fix_conds.begin() + i, fix_conds[i]) !=
            fix_conds.begin() + i) {
          continue;
        }
        if (!if_conds.empty()) {
          if_conds += " || ";
        }
        if_conds += '(';
        if_conds += fix_conds[i];
        if_conds += ')';
      }
      write_buf += "    if (";
      write_buf += if_conds;
      write_buf += ") {\n";
    }

    state.fmt_line(write_buf,
                   if_written ? 8 : 4,
                   "scratch_alloc_specific(AsmReg::{}, scratch_{}, {{{}}}, "
                   "reg_backup_{});",
                   reg_name_upper,
                   reg_name_lower,
                   fixed_reg_param_names,
                   reg_name_lower);

    if (if_written) {
      write_buf += "    }\n";
    }

    const auto is_ret_reg =
        std::find(state.return_regs.begin(), state.return_regs.end(), reg) !=
        state.return_regs.end();
    state.fmt_line(write_buf_inner,
                   4,
                   "scratch_check_fixed_backup(scratch_{}, reg_backup_{}, {});",
                   reg_name_lower,
                   reg_name_lower,
                   is_ret_reg ? "true" : "false");
  }

  // assignments for the results need to be after the check for fixed reg
  // backups
  assert(state.num_ret_regs >= 0);
  for (int idx = 0; idx < state.num_ret_regs; ++idx) {
    const auto reg = state.return_regs[idx];
    auto name = state.target->reg_name_lower(reg);

    if (auto it = state.asm_operand_refs.find(reg);
        it != state.asm_operand_refs.end()) {
      os << "  try_salvage_or_materialize(" << it->second << ", scratch_"
         << name << ", " << state.target->reg_bank(reg) << ", "
         << reg_size_bytes(func, reg) << ");\n";
      state.asm_operand_refs.erase(it);
    }

    os << "  result_" << idx << ".set_value(derived(), std::move(scratch_"
       << name << "));\n";
  }

  // TODO
  // for now, always return true
  os << "  return true;\n";

  // finish up function header
  {
    std::string func_args;
    llvm::raw_string_ostream func_args_os(func_args);
    for (auto &param : state.param_names) {
      func_args_os << (func_args.empty() ? "" : ", ") << "GenericValuePart &&"
                   << param;
    }
    // We generate two variants: one with lvalue ref results and one with rvalue
    // ref results. This permits callers to use call the function with rvalues.
    std::string func_args_rvalue = func_args;
    llvm::raw_string_ostream func_args_rvalue_os(func_args_rvalue);
    for (int i = 0; i < state.num_ret_regs; ++i) {
      func_args_os << (func_args.empty() ? "" : ", ") << "ValuePart &result_"
                   << i;
      func_args_rvalue_os << (func_args.empty() ? "" : ", ")
                          << "ValuePart &&result_" << i;
    }

    llvm::raw_string_ostream(impl_lines)
        << "template <typename Adaptor,\n"
           "          typename Derived,\n"
           "          template <typename, typename, typename>\n"
           "          class BaseTy,\n"
           "          typename Config>\n"
           "bool EncodeCompiler<Adaptor, Derived, BaseTy, Config>::encode_"
        << name << "(" << func_args << ") {\n"
        << write_buf << write_buf_inner << "\n}\n\n";
    llvm::raw_string_ostream decl_os(decl_lines);
    decl_os << "  bool encode_" << name << "(" << func_args << ");\n";
    if (state.num_ret_regs > 0) {
      decl_os << "  bool encode_" << name << "(" << func_args_rvalue << ") { ";
      decl_os << " return encode_" << name << "(";
      bool first = true;
      for (auto &param : state.param_names) {
        decl_os << (first ? "" : ", ") << "std::move(" << param << ")";
        first = false;
      }
      for (int i = 0; i < state.num_ret_regs; ++i) {
        decl_os << (first ? "" : ", ") << "result_" << i;
        first = false;
      }
      decl_os << "); }\n";
    }
  }

  return true;
}
} // namespace tpde_encgen
