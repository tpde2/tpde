// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/TargetParser/Triple.h>

#include "LLVMAdaptor.hpp"
#include "LLVMCompilerBase.hpp"
#include "encode_template_x64.hpp"
#include "tpde/base.hpp"
#include "tpde/util/misc.hpp"
#include "tpde/x64/CompilerX64.hpp"
#include "tpde/x64/FunctionWriterX64.hpp"

namespace tpde_llvm::x64 {

struct CompilerConfig : tpde::x64::PlatformConfig {
  static constexpr bool DEFAULT_VAR_REF_HANDLING = false;
};

struct LLVMCompilerX64 : tpde::x64::CompilerX64<LLVMAdaptor,
                                                LLVMCompilerX64,
                                                LLVMCompilerBase,
                                                CompilerConfig>,
                         tpde_encodegen::EncodeCompiler<LLVMAdaptor,
                                                        LLVMCompilerX64,
                                                        LLVMCompilerBase,
                                                        CompilerConfig> {
  using Base = tpde::x64::CompilerX64<LLVMAdaptor,
                                      LLVMCompilerX64,
                                      LLVMCompilerBase,
                                      CompilerConfig>;
  using EncCompiler = EncodeCompiler<LLVMAdaptor,
                                     LLVMCompilerX64,
                                     LLVMCompilerBase,
                                     CompilerConfig>;

  using ScratchReg = typename Base::ScratchReg;
  using ValuePartRef = typename Base::ValuePartRef;
  using ValuePart = typename Base::ValuePart;
  using ValueRef = typename Base::ValueRef;
  using GenericValuePart = typename Base::GenericValuePart;
  using InstRange = typename Base::InstRange;

  using Assembler = typename Base::Assembler;

  using AsmReg = typename Base::AsmReg;

  std::unique_ptr<LLVMAdaptor> adaptor;

  std::variant<std::monostate, tpde::x64::CCAssignerSysV> cc_assigners;

  static constexpr std::array<AsmReg, 2> LANDING_PAD_RES_REGS = {AsmReg::AX,
                                                                 AsmReg::DX};

  explicit LLVMCompilerX64(std::unique_ptr<LLVMAdaptor> &&adaptor)
      : Base{adaptor.get()}, adaptor(std::move(adaptor)) {
    static_assert(tpde::Compiler<LLVMCompilerX64, tpde::x64::PlatformConfig>);
  }

  void reset() noexcept {
    // TODO: move to LLVMCompilerBase
    Base::reset();
    EncCompiler::reset();
  }

  bool arg_is_int128(const IRValueRef value) const noexcept {
    return value->getType()->isIntegerTy(128);
  }

  bool arg_allow_split_reg_stack_passing(
      const IRValueRef val_idx) const noexcept {
    // we allow splitting the value if it is an aggregate but not if it is an
    // i128
    return !arg_is_int128(val_idx);
  }

  void finish_func(u32 func_idx) noexcept;

  void load_address_of_var_reference(AsmReg dst,
                                     tpde::AssignmentPartRef ap) noexcept;

  std::optional<CallBuilder>
      create_call_builder(const llvm::CallBase * = nullptr) noexcept;

  bool compile_br(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  void generate_conditional_branch(Jump jmp,
                                   IRBlockRef true_target,
                                   IRBlockRef false_target) noexcept;
  bool compile_inline_asm(const llvm::CallBase *) noexcept;
  bool compile_icmp(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  void compile_i32_cmp_zero(AsmReg reg, llvm::CmpInst::Predicate p) noexcept;

  GenericValuePart create_addr_for_alloca(tpde::AssignmentPartRef ap) noexcept;

  void switch_emit_cmp(AsmReg cmp_reg,
                       AsmReg tmp_reg,
                       u64 case_value,
                       bool width_is_32) noexcept;
  void switch_emit_cmpeq(tpde::Label case_label,
                         AsmReg cmp_reg,
                         AsmReg tmp_reg,
                         u64 case_value,
                         bool width_is_32) noexcept;
  bool switch_emit_jump_table(tpde::Label default_label,
                              std::span<tpde::Label> labels,
                              AsmReg cmp_reg,
                              AsmReg tmp_reg,
                              u64 low_bound,
                              u64 high_bound,
                              bool width_is_32) noexcept;
  void switch_emit_binary_step(tpde::Label case_label,
                               tpde::Label gt_label,
                               AsmReg cmp_reg,
                               AsmReg tmp_reg,
                               u64 case_value,
                               bool width_is_32) noexcept;

  void create_helper_call(std::span<IRValueRef> args,
                          ValueRef *result,
                          SymRef sym) noexcept;

  bool handle_intrin(const llvm::IntrinsicInst *) noexcept;

  bool handle_overflow_intrin_128(OverflowOp op,
                                  GenericValuePart &&lhs_lo,
                                  GenericValuePart &&lhs_hi,
                                  GenericValuePart &&rhs_lo,
                                  GenericValuePart &&rhs_hi,
                                  ValuePart &&res_lo,
                                  ValuePart &&res_hi,
                                  ValuePart &&res_of) noexcept;
};

void LLVMCompilerX64::finish_func(u32 func_idx) noexcept {
  Base::finish_func(func_idx);

  if (llvm::timeTraceProfilerEnabled()) {
    llvm::timeTraceProfilerEnd(time_entry);
    time_entry = nullptr;
  }
}

void LLVMCompilerX64::load_address_of_var_reference(
    AsmReg dst, tpde::AssignmentPartRef ap) noexcept {
  auto *global = this->adaptor->global_list[ap.variable_ref_data()];
  const auto sym = global_sym(global);
  assert(sym.valid());
  if (global->isThreadLocal()) {
    // LLVM historically allowed TLS globals to be used as pointers and
    // generate TLS access calls when the pointer is used. This caused
    // problems with coroutines, leading to the addition of the intrinsic
    // llvm.threadlocal.address in 2022; deprecation of the original behavior
    // was considered. Clang now only generates the intrinsic, and other
    // front-ends should do that, too.
    //
    // Here, generating a function call would be highly problematic. This
    // method gets called when allocating/locking ValuePartRefs; it is quite
    // likely that some registers are already fixed at this point. Doing a
    // regular function call would require to spill/move all these values,
    // adding fragile code for a somewhat-deprecated feature. Therefore, we
    // only support access to thread-local variables through the intrinsic.
    TPDE_FATAL("thread-local variable access without intrinsic");
  }
  if (!use_local_access(global)) {
    // mov the ptr from the GOT
    ASM(MOV64rm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
    reloc_text(sym, R_X86_64_GOTPCREL, text_writer.offset() - 4, -4);
  } else {
    // emit lea with relocation
    ASM(LEA64rm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
    reloc_text(sym, R_X86_64_PC32, text_writer.offset() - 4, -4);
  }
}

std::optional<LLVMCompilerX64::CallBuilder>
    LLVMCompilerX64::create_call_builder(const llvm::CallBase *cb) noexcept {
  bool var_arg = cb ? cb->getFunctionType()->isVarArg() : false;
  llvm::CallingConv::ID cc = llvm::CallingConv::C;
  if (cb) {
    cc = cb->getCallingConv();
  }
  switch (cc) {
  case llvm::CallingConv::C:
  case llvm::CallingConv::Fast:
    // On x86-64, fastcc behaves like the C calling convention.
    cc_assigners = tpde::x64::CCAssignerSysV(var_arg);
    return CallBuilder{*this,
                       std::get<tpde::x64::CCAssignerSysV>(cc_assigners)};
  default: return std::nullopt;
  }
}

bool LLVMCompilerX64::compile_br(const llvm::Instruction *inst,
                                 const ValInfo &,
                                 u64) noexcept {
  const auto *br = llvm::cast<llvm::BranchInst>(inst);
  if (br->isUnconditional()) {
    auto spilled = this->spill_before_branch();
    this->begin_branch_region();

    generate_branch_to_block(
        Jump::jmp, adaptor->block_lookup_idx(br->getSuccessor(0)), false, true);

    this->end_branch_region();
    release_spilled_regs(spilled);
    return true;
  }

  const auto true_block = adaptor->block_lookup_idx(br->getSuccessor(0));
  const auto false_block = adaptor->block_lookup_idx(br->getSuccessor(1));

  {
    auto [_, cond_ref] = this->val_ref_single(br->getCondition());
    const auto cond_reg = cond_ref.load_to_reg();
    ASM(TEST8ri, cond_reg, 1);
  }

  generate_conditional_branch(Jump::jne, true_block, false_block);

  return true;
}

void LLVMCompilerX64::generate_conditional_branch(
    Jump jmp, IRBlockRef true_target, IRBlockRef false_target) noexcept {
  const auto next_block = this->analyzer.block_ref(this->next_block());

  const auto true_needs_split = this->branch_needs_split(true_target);
  const auto false_needs_split = this->branch_needs_split(false_target);

  const auto spilled = this->spill_before_branch();
  this->begin_branch_region();

  if (next_block == true_target ||
      (next_block != false_target && true_needs_split)) {
    generate_branch_to_block(
        invert_jump(jmp), false_target, false_needs_split, false);
    generate_branch_to_block(Jump::jmp, true_target, false, true);
  } else if (next_block == false_target) {
    generate_branch_to_block(jmp, true_target, true_needs_split, false);
    generate_branch_to_block(Jump::jmp, false_target, false, true);
  } else {
    assert(!true_needs_split);
    this->generate_branch_to_block(jmp, true_target, false, false);
    this->generate_branch_to_block(Jump::jmp, false_target, false, true);
  }

  this->end_branch_region();
  this->release_spilled_regs(spilled);
}

bool LLVMCompilerX64::compile_inline_asm(const llvm::CallBase *call) noexcept {
  auto inline_asm = llvm::cast<llvm::InlineAsm>(call->getCalledOperand());
  // TODO: handle inline assembly that actually does something
  if (!inline_asm->getAsmString().empty() || inline_asm->isAlignStack() ||
      !call->getType()->isVoidTy() || call->arg_size() != 0) {
    return false;
  }

  auto constraints = inline_asm->ParseConstraints();
  for (const llvm::InlineAsm::ConstraintInfo &ci : constraints) {
    if (ci.Type != llvm::InlineAsm::isClobber) {
      continue;
    }
    for (const auto &code : ci.Codes) {
      if (code != "{memory}" && code != "{dirflag}" && code != "{fpsr}" &&
          code != "{flags}") {
        return false;
      }
    }
  }

  return true;
}

bool LLVMCompilerX64::compile_icmp(const llvm::Instruction *inst,
                                   const ValInfo &val_info,
                                   u64) noexcept {
  const auto *cmp = llvm::cast<llvm::ICmpInst>(inst);
  auto *cmp_ty = cmp->getOperand(0)->getType();
  if (cmp_ty->isVectorTy()) {
    return LLVMCompilerBase::compile_icmp_vector(inst, val_info, 0);
  }
  assert(cmp_ty->isIntegerTy() || cmp_ty->isPointerTy());
  u32 int_width = 64;
  if (cmp_ty->isIntegerTy()) {
    int_width = cmp_ty->getIntegerBitWidth();
  }

  Jump jump;
  bool is_signed = false;
  switch (cmp->getPredicate()) {
    using enum llvm::CmpInst::Predicate;
  case ICMP_EQ: jump = Jump::je; break;
  case ICMP_NE: jump = Jump::jne; break;
  case ICMP_UGT: jump = Jump::ja; break;
  case ICMP_UGE: jump = Jump::jae; break;
  case ICMP_ULT: jump = Jump::jb; break;
  case ICMP_ULE: jump = Jump::jbe; break;
  case ICMP_SGT:
    jump = Jump::jg;
    is_signed = true;
    break;
  case ICMP_SGE:
    jump = Jump::jge;
    is_signed = true;
    break;
  case ICMP_SLT:
    jump = Jump::jl;
    is_signed = true;
    break;
  case ICMP_SLE:
    jump = Jump::jle;
    is_signed = true;
    break;
  default: TPDE_UNREACHABLE("invalid icmp predicate");
  }

  const llvm::BranchInst *fuse_br = nullptr;
  const llvm::Instruction *fuse_ext = nullptr;

  bool single_use = cmp->hasNUses(1);
  const llvm::Instruction *next = cmp->getNextNode();
  if (auto *br = llvm::dyn_cast<llvm::BranchInst>(next);
      br && br->isConditional() && br->getCondition() == cmp) {
    fuse_br = br;
  } else if (single_use && *cmp->user_begin() == next) {
    if (llvm::isa<llvm::ZExtInst, llvm::SExtInst>(next) &&
        next->getType()->getIntegerBitWidth() <= 64) {
      fuse_ext = next;
    }
  }

  auto lhs = this->val_ref(cmp->getOperand(0));
  auto rhs = this->val_ref(cmp->getOperand(1));

  if (int_width == 128) {
    // for 128 bit compares, we need to swap the operands sometimes
    if ((jump == Jump::ja) || (jump == Jump::jbe) || (jump == Jump::jle) ||
        (jump == Jump::jg)) {
      std::swap(lhs, rhs);
      jump = swap_jump(jump);
    }

    auto rhs_lo = rhs.part(0);
    auto rhs_hi = rhs.part(1);
    auto rhs_reg_lo = rhs_lo.load_to_reg();
    auto rhs_reg_hi = rhs_hi.load_to_reg();

    // Compare the ints using carried subtraction
    ScratchReg res_scratch{this};
    if ((jump == Jump::je) || (jump == Jump::jne)) {
      // for eq,neq do something a bit quicker
      ScratchReg scratch{this};
      lhs.part(0).reload_into_specific_fixed(this, res_scratch.alloc_gp());
      lhs.part(1).reload_into_specific_fixed(this, scratch.alloc_gp());

      ASM(XOR64rr, res_scratch.cur_reg(), rhs_reg_lo);
      ASM(XOR64rr, scratch.cur_reg(), rhs_reg_hi);
      ASM(OR64rr, res_scratch.cur_reg(), scratch.cur_reg());
    } else {
      auto lhs_lo = lhs.part(0);
      auto lhs_reg_lo = lhs_lo.load_to_reg();
      auto lhs_high_tmp =
          lhs.part(1).reload_into_specific_fixed(this, res_scratch.alloc_gp());

      ASM(CMP64rr, lhs_reg_lo, rhs_reg_lo);
      ASM(SBB64rr, lhs_high_tmp, rhs_reg_hi);
    }
  } else {
    ValuePartRef lhs_op = lhs.part(0);
    ValuePartRef rhs_op = rhs.part(0);

    if (lhs_op.is_const() && !rhs_op.is_const()) {
      std::swap(lhs_op, rhs_op);
      jump = swap_jump(jump);
    }

    if (int_width != 32 && int_width != 64) {
      unsigned ext_bits = tpde::util::align_up(int_width, 32);
      lhs_op = std::move(lhs_op).into_extended(is_signed, int_width, ext_bits);
      rhs_op = std::move(rhs_op).into_extended(is_signed, int_width, ext_bits);
    }

    AsmReg lhs_reg = lhs_op.has_reg() ? lhs_op.cur_reg() : lhs_op.load_to_reg();
    if (int_width <= 32) {
      if (rhs_op.is_const()) {
        if (i32 rhs_val = i32(rhs_op.const_data()[0])) {
          ASM(CMP32ri, lhs_reg, rhs_val);
        } else {
          ASM(TEST32rr, lhs_reg, lhs_reg);
        }
      } else {
        AsmReg rhs_reg =
            rhs_op.has_reg() ? rhs_op.cur_reg() : rhs_op.load_to_reg();
        ASM(CMP32rr, lhs_reg, rhs_reg);
      }
    } else {
      if (rhs_op.is_const() &&
          i32(rhs_op.const_data()[0]) == i64(rhs_op.const_data()[0])) {
        if (i64 rhs_val = rhs_op.const_data()[0]) {
          ASM(CMP64ri, lhs_reg, rhs_val);
        } else {
          ASM(TEST64rr, lhs_reg, lhs_reg);
        }
      } else {
        AsmReg rhs_reg =
            rhs_op.has_reg() ? rhs_op.cur_reg() : rhs_op.load_to_reg();
        ASM(CMP64rr, lhs_reg, rhs_reg);
      }
    }
  }

  // ref-count, otherwise phi assignment will think that value is still used
  lhs.reset();
  rhs.reset();

  if (fuse_br) {
    if (!single_use) {
      (void)result_ref(cmp); // ref-count for branch
      generate_raw_set(
          jump, result_ref(cmp).part(0).alloc_reg(), /*zext=*/false);
    }
    auto true_block = adaptor->block_lookup_idx(fuse_br->getSuccessor(0));
    auto false_block = adaptor->block_lookup_idx(fuse_br->getSuccessor(1));
    generate_conditional_branch(jump, true_block, false_block);
    this->adaptor->inst_set_fused(fuse_br, true);
  } else if (fuse_ext) {
    auto [_, res_ref] = result_ref_single(fuse_ext);
    if (llvm::isa<llvm::ZExtInst>(fuse_ext)) {
      generate_raw_set(jump, res_ref.alloc_reg(), /*zext=*/true);
    } else {
      generate_raw_mask(jump, res_ref.alloc_reg());
    }
    this->adaptor->inst_set_fused(fuse_ext, true);
  } else {
    auto [_, res_ref] = result_ref_single(cmp);
    generate_raw_set(jump, res_ref.alloc_reg(), /*zext=*/false);
  }

  return true;
}

void LLVMCompilerX64::compile_i32_cmp_zero(
    AsmReg reg, llvm::CmpInst::Predicate pred) noexcept {
  ASM(TEST64rr, reg, reg);
  switch (pred) {
  case llvm::CmpInst::ICMP_EQ: ASM(SETZ8r, reg); break;
  case llvm::CmpInst::ICMP_NE: ASM(SETNZ8r, reg); break;
  case llvm::CmpInst::ICMP_SGT: ASM(SETG8r, reg); break;
  case llvm::CmpInst::ICMP_SGE: ASM(SETGE8r, reg); break;
  case llvm::CmpInst::ICMP_SLT: ASM(SETL8r, reg); break;
  case llvm::CmpInst::ICMP_SLE: ASM(SETLE8r, reg); break;
  case llvm::CmpInst::ICMP_UGT: ASM(SETA8r, reg); break;
  case llvm::CmpInst::ICMP_UGE: ASM(SETNC8r, reg); break;
  case llvm::CmpInst::ICMP_ULT: ASM(SETC8r, reg); break;
  case llvm::CmpInst::ICMP_ULE: ASM(SETBE8r, reg); break;
  default: TPDE_UNREACHABLE("invalid icmp_zero predicate");
  }
  ASM(MOVZXr32r8, reg, reg);
}

LLVMCompilerX64::GenericValuePart LLVMCompilerX64::create_addr_for_alloca(
    tpde::AssignmentPartRef ap) noexcept {
  return GenericValuePart::Expr{AsmReg::BP, ap.variable_stack_off()};
}

void LLVMCompilerX64::switch_emit_cmp(const AsmReg cmp_reg,
                                      const AsmReg tmp_reg,
                                      const u64 case_value,
                                      const bool width_is_32) noexcept {
  if (width_is_32) {
    ASM(CMP32ri, cmp_reg, case_value);
  } else {
    if ((i64)((i32)case_value) == (i64)case_value) {
      ASM(CMP64ri, cmp_reg, case_value);
    } else {
      ValuePartRef const_ref{this, case_value, 8, CompilerConfig::GP_BANK};
      ASM(CMP64rr, cmp_reg, const_ref.reload_into_specific_fixed(tmp_reg));
    }
  }
}

void LLVMCompilerX64::switch_emit_cmpeq(const tpde::Label case_label,
                                        const AsmReg cmp_reg,
                                        const AsmReg tmp_reg,
                                        const u64 case_value,
                                        const bool width_is_32) noexcept {
  switch_emit_cmp(cmp_reg, tmp_reg, case_value, width_is_32);
  generate_raw_jump(Jump::je, case_label);
}

bool LLVMCompilerX64::switch_emit_jump_table(tpde::Label default_label,
                                             std::span<tpde::Label> labels,
                                             AsmReg cmp_reg,
                                             AsmReg tmp_reg,
                                             u64 low_bound,
                                             u64 high_bound,
                                             bool width_is_32) noexcept {
  // NB: we must not evict any registers here.
  if (low_bound != 0) {
    switch_emit_cmp(cmp_reg, tmp_reg, low_bound, width_is_32);
    generate_raw_jump(Jump::jb, default_label);
  }
  switch_emit_cmp(cmp_reg, tmp_reg, high_bound, width_is_32);
  generate_raw_jump(Jump::ja, default_label);

  if (width_is_32) {
    // zero-extend cmp_reg since we use the full width
    ASM(MOV32rr, cmp_reg, cmp_reg);
  }

  if (low_bound != 0) {
    if ((i64)((i32)low_bound) == (i64)low_bound) {
      ASM(SUB64ri, cmp_reg, low_bound);
    } else {
      ValuePartRef const_ref{this, &low_bound, 8, CompilerConfig::GP_BANK};
      ASM(SUB64rr, cmp_reg, const_ref.reload_into_specific_fixed(tmp_reg));
    }
  }

  tpde::Label jump_table = text_writer.label_create();
  ASM(LEA64rm, tmp_reg, FE_MEM(FE_IP, 0, FE_NOREG, -1));
  // we reuse the jump offset stuff since the patch procedure is the same
  text_writer.label_ref(jump_table,
                        text_writer.offset() - 4,
                        tpde::LabelFixupKind::X64_JMP_OR_MEM_DISP);
  // load the 4 byte displacement from the jump table
  ASM(MOVSXr64m32, cmp_reg, FE_MEM(tmp_reg, 4, cmp_reg, 0));
  ASM(ADD64rr, tmp_reg, cmp_reg);
  ASM(JMPr, tmp_reg);

  text_writer.align(4);
  text_writer.ensure_space(4 + 4 * labels.size());
  label_place(jump_table);
  const auto table_off = text_writer.offset();
  for (u32 i = 0; i < labels.size(); i++) {
    if (text_writer.label_is_pending(labels[i])) {
      text_writer.label_ref(labels[i],
                            text_writer.offset(),
                            tpde::LabelFixupKind::X64_JUMP_TABLE);
      text_writer.write<u32>(table_off);
    } else {
      const auto label_off = text_writer.label_offset(labels[i]);
      text_writer.write<i32>((i32)label_off - (i32)table_off);
    }
  }
  return true;
}

void LLVMCompilerX64::switch_emit_binary_step(const tpde::Label case_label,
                                              const tpde::Label gt_label,
                                              const AsmReg cmp_reg,
                                              const AsmReg tmp_reg,
                                              const u64 case_value,
                                              const bool width_is_32) noexcept {
  switch_emit_cmpeq(case_label, cmp_reg, tmp_reg, case_value, width_is_32);
  generate_raw_jump(Jump::ja, gt_label);
}

void LLVMCompilerX64::create_helper_call(std::span<IRValueRef> args,
                                         ValueRef *result,
                                         SymRef sym) noexcept {
  tpde::util::SmallVector<CallArg, 8> arg_vec{};
  for (auto arg : args) {
    arg_vec.push_back(CallArg{arg});
  }

  generate_call(sym, arg_vec, result);
}

bool LLVMCompilerX64::handle_intrin(const llvm::IntrinsicInst *inst) noexcept {
  const auto intrin_id = inst->getIntrinsicID();
  switch (intrin_id) {
  case llvm::Intrinsic::vastart: {
    auto [_, list_ref] = this->val_ref_single(inst->getOperand(0));
    ScratchReg scratch1{this};
    auto list_reg = list_ref.load_to_reg();
    auto tmp_reg = scratch1.alloc_gp();

    u64 combined_off = (((static_cast<u64>(vec_arg_count) * 16) + 48) << 32) |
                       (static_cast<u64>(scalar_arg_count) * 8);
    ASM(MOV64ri, tmp_reg, combined_off);
    ASM(MOV64mr, FE_MEM(list_reg, 0, FE_NOREG, 0), tmp_reg);

    assert(-static_cast<i32>(reg_save_frame_off) < 0);
    ASM(LEA64rm, tmp_reg, FE_MEM(FE_BP, 0, FE_NOREG, -(i32)reg_save_frame_off));
    ASM(MOV64mr, FE_MEM(list_reg, 0, FE_NOREG, 16), tmp_reg);

    ASM(LEA64rm, tmp_reg, FE_MEM(FE_BP, 0, FE_NOREG, (i32)var_arg_stack_off));
    ASM(MOV64mr, FE_MEM(list_reg, 0, FE_NOREG, 8), tmp_reg);
    return true;
  }
  case llvm::Intrinsic::vacopy: {
    auto [dst_vr, dst_ref] = this->val_ref_single(inst->getOperand(0));
    auto [src_vr, src_ref] = this->val_ref_single(inst->getOperand(1));

    ScratchReg scratch{this};
    const auto src_reg = src_ref.load_to_reg();
    const auto dst_reg = dst_ref.load_to_reg();

    const auto tmp_reg = scratch.alloc(CompilerConfig::FP_BANK);
    ASM(SSE_MOVDQUrm, tmp_reg, FE_MEM(src_reg, 0, FE_NOREG, 0));
    ASM(SSE_MOVDQUmr, FE_MEM(dst_reg, 0, FE_NOREG, 0), tmp_reg);

    ASM(SSE_MOVQrm, tmp_reg, FE_MEM(src_reg, 0, FE_NOREG, 16));
    ASM(SSE_MOVQmr, FE_MEM(dst_reg, 0, FE_NOREG, 16), tmp_reg);
    return true;
  }
  case llvm::Intrinsic::stacksave: {
    ValuePartRef res{this, CompilerConfig::GP_BANK};
    ASM(MOV64rr, res.alloc_reg(), FE_SP);
    this->result_ref(inst).part(0).set_value(std::move(res));
    return true;
  }
  case llvm::Intrinsic::stackrestore: {
    auto [val_vr, val_ref] = this->val_ref_single(inst->getOperand(0));
    auto val_reg = val_ref.load_to_reg();
    ASM(MOV64rr, FE_SP, val_reg);
    return true;
  }
  case llvm::Intrinsic::x86_sse42_crc32_64_64: {
    auto [rhs_vr, rhs_ref] = this->val_ref_single(inst->getOperand(1));
    auto res = this->val_ref(inst->getOperand(0)).part(0).into_temporary();
    ASM(CRC32_64rr, res.cur_reg(), rhs_ref.load_to_reg());
    this->result_ref(inst).part(0).set_value(std::move(res));
    return true;
  }
  case llvm::Intrinsic::returnaddress: {
    ValuePartRef res{this, CompilerConfig::GP_BANK};
    res.alloc_reg();
    auto op = llvm::cast<llvm::ConstantInt>(inst->getOperand(0));
    if (op->isZeroValue()) {
      ASM(MOV64rm, res.cur_reg(), FE_MEM(FE_BP, 0, FE_NOREG, 8));
    } else {
      ASM(XOR32rr, res.cur_reg(), res.cur_reg());
    }
    this->result_ref(inst).part(0).set_value(std::move(res));
    return true;
  }
  case llvm::Intrinsic::frameaddress: {
    ValuePartRef res{this, CompilerConfig::GP_BANK};
    res.alloc_reg();
    auto op = llvm::cast<llvm::ConstantInt>(inst->getOperand(0));
    if (op->isZeroValue()) {
      ASM(MOV64rr, res.cur_reg(), FE_BP);
    } else {
      ASM(XOR32rr, res.cur_reg(), res.cur_reg());
    }
    this->result_ref(inst).part(0).set_value(std::move(res));
    return true;
  }
  case llvm::Intrinsic::x86_sse2_pause: ASM(PAUSE); return true;
  default: return false;
  }
}

bool LLVMCompilerX64::handle_overflow_intrin_128(OverflowOp op,
                                                 GenericValuePart &&lhs_lo,
                                                 GenericValuePart &&lhs_hi,
                                                 GenericValuePart &&rhs_lo,
                                                 GenericValuePart &&rhs_hi,
                                                 ValuePart &&res_lo,
                                                 ValuePart &&res_hi,
                                                 ValuePart &&res_of) noexcept {
  using EncodeFnTy = bool (LLVMCompilerX64::*)(GenericValuePart &&,
                                               GenericValuePart &&,
                                               GenericValuePart &&,
                                               GenericValuePart &&,
                                               ValuePart &,
                                               ValuePart &,
                                               ValuePart &);
  EncodeFnTy encode_fn = nullptr;
  switch (op) {
  case OverflowOp::uadd:
    encode_fn = &LLVMCompilerX64::encode_of_add_u128;
    break;
  case OverflowOp::sadd:
    encode_fn = &LLVMCompilerX64::encode_of_add_i128;
    break;
  case OverflowOp::usub:
    encode_fn = &LLVMCompilerX64::encode_of_sub_u128;
    break;
  case OverflowOp::ssub:
    encode_fn = &LLVMCompilerX64::encode_of_sub_i128;
    break;
  case OverflowOp::umul:
    encode_fn = &LLVMCompilerX64::encode_of_mul_u128;
    break;
  case OverflowOp::smul:
    encode_fn = &LLVMCompilerX64::encode_of_mul_i128;
    break;
  default: TPDE_UNREACHABLE("invalid operation");
  }

  return (this->*encode_fn)(std::move(lhs_lo),
                            std::move(lhs_hi),
                            std::move(rhs_lo),
                            std::move(rhs_hi),
                            res_lo,
                            res_hi,
                            res_of);
}

std::unique_ptr<LLVMCompiler>
    create_compiler(const llvm::Triple &triple) noexcept {
  if (!triple.isOSBinFormatELF()) {
    return nullptr;
  }

  llvm::StringRef dl_str = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64"
                           "-i128:128-f80:128-n8:16:32:64-S128";
  auto adaptor = std::make_unique<LLVMAdaptor>(llvm::DataLayout(dl_str));
  return std::make_unique<LLVMCompilerX64>(std::move(adaptor));
}

} // namespace tpde_llvm::x64
