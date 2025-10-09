// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicsAArch64.h>
#include <llvm/TargetParser/Triple.h>

#include "LLVMAdaptor.hpp"
#include "LLVMCompilerBase.hpp"
#include "encode_template_arm64.hpp"
#include "tpde/arm64/CompilerA64.hpp"
#include "tpde/base.hpp"

namespace tpde_llvm::arm64 {

struct CompilerConfig : tpde::a64::PlatformConfig {
  static constexpr bool DEFAULT_VAR_REF_HANDLING = false;
};

struct LLVMCompilerArm64 : tpde::a64::CompilerA64<LLVMAdaptor,
                                                  LLVMCompilerArm64,
                                                  LLVMCompilerBase,
                                                  CompilerConfig>,
                           tpde_encodegen::EncodeCompiler<LLVMAdaptor,
                                                          LLVMCompilerArm64,
                                                          LLVMCompilerBase,
                                                          CompilerConfig> {
  using Base = tpde::a64::CompilerA64<LLVMAdaptor,
                                      LLVMCompilerArm64,
                                      LLVMCompilerBase,
                                      CompilerConfig>;
  using EncCompiler = EncodeCompiler<LLVMAdaptor,
                                     LLVMCompilerArm64,
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

  std::variant<std::monostate, tpde::a64::CCAssignerAAPCS> cc_assigners;

  static constexpr std::array<AsmReg, 2> LANDING_PAD_RES_REGS = {AsmReg::R0,
                                                                 AsmReg::R1};

  explicit LLVMCompilerArm64(std::unique_ptr<LLVMAdaptor> &&adaptor)
      : Base{adaptor.get()}, adaptor(std::move(adaptor)) {
    static_assert(tpde::Compiler<LLVMCompilerArm64, tpde::a64::PlatformConfig>);
  }

  void reset() noexcept {
    // TODO: move to LLVMCompilerBase
    Base::reset();
    EncCompiler::reset();
  }

  bool arg_is_int128(const IRValueRef value) const noexcept {
    return value->getType()->isIntegerTy(128);
  }

  bool arg_allow_split_reg_stack_passing(IRValueRef value) const noexcept {
    // we allow splitting the value if it is an aggregate but not if it is an
    // i128 or array
    llvm::Type *ty = value->getType();
    return !ty->isIntegerTy(128) && !ty->isArrayTy();
  }

  void finish_func(u32 func_idx) noexcept;

  void load_address_of_var_reference(AsmReg dst,
                                     tpde::AssignmentPartRef ap) noexcept;

  std::optional<CallBuilder>
      create_call_builder(const llvm::CallBase * = nullptr) noexcept;

  void extract_element(ValueRef &vec_vr,
                       unsigned idx,
                       LLVMBasicValType ty,
                       ValuePart &out) noexcept;
  void insert_element(ValueRef &vec_vr,
                      unsigned idx,
                      LLVMBasicValType ty,
                      GenericValuePart el) noexcept;

  bool compile_br(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  bool compile_inline_asm(const llvm::CallBase *) noexcept;
  bool compile_icmp(const llvm::Instruction *, const ValInfo &, u64) noexcept;
  void compile_i32_cmp_zero(AsmReg reg, llvm::CmpInst::Predicate p) noexcept;

  GenericValuePart create_addr_for_alloca(tpde::AssignmentPartRef ap) noexcept;

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

void LLVMCompilerArm64::finish_func(u32 func_idx) noexcept {
  Base::finish_func(func_idx);

  if (llvm::timeTraceProfilerEnabled()) {
    llvm::timeTraceProfilerEnd(time_entry);
    time_entry = nullptr;
  }
}

void LLVMCompilerArm64::load_address_of_var_reference(
    AsmReg dst, tpde::AssignmentPartRef ap) noexcept {
  auto *global = this->adaptor->global_list[ap.variable_ref_data()];
  const auto sym = global_sym(global);
  assert(sym.valid());
  if (global->isThreadLocal()) {
    // See LLVMCompilerX64 for a discussion on not supporting this case.
    TPDE_FATAL("thread-local variable access without intrinsic");
  }
  // These pairs must be contiguous, avoid possible veneers in between.
  this->text_writer.ensure_space(8);
  if (!use_local_access(global)) {
    // mov the ptr from the GOT
    reloc_text(sym, R_AARCH64_ADR_GOT_PAGE, this->text_writer.offset());
    ASMNC(ADRP, dst, 0, 0);
    reloc_text(sym, R_AARCH64_LD64_GOT_LO12_NC, this->text_writer.offset());
    ASMNC(LDRxu, dst, dst, 0);
  } else {
    // emit lea with relocation
    reloc_text(sym, R_AARCH64_ADR_PREL_PG_HI21, this->text_writer.offset());
    ASMNC(ADRP, dst, 0, 0);
    reloc_text(sym, R_AARCH64_ADD_ABS_LO12_NC, this->text_writer.offset());
    ASMNC(ADDxi, dst, dst, 0);
  }
}

std::optional<LLVMCompilerArm64::CallBuilder>
    LLVMCompilerArm64::create_call_builder(const llvm::CallBase *cb) noexcept {
  llvm::CallingConv::ID cc = llvm::CallingConv::C;
  if (cb) {
    cc = cb->getCallingConv();
  }
  switch (cc) {
  case llvm::CallingConv::C:
  case llvm::CallingConv::Fast:
    // On AArch6464, fastcc behaves like the C calling convention.
    cc_assigners = tpde::a64::CCAssignerAAPCS();
    return CallBuilder{*this,
                       std::get<tpde::a64::CCAssignerAAPCS>(cc_assigners)};
  default: return std::nullopt;
  }
}

void LLVMCompilerArm64::extract_element(ValueRef &vec_vr,
                                        unsigned idx,
                                        LLVMBasicValType ty,
                                        ValuePart &out) noexcept {
  if (!vec_vr.has_assignment()) {
    // Constant handling is target-independent.
    return LLVMCompilerBase::extract_element(vec_vr, idx, ty, out);
  }

  tpde::ValueAssignment *va = vec_vr.assignment();
  u32 elem_sz = this->adaptor->basic_ty_part_size(ty);
  if (elem_sz == va->max_part_size) {
    // We can't handle the scalarized case better than the generic impl.
    return LLVMCompilerBase::extract_element(vec_vr, idx, ty, out);
  }

  assert(va->max_part_size % elem_sz == 0);
  u32 elems_per_part = va->max_part_size / elem_sz;
  // A vector can consist of multiple, equally sized parts.
  u32 part = idx / elems_per_part;
  u32 part_idx = idx % elems_per_part;
  assert(part < va->part_count);

  auto vec_ref = vec_vr.part(part);
  assert(vec_ref.bank() == CompilerConfig::FP_BANK);
  AsmReg vec_reg = vec_ref.load_to_reg();
  // TODO: reuse vec_reg if possible
  AsmReg dst_reg = out.alloc_reg(this);
  switch (ty) {
    using enum LLVMBasicValType;
  case i8: ASM(UMOVwb, dst_reg, vec_reg, part_idx); break;
  case i16: ASM(UMOVwh, dst_reg, vec_reg, part_idx); break;
  case i32: ASM(UMOVws, dst_reg, vec_reg, part_idx); break;
  case i64:
  case ptr: ASM(UMOVxd, dst_reg, vec_reg, part_idx); break;
  case f32: ASM(DUPs, dst_reg, vec_reg, part_idx); break;
  case f64: ASM(DUPd, dst_reg, vec_reg, part_idx); break;
  default: TPDE_UNREACHABLE("unexpected vector element type");
  }
}

void LLVMCompilerArm64::insert_element(ValueRef &vec_vr,
                                       unsigned idx,
                                       LLVMBasicValType ty,
                                       GenericValuePart el) noexcept {
  tpde::ValueAssignment *va = vec_vr.assignment();
  u32 elem_sz = this->adaptor->basic_ty_part_size(ty);
  if (ty == LLVMBasicValType::i1 || elem_sz == va->max_part_size) {
    // We can't handle the scalarized or i1 case better than the generic impl.
    return LLVMCompilerBase::insert_element(vec_vr, idx, ty, std::move(el));
  }

  assert(va->max_part_size % elem_sz == 0);
  u32 elems_per_part = va->max_part_size / elem_sz;
  // A vector can consist of multiple, equally sized parts.
  u32 part = idx / elems_per_part;
  u32 part_idx = idx % elems_per_part;
  assert(part < va->part_count);

  auto vec_ref = vec_vr.part_unowned(part);
  assert(vec_ref.bank() == CompilerConfig::FP_BANK);
  if (vec_ref.assignment().stack_valid() ||
      vec_ref.assignment().register_valid()) {
    vec_ref.load_to_reg();
  } else {
    vec_ref.alloc_reg(); // Uninitialized
  }
  AsmReg vec_reg = vec_ref.cur_reg();
  AsmReg src_reg = this->gval_as_reg(el);
  switch (ty) {
    using enum LLVMBasicValType;
  case i8: ASM(INSbw, vec_reg, part_idx, src_reg); break;
  case i16: ASM(INShw, vec_reg, part_idx, src_reg); break;
  case i32: ASM(INSsw, vec_reg, part_idx, src_reg); break;
  case i64:
  case ptr: ASM(INSdx, vec_reg, part_idx, src_reg); break;
  case f32: ASM(INSs, vec_reg, part_idx, src_reg, 0); break;
  case f64: ASM(INSd, vec_reg, part_idx, src_reg, 0); break;
  default: TPDE_UNREACHABLE("unexpected vector element type");
  }

  vec_ref.set_modified();
}

bool LLVMCompilerArm64::compile_br(const llvm::Instruction *inst,
                                   const ValInfo &,
                                   u64) noexcept {
  const auto *br = llvm::cast<llvm::BranchInst>(inst);
  if (br->isUnconditional()) {
    generate_uncond_branch(adaptor->block_lookup_idx(br->getSuccessor(0)));
    return true;
  }

  const auto true_block = adaptor->block_lookup_idx(br->getSuccessor(0));
  const auto false_block = adaptor->block_lookup_idx(br->getSuccessor(1));

  // TODO: use Tbz/Tbnz. Must retain register until branch.
  {
    auto [_, cond_ref] = this->val_ref_single(br->getCondition());
    const auto cond_reg = cond_ref.load_to_reg();
    ASM(TSTwi, cond_reg, 1);
  }

  generate_cond_branch(Jump::Jne, true_block, false_block);

  return true;
}

bool LLVMCompilerArm64::compile_inline_asm(
    const llvm::CallBase *call) noexcept {
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
      if (code != "{memory}") {
        return false;
      }
    }
  }

  return true;
}

bool LLVMCompilerArm64::compile_icmp(const llvm::Instruction *inst,
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

  Jump::Kind jump;
  bool is_signed = false;
  switch (cmp->getPredicate()) {
    using enum llvm::CmpInst::Predicate;
  case ICMP_EQ: jump = Jump::Jeq; break;
  case ICMP_NE: jump = Jump::Jne; break;
  case ICMP_UGT: jump = Jump::Jhi; break;
  case ICMP_UGE: jump = Jump::Jhs; break;
  case ICMP_ULT: jump = Jump::Jlo; break;
  case ICMP_ULE: jump = Jump::Jls; break;
  case ICMP_SGT:
    jump = Jump::Jgt;
    is_signed = true;
    break;
  case ICMP_SGE:
    jump = Jump::Jge;
    is_signed = true;
    break;
  case ICMP_SLT:
    jump = Jump::Jlt;
    is_signed = true;
    break;
  case ICMP_SLE:
    jump = Jump::Jle;
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
    auto lhs_lo = lhs.part(0);
    auto lhs_hi = lhs.part(1);
    auto rhs_lo = rhs.part(0);
    auto rhs_hi = rhs.part(1);
    auto lhs_reg_lo = lhs_lo.load_to_reg();
    auto lhs_reg_hi = lhs_hi.load_to_reg();
    auto rhs_reg_lo = rhs_lo.load_to_reg();
    auto rhs_reg_hi = rhs_hi.load_to_reg();
    if ((jump == Jump::Jeq) || (jump == Jump::Jne)) {
      // Use CCMP for equality
      ASM(CMPx, lhs_reg_lo, rhs_reg_lo);
      ASM(CCMPx, lhs_reg_hi, rhs_reg_hi, 0, DA_EQ);
    } else {
      // Compare the ints using carried subtraction
      ASM(CMPx, lhs_reg_lo, rhs_reg_lo);
      ASM(SBCSx, DA_ZR, lhs_reg_hi, rhs_reg_hi);
    }
  } else {
    ValuePartRef lhs_op = lhs.part(0);
    ValuePartRef rhs_op = rhs.part(0);

    if (lhs_op.is_const() && !rhs_op.is_const()) {
      std::swap(lhs_op, rhs_op);
      jump = swap_jump(jump).kind;
    }

    if (int_width != 32 && int_width != 64) {
      unsigned ext_bits = tpde::util::align_up(int_width, 32);
      lhs_op = std::move(lhs_op).into_extended(is_signed, int_width, ext_bits);
      rhs_op = std::move(rhs_op).into_extended(is_signed, int_width, ext_bits);
    }

    AsmReg lhs_reg = lhs_op.has_reg() ? lhs_op.cur_reg() : lhs_op.load_to_reg();

    if (rhs_op.is_const()) {
      u64 imm = rhs_op.const_data()[0];
      if (imm == 0 && single_use && fuse_br &&
          (jump == Jump::Jeq || jump == Jump::Jne)) {
        // Generate CBZ/CBNZ if possible. However, lhs_reg might be the register
        // corresponding to a PHI node, which gets modified before the branch.
        // We have to detect this case and generate a copy into a separate
        // register. This case is not easy to detect here, though. Therefore,
        // for now we always copy the value into a register that we own.
        // TODO: copy only when lhs_reg belongs to an overwritten PHI node.
        ScratchReg res_scratch{this};
        if (!lhs_op.can_salvage()) {
          AsmReg src_reg = lhs_reg;
          lhs_reg = res_scratch.alloc_gp();
          this->mov(lhs_reg, src_reg, int_width <= 32 ? 4 : 8);
        } else {
          res_scratch.alloc_specific(lhs_op.salvage());
        }
        lhs_op.reset();
        rhs_op.reset();
        lhs.reset();
        rhs.reset();

        auto jump_kind = jump == Jump::Jeq ? Jump::Cbz : Jump::Cbnz;
        Jump cbz{jump_kind, lhs_reg, int_width <= 32};
        auto true_block = adaptor->block_lookup_idx(fuse_br->getSuccessor(0));
        auto false_block = adaptor->block_lookup_idx(fuse_br->getSuccessor(1));
        generate_cond_branch(cbz, true_block, false_block);
        this->adaptor->inst_set_fused(fuse_br, true);
        return true;
      }

      AsmReg tmp = permanent_scratch_reg;
      if (int_width <= 32) {
        if (!ASMIF(CMPwi, lhs_reg, imm)) {
          this->materialize_constant(imm, CompilerConfig::GP_BANK, 4, tmp);
          ASM(CMPw, lhs_reg, tmp);
        }
      } else {
        if (!ASMIF(CMPxi, lhs_reg, imm)) {
          this->materialize_constant(imm, CompilerConfig::GP_BANK, 4, tmp);
          ASM(CMPx, lhs_reg, tmp);
        }
      }
    } else {
      AsmReg rhs_reg =
          rhs_op.has_reg() ? rhs_op.cur_reg() : rhs_op.load_to_reg();
      if (int_width <= 32) {
        ASM(CMPw, lhs_reg, rhs_reg);
      } else {
        ASM(CMPx, lhs_reg, rhs_reg);
      }
    }
  }

  // ref-count, otherwise phi assignment will think that value is still used
  lhs.reset();
  rhs.reset();

  if (fuse_br) {
    if (!single_use) {
      (void)result_ref(cmp); // ref-count for branch
      generate_raw_set(jump, result_ref(cmp).part(0).alloc_reg());
    }
    auto true_block = adaptor->block_lookup_idx(fuse_br->getSuccessor(0));
    auto false_block = adaptor->block_lookup_idx(fuse_br->getSuccessor(1));
    generate_cond_branch(jump, true_block, false_block);
    this->adaptor->inst_set_fused(fuse_br, true);
  } else if (fuse_ext) {
    auto [_, res_ref] = this->result_ref_single(fuse_ext);
    if (llvm::isa<llvm::ZExtInst>(fuse_ext)) {
      generate_raw_set(jump, res_ref.alloc_reg());
    } else {
      generate_raw_mask(jump, res_ref.alloc_reg());
    }
    this->adaptor->inst_set_fused(fuse_ext, true);
  } else {
    auto [_, res_ref] = this->result_ref_single(cmp);
    generate_raw_set(jump, res_ref.alloc_reg());
  }

  return true;
}

void LLVMCompilerArm64::compile_i32_cmp_zero(
    AsmReg reg, llvm::CmpInst::Predicate pred) noexcept {
  Da64Cond cond = DA_AL;
  switch (pred) {
  case llvm::CmpInst::ICMP_EQ: cond = DA_EQ; break;
  case llvm::CmpInst::ICMP_NE: cond = DA_NE; break;
  case llvm::CmpInst::ICMP_SGT: cond = DA_GT; break;
  case llvm::CmpInst::ICMP_SGE: cond = DA_GE; break;
  case llvm::CmpInst::ICMP_SLT: cond = DA_LT; break;
  case llvm::CmpInst::ICMP_SLE: cond = DA_LE; break;
  case llvm::CmpInst::ICMP_UGT: cond = DA_HI; break;
  case llvm::CmpInst::ICMP_UGE: cond = DA_HS; break;
  case llvm::CmpInst::ICMP_ULT: cond = DA_LO; break;
  case llvm::CmpInst::ICMP_ULE: cond = DA_LS; break;
  default: TPDE_UNREACHABLE("invalid icmp_zero predicate");
  }
  ASM(CMPwi, reg, 0);
  ASM(CSETw, reg, cond);
}

LLVMCompilerArm64::GenericValuePart LLVMCompilerArm64::create_addr_for_alloca(
    tpde::AssignmentPartRef ap) noexcept {
  return GenericValuePart::Expr{AsmReg::R29, ap.variable_stack_off()};
}

void LLVMCompilerArm64::create_helper_call(std::span<IRValueRef> args,
                                           ValueRef *result,
                                           SymRef sym) noexcept {
  tpde::util::SmallVector<CallArg, 8> arg_vec{};
  for (auto arg : args) {
    arg_vec.push_back(CallArg{arg});
  }

  generate_call(sym, arg_vec, result);
}

bool LLVMCompilerArm64::handle_intrin(
    const llvm::IntrinsicInst *inst) noexcept {
  const auto intrin_id = inst->getIntrinsicID();
  switch (intrin_id) {
  case llvm::Intrinsic::vastart: {
    auto [_, list_ref] = this->val_ref_single(inst->getOperand(0));
    auto list_reg = list_ref.load_to_reg();
    auto tmp_reg = permanent_scratch_reg;

    // next stack param
    ASM(LDRxu, tmp_reg, DA_GP(29), reg_save_frame_off + 192);
    ASM(STRxu, tmp_reg, list_reg, 0);
    // gr_top = end of GP arg reg save area
    ASM(ADDxi, tmp_reg, DA_GP(29), reg_save_frame_off + 64);
    ASM(STRxu, tmp_reg, list_reg, 8);
    // vr_top = end of VR arg reg save area
    if (reg_save_frame_off + 64 + 128 == 0) {
      ASM(STRxu, DA_GP(29), list_reg, 16);
    } else {
      ASM(ADDxi, tmp_reg, DA_GP(29), reg_save_frame_off + 64 + 128);
      ASM(STRxu, tmp_reg, list_reg, 16);
    }

    uint32_t gr_offs = 0 - ((8 - scalar_arg_count) * 8);
    uint32_t vr_offs = 0 - ((8 - vec_arg_count) * 16);
    materialize_constant(gr_offs, CompilerConfig::GP_BANK, 4, tmp_reg);
    ASM(STRwu, tmp_reg, list_reg, 24);
    materialize_constant(vr_offs, CompilerConfig::GP_BANK, 4, tmp_reg);
    ASM(STRwu, tmp_reg, list_reg, 28);
    return true;
  }
  case llvm::Intrinsic::vacopy: {
    auto [dst_vr, dst_ref] = this->val_ref_single(inst->getOperand(0));
    auto [src_vr, src_ref] = this->val_ref_single(inst->getOperand(1));

    ScratchReg scratch1{this}, scratch2{this};
    const auto src_reg = src_ref.load_to_reg();
    const auto dst_reg = dst_ref.load_to_reg();

    const auto tmp_reg1 = scratch1.alloc(CompilerConfig::FP_BANK);
    const auto tmp_reg2 = scratch2.alloc(CompilerConfig::FP_BANK);
    ASM(LDPq, tmp_reg1, tmp_reg2, src_reg, 0);
    ASM(STPq, tmp_reg1, tmp_reg2, dst_reg, 0);
    return true;
  }
  case llvm::Intrinsic::stacksave: {
    auto [_, res_vr] = this->result_ref_single(inst);
    ASM(MOV_SPx, res_vr.alloc_reg(), DA_SP);
    return true;
  }
  case llvm::Intrinsic::stackrestore: {
    auto [val_vr, val_ref] = this->val_ref_single(inst->getOperand(0));
    auto val_reg = val_ref.load_to_reg();
    ASM(MOV_SPx, DA_SP, val_reg);
    return true;
  }
  case llvm::Intrinsic::returnaddress: {
    auto [_, res_vr] = this->result_ref_single(inst);
    auto op = llvm::cast<llvm::ConstantInt>(inst->getOperand(0));
    if (op->isZero()) {
      ASM(LDRxu, res_vr.alloc_reg(), DA_GP(29), 8);
    } else {
      ASM(MOVZx, res_vr.alloc_reg(), 0);
    }
    return true;
  }
  case llvm::Intrinsic::frameaddress: {
    auto [_, res_vr] = this->result_ref_single(inst);
    auto op = llvm::cast<llvm::ConstantInt>(inst->getOperand(0));
    ASM(MOVx, res_vr.alloc_reg(), op->isZeroValue() ? DA_GP(29) : DA_ZR);
    return true;
  }
  case llvm::Intrinsic::aarch64_crc32cx: {
    auto [lhs_vr, lhs_ref] = this->val_ref_single(inst->getOperand(0));
    auto [rhs_vr, rhs_ref] = this->val_ref_single(inst->getOperand(1));
    auto [res_vr, res_ref] = this->result_ref_single(inst);

    auto lhs_reg = lhs_ref.load_to_reg();
    auto rhs_reg = rhs_ref.load_to_reg();
    ASM(CRC32CX, res_ref.alloc_try_reuse(lhs_ref), lhs_reg, rhs_reg);
    res_ref.set_modified();
    return true;
  }
  case llvm::Intrinsic::aarch64_neon_umull: {
    auto *vec_ty = llvm::cast<llvm::FixedVectorType>(inst->getType());
    unsigned nelem = vec_ty->getNumElements();

    auto [lhs_vr, lhs_ref] = this->val_ref_single(inst->getOperand(0));
    auto [rhs_vr, rhs_ref] = this->val_ref_single(inst->getOperand(1));
    auto [res_vr, res_ref] = this->result_ref_single(inst);

    auto lhs_reg = lhs_ref.load_to_reg();
    auto rhs_reg = rhs_ref.load_to_reg();
    auto res_reg = res_ref.alloc_try_reuse(lhs_ref);
    switch (nelem) {
    case 2: ASM(UMULL_2d, res_reg, lhs_reg, rhs_reg); break;
    case 4: ASM(UMULL_4s, res_reg, lhs_reg, rhs_reg); break;
    case 8: ASM(UMULL_8h, res_reg, lhs_reg, rhs_reg); break;
    default: TPDE_UNREACHABLE("invalid intrinsic type");
    }
    res_ref.set_modified();
    return true;
  }
  default: return false;
  }
}

bool LLVMCompilerArm64::handle_overflow_intrin_128(
    OverflowOp op,
    GenericValuePart &&lhs_lo,
    GenericValuePart &&lhs_hi,
    GenericValuePart &&rhs_lo,
    GenericValuePart &&rhs_hi,
    ValuePart &&res_lo,
    ValuePart &&res_hi,
    ValuePart &&res_of) noexcept {
  switch (op) {
  case OverflowOp::uadd: {
    AsmReg lhs_lo_reg = gval_as_reg(lhs_lo); // TODO: reuse reg
    AsmReg rhs_lo_reg = gval_as_reg(rhs_lo); // TODO: reuse reg
    AsmReg res_lo_reg = res_lo.alloc_reg(this);
    ASM(ADDSx, res_lo_reg, lhs_lo_reg, rhs_lo_reg);
    AsmReg lhs_hi_reg = gval_as_reg(lhs_hi); // TODO: reuse reg
    AsmReg rhs_hi_reg = gval_as_reg(rhs_hi); // TODO: reuse reg
    AsmReg res_hi_reg = res_hi.alloc_reg(this);
    ASM(ADCSx, res_hi_reg, lhs_hi_reg, rhs_hi_reg);
    AsmReg res_of_reg = res_of.alloc_reg(this);
    ASM(CSETw, res_of_reg, DA_CS);
    return true;
  }
  case OverflowOp::sadd: {
    AsmReg lhs_lo_reg = gval_as_reg(lhs_lo); // TODO: reuse reg
    AsmReg rhs_lo_reg = gval_as_reg(rhs_lo); // TODO: reuse reg
    AsmReg res_lo_reg = res_lo.alloc_reg(this);
    ASM(ADDSx, res_lo_reg, lhs_lo_reg, rhs_lo_reg);
    AsmReg lhs_hi_reg = gval_as_reg(lhs_hi); // TODO: reuse reg
    AsmReg rhs_hi_reg = gval_as_reg(rhs_hi); // TODO: reuse reg
    AsmReg res_hi_reg = res_hi.alloc_reg(this);
    ASM(ADCSx, res_hi_reg, lhs_hi_reg, rhs_hi_reg);
    AsmReg res_of_reg = res_of.alloc_reg(this);
    ASM(CSETw, res_of_reg, DA_VS);
    return true;
  }

  case OverflowOp::usub: {
    AsmReg lhs_lo_reg = gval_as_reg(lhs_lo); // TODO: reuse reg
    AsmReg rhs_lo_reg = gval_as_reg(rhs_lo); // TODO: reuse reg
    AsmReg res_lo_reg = res_lo.alloc_reg(this);
    ASM(SUBSx, res_lo_reg, lhs_lo_reg, rhs_lo_reg);
    AsmReg lhs_hi_reg = gval_as_reg(lhs_hi); // TODO: reuse reg
    AsmReg rhs_hi_reg = gval_as_reg(rhs_hi); // TODO: reuse reg
    AsmReg res_hi_reg = res_hi.alloc_reg(this);
    ASM(SBCSx, res_hi_reg, lhs_hi_reg, rhs_hi_reg);
    AsmReg res_of_reg = res_of.alloc_reg(this);
    ASM(CSETw, res_of_reg, DA_CC);
    return true;
  }
  case OverflowOp::ssub: {
    AsmReg lhs_lo_reg = gval_as_reg(lhs_lo); // TODO: reuse reg
    AsmReg rhs_lo_reg = gval_as_reg(rhs_lo); // TODO: reuse reg
    AsmReg res_lo_reg = res_lo.alloc_reg(this);
    ASM(SUBSx, res_lo_reg, lhs_lo_reg, rhs_lo_reg);
    AsmReg lhs_hi_reg = gval_as_reg(lhs_hi); // TODO: reuse reg
    AsmReg rhs_hi_reg = gval_as_reg(rhs_hi); // TODO: reuse reg
    AsmReg res_hi_reg = res_hi.alloc_reg(this);
    ASM(SBCSx, res_hi_reg, lhs_hi_reg, rhs_hi_reg);
    AsmReg res_of_reg = res_of.alloc_reg(this);
    ASM(CSETw, res_of_reg, DA_VS);
    return true;
  }

  case OverflowOp::umul:
  case OverflowOp::smul: {
#if 0
        const auto frame_off = allocate_stack_slot(1);
        ScratchReg scratch{this};
        AsmReg tmp = scratch.alloc_gp();
        if (!ASMIF(ADDxi, tmp, DA_GP(29), frame_off)) {
            materialize_constant(frame_off, 0, 4, tmp);
            ASM(ADDx, tmp, DA_GP(29), tmp);
        }
        // TODO: generate call
        free_stack_slot(frame_off, 1);
#endif
    return false;
  }

  default: TPDE_UNREACHABLE("invalid operation");
  }
}

std::unique_ptr<LLVMCompiler>
    create_compiler(const llvm::Triple &triple) noexcept {
  if (!triple.isOSBinFormatELF()) {
    return nullptr;
  }

  llvm::StringRef dl_str = "e-m:e-p270:32:32-p271:32:32-p272:64:64-"
                           "i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32";
  auto adaptor = std::make_unique<LLVMAdaptor>(llvm::DataLayout(dl_str));
  return std::make_unique<LLVMCompilerArm64>(std::move(adaptor));
}

} // namespace tpde_llvm::arm64
