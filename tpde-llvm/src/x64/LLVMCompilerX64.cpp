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
#include "tpde/ELF.hpp"
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

  using ScratchReg = typename Base::ScratchReg;
  using ValuePartRef = typename Base::ValuePartRef;
  using ValuePart = typename Base::ValuePart;
  using ValueRef = typename Base::ValueRef;
  using GenericValuePart = typename Base::GenericValuePart;

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
    EncodeCompiler::reset();
  }

  bool arg_allow_split_reg_stack_passing(IRValueRef value) const noexcept {
    // All types except i128 can be split across registers/stack.
    return !value->getType()->isIntegerTy(128);
  }

  void prologue_assign_arg(tpde::CCAssigner *cc_assigner,
                           u32 arg_idx,
                           IRValueRef arg) noexcept {
    if (arg->getType()->isX86_FP80Ty()) [[unlikely]] {
      fp80_assign_arg(cc_assigner, arg);
    } else {
      Base::prologue_assign_arg(cc_assigner, arg_idx, arg);
    }
  }

  void load_address_of_var_reference(AsmReg dst,
                                     tpde::AssignmentPartRef ap) noexcept;

  std::optional<CallBuilder>
      create_call_builder(const llvm::CallBase * = nullptr) noexcept;

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

  // x86_fp80 support.

  /// Get memory operand for spill slot of a value, which must have an
  /// assignment. If write is false, the value is spilled; otherwise, if write
  /// is true, the value is marked as spilled (stack valid).
  FeMem spill_slot_op(ValuePart &val, bool write = false) {
    if (write) {
      allocate_spill_slot(val.assignment());
      val.assignment().set_stack_valid();
    } else {
      spill(val.assignment());
    }
    return FE_MEM(FE_BP, 0, FE_NOREG, val.assignment().frame_off());
  }

  void fp80_assign_arg(tpde::CCAssigner *, IRValueRef arg) noexcept;
  void fp80_push(ValuePart &&value) noexcept;
  void fp80_pop(ValuePart &val) noexcept {
    ASM(FSTPm80, spill_slot_op(val, true));
  }
  void fp80_load(GenericValuePart &&addr, ValuePart &&res) noexcept {
    // TODO: use encodeable_with? need to move that to CompilerX64.
    ASM(FLDm80, FE_MEM(gval_as_reg(addr), 0, FE_NOREG, 0));
    fp80_pop(res);
  }
  void fp80_store(GenericValuePart &&addr, ValuePart &&val) noexcept {
    fp80_push(std::move(val));
    // TODO: use encodeable_with? need to move that to CompilerX64.
    ASM(FSTPm80, FE_MEM(gval_as_reg(addr), 0, FE_NOREG, 0));
  }
  void fp80_ext_float(ValuePart &&src, ValuePart &&dst) noexcept {
    ASM(FLDm32, spill_slot_op(src, false));
    fp80_pop(dst);
  }
  void fp80_ext_double(ValuePart &&src, ValuePart &&dst) noexcept {
    ASM(FLDm64, spill_slot_op(src, false));
    fp80_pop(dst);
  }
  void fp80_trunc_float(ValuePart &&src, ValuePart &&dst) noexcept {
    fp80_push(std::move(src));
    ASM(FSTPm32, spill_slot_op(dst, true));
  }
  void fp80_trunc_double(ValuePart &&src, ValuePart &&dst) noexcept {
    fp80_push(std::move(src));
    ASM(FSTPm64, spill_slot_op(dst, true));
  }
  void fp80_cmp(llvm::CmpInst::Predicate pred,
                ValuePart &&lhs,
                ValuePart &&rhs,
                ValuePart &&res) noexcept;
};

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
    reloc_text(sym, tpde::elf::R_X86_64_GOTPCREL, text_writer.offset() - 4, -4);
  } else {
    // emit lea with relocation
    ASM(LEA64rm, dst, FE_MEM(FE_IP, 0, FE_NOREG, -1));
    reloc_text(sym, tpde::elf::R_X86_64_PC32, text_writer.offset() - 4, -4);
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
    generate_uncond_branch(adaptor->block_lookup_idx(br->getSuccessor(0)));
    return true;
  }

  const auto true_block = adaptor->block_lookup_idx(br->getSuccessor(0));
  const auto false_block = adaptor->block_lookup_idx(br->getSuccessor(1));

  {
    auto [_, cond_ref] = this->val_ref_single(br->getCondition());
    const auto cond_reg = cond_ref.load_to_reg();
    ASM(TEST8ri, cond_reg, 1);
  }

  generate_cond_branch(Jump::jne, true_block, false_block);

  return true;
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

    if (int_width < 8 || (int_width & (int_width - 1))) {
      // We could handle comparisons of integers <32 bit against zero with
      // TESTri. They occur very rarely and are not worth the effort.
      unsigned ext_bits = tpde::util::align_up(int_width, 32);
      lhs_op = std::move(lhs_op).into_extended(is_signed, int_width, ext_bits);
      rhs_op = std::move(rhs_op).into_extended(is_signed, int_width, ext_bits);
      int_width = ext_bits;
    }

    // We can do comparisons against small immediates more efficiently.
    i64 rhs_val = rhs_op.is_const() ? rhs_op.const_data()[0] : 0;
    if (rhs_op.is_const() && (int_width <= 32 || i32(rhs_val) == rhs_val)) {
      // Comparison of 8/16/32/64-bit can use CMPmi. Only do so if the value
      // doesn't reside in a register.
      if (lhs_op.has_assignment()) {
        tpde::AssignmentPartRef ap = lhs_op.assignment();
        if (!ap.register_valid() && ap.stack_valid()) {
          FeMem mem = FE_MEM(FE_BP, 0, FE_NOREG, ap.frame_off());
          switch (int_width) {
          case 8: ASM(CMP8mi, mem, i8(rhs_val)); goto done_compare;
          case 16: ASM(CMP16mi, mem, i16(rhs_val)); goto done_compare;
          case 32: ASM(CMP32mi, mem, i32(rhs_val)); goto done_compare;
          case 64: ASM(CMP64mi, mem, rhs_val); goto done_compare;
          default: TPDE_UNREACHABLE("impossible int bit width");
          }
        }
      }

      auto lhs_reg = lhs_op.has_reg() ? lhs_op.cur_reg() : lhs_op.load_to_reg();
      if (rhs_val == 0) {
        // Comparison of register with zero is TESTrr/TESTri.
        switch (int_width) {
        case 8: ASM(TEST8rr, lhs_reg, lhs_reg); break;
        case 16: ASM(TEST16rr, lhs_reg, lhs_reg); break;
        case 32: ASM(TEST32rr, lhs_reg, lhs_reg); break;
        case 64: ASM(TEST64rr, lhs_reg, lhs_reg); break;
        default: TPDE_UNREACHABLE("impossible int bit width");
        }
      } else {
        // Comparison of 8/16/32/64-bit is CMPri.
        switch (int_width) {
        case 8: ASM(CMP8ri, lhs_reg, i8(rhs_val)); break;
        case 16: ASM(CMP16ri, lhs_reg, i16(rhs_val)); break;
        case 32: ASM(CMP32ri, lhs_reg, i32(rhs_val)); break;
        case 64: ASM(CMP64ri, lhs_reg, rhs_val); break;
        default: TPDE_UNREACHABLE("impossible int bit width");
        }
      }
    } else {
      auto lhs_reg = lhs_op.has_reg() ? lhs_op.cur_reg() : lhs_op.load_to_reg();
      auto rhs_reg = rhs_op.has_reg() ? rhs_op.cur_reg() : rhs_op.load_to_reg();
      switch (int_width) {
      case 8: ASM(CMP8rr, lhs_reg, rhs_reg); break;
      case 16: ASM(CMP16rr, lhs_reg, rhs_reg); break;
      case 32: ASM(CMP32rr, lhs_reg, rhs_reg); break;
      case 64: ASM(CMP64rr, lhs_reg, rhs_reg); break;
      default: TPDE_UNREACHABLE("impossible int bit width");
      }
    }

  done_compare:;
  }

  // No need for set_preserve_flags; we don't call helpers that could
  // potentially clobber them.

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
    generate_cond_branch(jump, true_block, false_block);
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

void LLVMCompilerX64::fp80_assign_arg(tpde::CCAssigner *cc_assigner,
                                      IRValueRef arg) noexcept {
  auto [vr, vpr] = result_ref_single(arg);
  assert(vr.assignment()->part_count == 1);
  tpde::CCAssignment cca{.align = 16, .bank = tpde::RegBank(-2), .size = 16};
  cc_assigner->assign_arg(cca);
  prologue_assign_arg_part(std::move(vpr), cca);
}

void LLVMCompilerX64::fp80_push(ValuePart &&value) noexcept {
  if (value.has_assignment()) {
    spill(value.assignment());
    ASM(FLDm80, FE_MEM(FE_BP, 0, FE_NOREG, value.assignment().frame_off()));
  } else {
    assert(value.is_const());
    std::span<const u64> data = value.const_data();
    assert(data.size() == 2);
    if (data[0] == 0 && data[1] == 0) {
      ASM(FLDZ);
    } else if (data[0] == 0x8000'0000'0000'0000 && data[1] == 0x3fff) {
      ASM(FLD1);
    } else {
      std::span<const u8> raw{reinterpret_cast<const u8 *>(data.data()), 10};
      // TODO: deduplicate/pool constants?
      tpde::SecRef rodata = this->assembler.get_data_section(true, false);
      tpde::SymRef sym = this->assembler.sym_def_data(
          rodata, "", raw, 16, tpde::Assembler::SymBinding::LOCAL);
      ASM(FLDm80, FE_MEM(FE_IP, 0, FE_NOREG, -1));
      this->reloc_text(
          sym, tpde::elf::R_X86_64_PC32, this->text_writer.offset() - 4, -4);
    }
  }
  value.reset(this);
}

void LLVMCompilerX64::fp80_cmp(llvm::CmpInst::Predicate pred,
                               ValuePart &&lhs,
                               ValuePart &&rhs,
                               ValuePart &&res) noexcept {
  using enum llvm::CmpInst::Predicate;
  bool swap = false;
  switch (pred) {
  case FCMP_OLT: swap = true, pred = FCMP_OGT; break;
  case FCMP_UGE: swap = true, pred = FCMP_ULE; break;
  case FCMP_OLE: swap = true, pred = FCMP_OGE; break;
  case FCMP_UGT: swap = true, pred = FCMP_ULT; break;
  default: break;
  }

  fp80_push(std::move(swap ? lhs : rhs));
  fp80_push(std::move(swap ? rhs : lhs));
  ASM(FUCOMIPrr, FE_ST(0), FE_ST(1));
  ASM(FSTPr, FE_ST(0));
  AsmReg dst = res.alloc_reg(this);
  ScratchReg tmp{this};
  switch (pred) {
  case llvm::CmpInst::FCMP_OEQ:
    ASM(SETNP8r, tmp.alloc_gp());
    ASM(SETZ8r, dst);
    ASM(AND8rr, dst, tmp.cur_reg());
    break;
  case llvm::CmpInst::FCMP_UNE:
    ASM(SETP8r, tmp.alloc_gp());
    ASM(SETNZ8r, dst);
    ASM(OR8rr, dst, tmp.cur_reg());
    break;
  case llvm::CmpInst::FCMP_OGT: ASM(SETA8r, dst); break;
  case llvm::CmpInst::FCMP_ULE: ASM(SETBE8r, dst); break;
  case llvm::CmpInst::FCMP_OGE: ASM(SETNC8r, dst); break;
  case llvm::CmpInst::FCMP_ULT: ASM(SETC8r, dst); break;
  case llvm::CmpInst::FCMP_ORD: ASM(SETNP8r, dst); break;
  case llvm::CmpInst::FCMP_UNO: ASM(SETP8r, dst); break;
  case llvm::CmpInst::FCMP_ONE: ASM(SETNZ8r, dst); break;
  case llvm::CmpInst::FCMP_UEQ: ASM(SETZ8r, dst); break;
  default: TPDE_UNREACHABLE("unexpected fcmp predicate");
  }
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
