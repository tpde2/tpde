// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestIRCompiler.hpp"

namespace tpde::test {
using namespace tpde::x64;

bool TestIRCompilerX64::compile_inst(IRInstRef inst_idx, InstRange) noexcept {
  const TestIR::Value &value =
      this->analyzer.adaptor->ir->values[static_cast<u32>(inst_idx)];
  assert(value.type == TestIR::Value::Type::normal ||
         value.type == TestIR::Value::Type::terminator);

  switch (value.op) {
    using enum TestIR::Value::Op;
  case add: return compile_add(inst_idx);
  case sub: return compile_sub(inst_idx);
  case condselect: return compile_condselect(inst_idx);
  case terminate:
  case ret: {
    RetBuilder rb{*derived(), *cur_cc_assigner()};
    if (value.op_count == 1) {
      const auto op = static_cast<IRValueRef>(
          this->adaptor->ir->value_operands[value.op_begin_idx]);

      rb.add(op);
    }
    rb.ret();
    return true;
  }
  case trap:
    ASM(UD2);
    this->release_regs_after_return();
    return true;
  case alloca: return true;
  case zerofill: {
    auto size = ir()->value_operands[value.op_begin_idx];
    this->text_writer.ensure_space(size);
    ASM(JMP, this->text_writer.cur_ptr() + size);
    std::memset(this->text_writer.cur_ptr(), 0, size);
    this->text_writer.cur_ptr() += size;
    return true;
  }
  case br: {
    auto block_idx = ir()->value_operands[value.op_begin_idx];
    this->generate_uncond_branch(IRBlockRef(block_idx));
    return true;
  }
  case condbr:
  case tbz: {
    auto val_idx = IRValueRef(ir()->value_operands[value.op_begin_idx]);
    auto true_block = IRBlockRef(ir()->value_operands[value.op_begin_idx + 1]);
    auto false_block = IRBlockRef(ir()->value_operands[value.op_begin_idx + 2]);

    auto [cond_ref, cond_part] = this->val_ref_single(val_idx);
    auto cond_reg = cond_part.load_to_reg();
    Jump cc = Jump::jne;
    if (value.op == condbr) {
      ASM(TEST64rr, cond_reg, cond_reg);
    } else {
      u32 bit = ir()->value_operands[value.op_begin_idx + 3];
      if (bit <= 32) {
        ASM(TEST32ri, cond_reg, u32{1} << bit);
      } else {
        ASM(BT64ri, cond_reg, bit);
        cc = Jump::jb;
      }
    }
    this->generate_cond_branch(cc, true_block, false_block);
    return true;
  }
  case call: {
    const auto func_idx = value.call_func_idx;
    auto operands = std::span<IRValueRef>{
        reinterpret_cast<IRValueRef *>(ir()->value_operands.data() +
                                       value.op_begin_idx),
        value.op_count};

    auto res_ref = this->result_ref(static_cast<IRValueRef>(inst_idx));

    util::SmallVector<CallArg, 8> arguments{};
    for (auto op : operands) {
      arguments.push_back(CallArg{op});
    }

    this->generate_call(this->func_syms[func_idx], arguments, &res_ref);
    return true;
  }
  default: return false;
  }

  return false;
}

bool TestIRCompilerX64::compile_add(IRInstRef inst_idx) noexcept {
  const TestIR::Value &value = ir()->values[static_cast<u32>(inst_idx)];

  const auto lhs_idx =
      static_cast<IRValueRef>(ir()->value_operands[value.op_begin_idx]);
  const auto rhs_idx =
      static_cast<IRValueRef>(ir()->value_operands[value.op_begin_idx + 1]);

  auto [lhs_vr, lhs] = this->val_ref_single(lhs_idx);
  auto [rhs_vr, rhs] = this->val_ref_single(rhs_idx);
  auto [res_vr, res] =
      this->result_ref_single(static_cast<IRValueRef>(inst_idx));

  AsmReg lhs_reg = lhs.load_to_reg();
  AsmReg rhs_reg = rhs.load_to_reg();
  AsmReg res_reg = res.alloc_try_reuse(lhs);

  if (res_reg == lhs_reg) {
    ASM(ADD64rr, res_reg, rhs_reg);
  } else {
    ASM(LEA64rm, res_reg, FE_MEM(lhs_reg, 1, rhs_reg, 0));
  }
  res.set_modified();
  return true;
}

bool TestIRCompilerX64::compile_sub(IRInstRef inst_idx) noexcept {
  const TestIR::Value &value = ir()->values[static_cast<u32>(inst_idx)];

  const auto lhs_idx =
      static_cast<IRValueRef>(ir()->value_operands[value.op_begin_idx]);
  const auto rhs_idx =
      static_cast<IRValueRef>(ir()->value_operands[value.op_begin_idx + 1]);

  auto lhs = this->val_ref(lhs_idx);
  auto rhs = this->val_ref(rhs_idx);

  ValuePartRef result = lhs.part(0).into_temporary();
  ASM(SUB64rr, result.cur_reg(), rhs.part(0).load_to_reg());
  this->result_ref(static_cast<IRValueRef>(inst_idx))
      .part(0)
      .set_value(std::move(result));
  return true;
}
bool TestIRCompilerX64::compile_condselect(IRInstRef inst_idx) noexcept {
  const TestIR::Value &value = ir()->values[static_cast<u32>(inst_idx)];

  const auto lhs_comp_idx =
      static_cast<IRValueRef>(ir()->value_operands[value.op_begin_idx]);
  const auto rhs_comp_idx =
      static_cast<IRValueRef>(ir()->value_operands[value.op_begin_idx + 1]);
  const auto lhs_idx =
      static_cast<IRValueRef>(ir()->value_operands[value.op_begin_idx + 2]);
  const auto rhs_idx =
      static_cast<IRValueRef>(ir()->value_operands[value.op_begin_idx + 3]);

  {
    auto [lhs_vr, lhs] = this->val_ref_single(lhs_comp_idx);
    auto [rhs_vr, rhs] = this->val_ref_single(rhs_comp_idx);

    ASM(CMP64rr, lhs.load_to_reg(), rhs.load_to_reg());
  }

  auto [lhs_vr, lhs] = this->val_ref_single(lhs_idx);
  auto [rhs_vr, rhs] = this->val_ref_single(rhs_idx);
  auto [res_vr, res] =
      this->result_ref_single(static_cast<IRValueRef>(inst_idx));

  auto res_tmp = std::move(lhs).into_temporary();
  auto rhs_reg = rhs.load_to_reg();

  Jump cc;
  switch (static_cast<TestIR::Value::Cond>(
      ir()->value_operands[value.op_begin_idx + 4])) {
    using enum TestIR::Value::Cond;
  case eq: cc = Jump::je; break;
  case neq: cc = Jump::jne; break;
  case uge: cc = Jump::jae; break;
  case ugt: cc = Jump::ja; break;
  case ule: cc = Jump::jbe; break;
  case ult: cc = Jump::jb; break;
  case sge: cc = Jump::jge; break;
  case sgt: cc = Jump::jg; break;
  case sle: cc = Jump::jle; break;
  case slt: cc = Jump::jl; break;
  default: return false;
  }
  cc = invert_jump(cc);

  generate_raw_cmov(cc, res_tmp.cur_reg(), rhs_reg, true);

  res.set_value(std::move(res_tmp));
  return true;
}
} // namespace tpde::test
