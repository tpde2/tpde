// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fstream>

#include "tpde/arm64/CompilerA64.hpp"

#include "TestIR.hpp"
#include "TestIRCompilerA64.hpp"

namespace {
using namespace tpde;
using namespace tpde::test;

struct TestIRCompilerA64 : a64::CompilerA64<TestIRAdaptor, TestIRCompilerA64> {
  using Base = a64::CompilerA64<TestIRAdaptor, TestIRCompilerA64>;

  using IRValueRef = typename Base::IRValueRef;
  using IRFuncRef = typename Base::IRFuncRef;
  using ValuePartRef = typename Base::ValuePartRef;
  using ScratchReg = typename Base::ScratchReg;
  using AsmReg = typename Base::AsmReg;
  using InstRange = typename Base::InstRange;

  bool no_fixed_assignments;

  explicit TestIRCompilerA64(TestIRAdaptor *adaptor, bool no_fixed_assignments)
      : Base{adaptor}, no_fixed_assignments(no_fixed_assignments) {}

  SymRef cur_personality_func() const noexcept { return {}; }

  static bool arg_is_int128(IRValueRef) noexcept { return false; }
  static bool arg_allow_split_reg_stack_passing(IRValueRef) noexcept {
    return false;
  }

  bool cur_func_may_emit_calls() const noexcept {
    return this->ir()->functions[this->adaptor->cur_func].has_call;
  }

  struct ValueParts {
    static u32 count() noexcept { return 1; }
    static u32 size_bytes(u32) noexcept { return 8; }
    static RegBank reg_bank(u32) noexcept {
      return a64::PlatformConfig::GP_BANK;
    }
  };

  ValueParts val_parts(IRValueRef) { return ValueParts{}; }

  AsmReg select_fixed_assignment_reg(AssignmentPartRef ap,
                                     const IRValueRef value) noexcept {
    if (no_fixed_assignments && !try_force_fixed_assignment(value)) {
      return AsmReg::make_invalid();
    }

    return Base::select_fixed_assignment_reg(ap, value);
  }

  bool try_force_fixed_assignment(const IRValueRef value) const noexcept {
    return ir()->values[static_cast<u32>(value)].force_fixed_assignment;
  }

  std::optional<ValRefSpecial> val_ref_special(IRValueRef) noexcept {
    return {};
  }

  ValuePart val_part_ref_special(ValRefSpecial &, u32) noexcept {
    TPDE_UNREACHABLE("val_part_ref_special on IR without special values");
  }

  void define_func_idx(IRFuncRef func, const u32 idx) noexcept {
    assert(static_cast<u32>(func) == idx);
    (void)func;
    (void)idx;
  }

  [[nodiscard]] bool compile_inst(IRInstRef, InstRange) noexcept;

  TestIR *ir() noexcept { return this->adaptor->ir; }

  const TestIR *ir() const noexcept { return this->adaptor->ir; }

  bool compile_add(IRInstRef) noexcept;
  bool compile_sub(IRInstRef) noexcept;
  bool compile_condselect(IRInstRef) noexcept;
};

bool TestIRCompilerA64::compile_inst(IRInstRef inst_idx, InstRange) noexcept {
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
  case alloca: return true;
  case br: {
    auto block_idx = ir()->value_operands[value.op_begin_idx];
    auto spilled = this->spill_before_branch();

    this->generate_branch_to_block(
        Jump::jmp, static_cast<IRBlockRef>(block_idx), false, true);

    this->release_spilled_regs(spilled);
    return true;
  }
  case zerofill: {
    auto size = ir()->value_operands[value.op_begin_idx];
    this->text_writer.ensure_space(size);
    ASM(B, size / 4);
    this->text_writer.cur_ptr() += (size - 4) & -4u;
    return true;
  }
  case condbr:
  case tbz: {
    auto val_idx =
        static_cast<IRValueRef>(ir()->value_operands[value.op_begin_idx]);
    auto true_block =
        static_cast<IRBlockRef>(ir()->value_operands[value.op_begin_idx + 1]);
    auto false_block =
        static_cast<IRBlockRef>(ir()->value_operands[value.op_begin_idx + 2]);

    auto [_, val] = this->val_ref_single(val_idx);

    auto true_needs_split = this->branch_needs_split(true_block);
    auto false_needs_split = this->branch_needs_split(false_block);

    auto val_reg = val.load_to_reg();

    auto spilled = this->spill_before_branch();

    Jump jump;
    if (value.op == condbr) {
      ASM(CMPxi, val_reg, 0);
      jump = Jump::Jne;
    } else {
      u32 bit = ir()->value_operands[value.op_begin_idx + 3];
      jump = Jump(Jump::Tbz, val_reg, u8(bit));
    }

    Jump inv_jump = invert_jump(jump);

    if (this->analyzer.block_ref(this->next_block()) == true_block) {
      this->generate_branch_to_block(
          inv_jump, false_block, false_needs_split, false);
      this->generate_branch_to_block(Jump::jmp, true_block, false, true);
    } else if (this->analyzer.block_ref(this->next_block()) == false_block) {
      this->generate_branch_to_block(jump, true_block, true_needs_split, false);
      this->generate_branch_to_block(Jump::jmp, false_block, false, true);
    } else if (!true_needs_split) {
      this->generate_branch_to_block(jump, true_block, false, false);
      this->generate_branch_to_block(Jump::jmp, false_block, false, true);
    } else {
      this->generate_branch_to_block(
          inv_jump, false_block, false_needs_split, false);
      this->generate_branch_to_block(Jump::jmp, true_block, false, true);
    }

    this->release_spilled_regs(spilled);
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

bool TestIRCompilerA64::compile_add(IRInstRef inst_idx) noexcept {
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
  ASM(ADDx, res_reg, lhs_reg, rhs_reg);
  res.set_modified();
  return true;
}

bool TestIRCompilerA64::compile_sub(IRInstRef inst_idx) noexcept {
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
  ASM(SUBx, res_reg, lhs_reg, rhs_reg);
  res.set_modified();
  return true;
}
bool TestIRCompilerA64::compile_condselect(IRInstRef inst_idx) noexcept {
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

    ASM(CMPx, lhs.load_to_reg(), rhs.load_to_reg());
  }

  auto [lhs_vr, lhs] = this->val_ref_single(lhs_idx);
  auto [rhs_vr, rhs] = this->val_ref_single(rhs_idx);
  auto [res_vr, res] =
      this->result_ref_single(static_cast<IRValueRef>(inst_idx));

  auto lhs_reg = lhs.load_to_reg();
  auto rhs_reg = rhs.load_to_reg();

  auto res_reg = res.alloc_try_reuse(lhs);

  Jump cc;
  switch (static_cast<TestIR::Value::Cond>(
      ir()->value_operands[value.op_begin_idx + 4])) {
    using enum TestIR::Value::Cond;
  case eq: cc = Jump::Jeq; break;
  case neq: cc = Jump::Jne; break;
  case uge: cc = Jump::Jcs; break;
  case ugt: cc = Jump::Jhi; break;
  case ule: cc = Jump::Jls; break;
  case ult: cc = Jump::Jcc; break;
  case sge: cc = Jump::Jge; break;
  case sgt: cc = Jump::Jgt; break;
  case sle: cc = Jump::Jle; break;
  case slt: cc = Jump::Jlt; break;
  default: return false;
  }

  generate_raw_select(cc, res_reg, lhs_reg, rhs_reg, true);

  res.set_modified();
  return true;
}
} // namespace

bool test::compile_ir_arm64(TestIR *ir,
                            bool no_fixed_assignments,
                            const std::string &obj_out_path) {
  test::TestIRAdaptor adaptor{ir};
  TestIRCompilerA64 compiler{&adaptor, no_fixed_assignments};

  if (!compiler.compile()) {
    TPDE_LOG_ERR("Failed to compile IR");
    return false;
  }

  if (!obj_out_path.empty()) {
    const std::vector<u8> data = compiler.assembler.build_object_file();
    std::ofstream out_file{obj_out_path, std::ios::binary};
    if (!out_file.is_open()) {
      TPDE_LOG_ERR("Failed to open output file");
      return false;
    }
    out_file.write(reinterpret_cast<const char *>(data.data()), data.size());
  }

  return true;
}
