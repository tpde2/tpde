// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "TestIR.hpp"
#include "tpde/base.hpp"
#include "tpde/x64/CompilerX64.hpp"

namespace tpde::test {
struct TestIRCompilerX64 : x64::CompilerX64<TestIRAdaptor, TestIRCompilerX64> {
  using Base = x64::CompilerX64<TestIRAdaptor, TestIRCompilerX64>;

  using IRValueRef = typename Base::IRValueRef;
  using IRFuncRef = typename Base::IRFuncRef;
  using ValuePartRef = typename Base::ValuePartRef;
  using ScratchReg = typename Base::ScratchReg;
  using AsmReg = typename Base::AsmReg;
  using InstRange = typename Base::InstRange;

  bool no_fixed_assignments;

  explicit TestIRCompilerX64(TestIRAdaptor *adaptor, bool no_fixed_assignments)
      : Base{adaptor}, no_fixed_assignments(no_fixed_assignments) {}

  static bool arg_is_int128(IRValueRef) noexcept { return false; }
  static bool arg_allow_split_reg_stack_passing(IRValueRef) noexcept {
    return false;
  }

  bool cur_func_may_emit_calls() const noexcept {
    return this->ir()->functions[this->adaptor->cur_func].has_call;
  }

  SymRef cur_personality_func() const noexcept { return {}; }

  struct ValueParts {
    static u32 count() noexcept { return 1; }
    static u32 size_bytes(u32) noexcept { return 8; }
    static tpde::RegBank reg_bank(u32) noexcept {
      return x64::PlatformConfig::GP_BANK;
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

  ValuePartRef val_part_ref_special(ValRefSpecial &, u32) noexcept {
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
} // namespace tpde::test
