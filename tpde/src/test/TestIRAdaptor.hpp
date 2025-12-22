// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <ranges>

#include "TestIR.hpp"
#include "tpde/IRAdaptor.hpp"
#include "tpde/ValLocalIdx.hpp"
#include "tpde/base.hpp"

namespace tpde::test {

class IRValueRef {
  u32 val;

public:
  explicit constexpr IRValueRef(u32 val) : val(val) {}
  explicit constexpr operator u32() const { return val; }
  bool operator==(const IRValueRef &other) const { return val == other.val; }
};

class IRInstRef {
  u32 val;

public:
  explicit constexpr IRInstRef(u32 val) : val(val) {}
  explicit constexpr operator u32() const { return val; }
  explicit constexpr operator IRValueRef() const { return IRValueRef(val); }
  bool operator==(const IRInstRef &other) const { return val == other.val; }
};

struct TestIRAdaptor {
  TestIR *ir;

  explicit TestIRAdaptor(TestIR *ir) : ir(ir) {}

  using IRValueRef = tpde::test::IRValueRef;
  using IRInstRef = tpde::test::IRInstRef;
  enum class IRBlockRef : u32 {
  };
  enum class IRFuncRef : u32 {
  };

  static constexpr IRValueRef INVALID_VALUE_REF = static_cast<IRValueRef>(~0u);
  static constexpr IRBlockRef INVALID_BLOCK_REF = static_cast<IRBlockRef>(~0u);
  static constexpr IRFuncRef INVALID_FUNC_REF = static_cast<IRFuncRef>(~0u);

  static constexpr bool TPDE_PROVIDES_HIGHEST_VAL_IDX = true;
  u32 highest_local_val_idx;

  static constexpr bool TPDE_LIVENESS_VISIT_ARGS = true;

  [[nodiscard]] u32 func_count() const {
    return static_cast<u32>(ir->functions.size());
  }

  auto funcs() const {
    return std::views::iota(size_t{0}, ir->functions.size()) |
           std::views::transform([](u32 fn) { return IRFuncRef(fn); });
  }

  [[nodiscard]] auto funcs_to_compile() const { return funcs(); }

  [[nodiscard]] std::string_view func_link_name(const IRFuncRef func) const {
    return ir->functions[static_cast<u32>(func)].name;
  }

  [[nodiscard]] bool func_extern(const IRFuncRef func) const {
    return ir->functions[static_cast<u32>(func)].declaration;
  }

  [[nodiscard]] bool func_only_local(const IRFuncRef func) const {
    return ir->functions[static_cast<u32>(func)].local_only;
  }

  [[nodiscard]] bool func_has_weak_linkage(const IRFuncRef) const {
    return false;
  }

  u32 cur_func;

  [[nodiscard]] bool cur_needs_unwind_info() const { return false; }

  [[nodiscard]] bool cur_is_vararg() const { return false; }

  [[nodiscard]] u32 cur_highest_val_idx() const {
    return highest_local_val_idx;
  }

  [[nodiscard]] auto cur_args() const {
    const auto &func = ir->functions[cur_func];
    return std::views::iota(func.arg_begin_idx, func.arg_end_idx) |
           std::views::transform([](u32 val) { return IRValueRef(val); });
  }

  [[nodiscard]] static bool cur_arg_is_byval(u32) { return false; }

  [[nodiscard]] static u32 cur_arg_byval_align(u32) { return 0; }

  [[nodiscard]] static u32 cur_arg_byval_size(u32) { return 0; }

  [[nodiscard]] static bool cur_arg_is_sret(u32) { return false; }

  [[nodiscard]] auto cur_static_allocas() const {
    assert(ir->functions[cur_func].block_begin_idx !=
           ir->functions[cur_func].block_end_idx);
    const auto &block = ir->blocks[ir->functions[cur_func].block_begin_idx];
    return std::views::iota(block.inst_begin_idx, block.inst_end_idx) |
           std::views::filter([ir = ir](u32 val) {
             return ir->values[val].op == TestIR::Value::Op::alloca;
           }) |
           std::views::transform([](u32 val) { return IRValueRef(val); });
  }

  [[nodiscard]] static bool cur_has_dynamic_alloca() { return false; }

  [[nodiscard]] IRBlockRef cur_entry_block() const {
    const auto &func = ir->functions[cur_func];
    assert(ir->functions[cur_func].block_begin_idx !=
           ir->functions[cur_func].block_end_idx);

    return static_cast<IRBlockRef>(func.block_begin_idx);
  }

  [[nodiscard]] auto cur_blocks() const {
    const auto &func = ir->functions[cur_func];
    return std::views::iota(func.block_begin_idx, func.block_end_idx) |
           std::views::transform([](u32 val) { return IRBlockRef(val); });
  }

  [[nodiscard]] auto block_succs(const IRBlockRef block) const {
    const auto &info = ir->blocks[static_cast<u32>(block)];
    const auto *data = ir->value_operands.data();
    return std::ranges::subrange(data + info.succ_begin_idx,
                                 data + info.succ_end_idx) |
           std::views::transform([](u32 val) { return IRBlockRef(val); });
  }

  [[nodiscard]] auto block_insts(const IRBlockRef block) const {
    const auto &info = ir->blocks[static_cast<u32>(block)];
    return std::views::iota(info.phi_end_idx, info.inst_end_idx) |
           std::views::transform([](u32 val) { return IRInstRef(val); });
  }

  [[nodiscard]] auto block_phis(const IRBlockRef block) const {
    const auto &info = ir->blocks[static_cast<u32>(block)];
    return std::views::iota(info.inst_begin_idx, info.phi_end_idx) |
           std::views::transform([](u32 val) { return IRValueRef(val); });
  }

  [[nodiscard]] u32 block_info(IRBlockRef block) const {
    return ir->blocks[static_cast<u32>(block)].block_info;
  }

  void block_set_info(IRBlockRef block, u32 value) {
    ir->blocks[static_cast<u32>(block)].block_info = value;
  }

  [[nodiscard]] u32 block_info2(IRBlockRef block) const {
    return ir->blocks[static_cast<u32>(block)].block_info2;
  }

  void block_set_info2(IRBlockRef block, u32 value) {
    ir->blocks[static_cast<u32>(block)].block_info2 = value;
  }

  [[nodiscard]] std::string_view block_fmt_ref(IRBlockRef block) const {
    return ir->blocks[static_cast<u32>(block)].name;
  }

  [[nodiscard]] std::string_view value_fmt_ref(IRValueRef val) const {
    return ir->values[static_cast<u32>(val)].name;
  }

  [[nodiscard]] std::string_view inst_fmt_ref(IRInstRef inst) const {
    return ir->values[static_cast<u32>(inst)].name;
  }

  [[nodiscard]] tpde::ValLocalIdx val_local_idx(IRValueRef val) {
    assert(static_cast<u32>(val) >= ir->functions[cur_func].arg_begin_idx);
    return ValLocalIdx(u32(val) - ir->functions[cur_func].arg_begin_idx);
  }

  bool val_is_phi(IRValueRef val) const {
    return ir->values[u32(val)].type == TestIR::Value::Type::phi;
  }

  [[nodiscard]] auto inst_operands(IRInstRef inst) {
    const auto &info = ir->values[static_cast<u32>(inst)];
    const auto *data = ir->value_operands.data();
    return std::ranges::subrange(data + info.op_begin_idx,
                                 data + info.op_begin_idx + info.op_count) |
           std::views::transform([](u32 val) { return IRValueRef(val); });
  }

  [[nodiscard]] bool val_ignore_in_liveness_analysis(IRValueRef value) const {
    return ir->values[static_cast<u32>(value)].op == TestIR::Value::Op::alloca;
  }

  [[nodiscard]] auto inst_results(IRInstRef inst) const {
    const auto &info = ir->values[static_cast<u32>(inst)];
    bool is_def = TestIR::Value::OP_INFOS[static_cast<u32>(info.op)].is_def;
    return std::views::single(IRValueRef(inst)) | std::views::drop(!is_def);
  }

  static bool inst_fused(IRInstRef) { return false; }

  [[nodiscard]] auto val_as_phi(IRValueRef value) const {
    struct PHIRef {
      const u32 *op_begin, *block_begin;

      [[nodiscard]] u32 incoming_count() const {
        return block_begin - op_begin;
      }

      [[nodiscard]] IRValueRef incoming_val_for_slot(const u32 slot) const {
        assert(slot < incoming_count());
        return static_cast<IRValueRef>(op_begin[slot]);
      }

      [[nodiscard]] IRBlockRef incoming_block_for_slot(const u32 slot) const {
        assert(slot < incoming_count());
        return static_cast<IRBlockRef>(block_begin[slot]);
      }

      [[nodiscard]] IRValueRef
          incoming_val_for_block(const IRBlockRef block_ref) const {
        const auto block = static_cast<u32>(block_ref);
        for (auto *op = op_begin; op < block_begin; ++op) {
          if (block_begin[op - op_begin] == block) {
            return static_cast<IRValueRef>(*op);
          }
        }

        return INVALID_VALUE_REF;
      }
    };

    const auto val_idx = static_cast<u32>(value);
    assert(ir->values[val_idx].type == TestIR::Value::Type::phi);
    const auto &info = ir->values[val_idx];
    const auto *data = ir->value_operands.data();
    return PHIRef{data + info.op_begin_idx,
                  data + info.op_begin_idx + info.op_count};
  }

  [[nodiscard]] u32 val_alloca_size(IRValueRef value) const {
    const auto val_idx = static_cast<u32>(value);
    assert(ir->values[val_idx].op == TestIR::Value::Op::alloca);
    const auto *data = ir->value_operands.data();
    return data[ir->values[val_idx].op_begin_idx];
  }

  [[nodiscard]] u32 val_alloca_align(IRValueRef value) const {
    const auto val_idx = static_cast<u32>(value);
    assert(ir->values[val_idx].op == TestIR::Value::Op::alloca);
    const auto *data = ir->value_operands.data();
    return data[ir->values[val_idx].op_begin_idx + 1];
  }

  void start_compile() const {}

  void end_compile() const {}

  bool switch_func(IRFuncRef func) {
    cur_func = static_cast<u32>(func);

    highest_local_val_idx = 0;
    const auto &info = ir->functions[cur_func];
    assert(info.block_begin_idx != info.block_end_idx);

    highest_local_val_idx =
        ir->blocks[info.block_end_idx - 1].inst_end_idx - info.arg_begin_idx;
    if (highest_local_val_idx > 0) {
      --highest_local_val_idx;
    }
    return true;
  }

  void reset() { highest_local_val_idx = 0; }
};

static_assert(tpde::IRAdaptor<TestIRAdaptor>);

} // namespace tpde::test
