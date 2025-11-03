// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <ranges>
#include <string>
#include <vector>

#include "tpde/ValLocalIdx.hpp"
#include "tpde/base.hpp"

namespace tpde::test {

struct TestIR {
  struct Value {
    enum class Type : u8 {
      normal,
      arg,
      phi,
      terminator,
    };

    enum class Op : u8 {
      none,
      any,
      add,
      sub,
      condselect,
      alloca,
      terminate,
      trap,
      ret,
      br,
      condbr,
      tbz,
      jump,
      call,
      zerofill,
    };

    enum class Cond : u8 {
      eq,
      neq,
      uge,
      ugt,
      ule,
      ult,
      sge,
      sgt,
      sle,
      slt,
    };

    struct OpInfo {
      std::string_view name;
      bool is_terminator;
      bool is_def;
      u32 op_count;
      u32 succ_count;
      u32 imm_count;
    };

    inline static constexpr OpInfo OP_INFOS[] = {
        // name                term              def          ops succ imm
        {    "<none>", false, false,   0,   0, 0},
        {       "any", false,  true, ~0u,   0, 0},
        {       "add", false,  true,   2,   0, 0},
        {       "sub", false,  true,   2,   0, 0},
        {"condselect", false,  true,   4,   0, 1},
        {    "alloca", false,  true,   0,   0, 2},
        { "terminate",  true, false,   0,   0, 0},
        {      "trap",  true, false,   0,   0, 0},
        {       "ret",  true, false,   1,   0, 0},
        {        "br",  true, false,   0,   1, 0},
        {    "condbr",  true, false,   1,   2, 0},
        {       "tbz",  true, false,   1,   2, 1},
        {      "jump",  true, false,   0, ~0u, 0},
        {      "call", false,  true, ~0u,   0, 0},
        {  "zerofill", false,  true,   0,   0, 1},
    };

    std::string name;
    Type type;
    Op op = Op::none;
    bool force_fixed_assignment = false;
    /// For call only: called function
    u32 call_func_idx;
    /// Number of value operands
    u32 op_count;

    // op_count value operands first, then blocks, then constants
    u32 op_begin_idx, op_end_idx;

    Value(Type type, std::string_view name) : name(name), type(type) {}
  };

  struct Block {
    std::string name;
    u32 succ_begin_idx = 0, succ_end_idx = 0;
    u32 inst_begin_idx = 0, phi_end_idx = 0, inst_end_idx = 0;
    u32 block_info = 0, block_info2 = 0;
  };

  struct Function {
    std::string name;
    bool declaration = false;
    bool local_only = false;
    bool has_call = false;
    u32 block_begin_idx = 0, block_end_idx = 0;
    u32 arg_begin_idx = 0, arg_end_idx = 0;
  };

  std::vector<Value> values;
  std::vector<u32> value_operands;
  std::vector<Block> blocks;
  std::vector<Function> functions;

  [[nodiscard]] bool parse_ir(std::string_view text) noexcept;

  void dump_debug() const noexcept;
  void print() const noexcept;
};

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

  [[nodiscard]] u32 func_count() const noexcept {
    return static_cast<u32>(ir->functions.size());
  }

  auto funcs() const noexcept {
    return std::views::iota(size_t{0}, ir->functions.size()) |
           std::views::transform([](u32 fn) { return IRFuncRef(fn); });
  }

  [[nodiscard]] auto funcs_to_compile() const noexcept { return funcs(); }

  [[nodiscard]] std::string_view
      func_link_name(const IRFuncRef func) const noexcept {
    return ir->functions[static_cast<u32>(func)].name;
  }

  [[nodiscard]] bool func_extern(const IRFuncRef func) const noexcept {
    return ir->functions[static_cast<u32>(func)].declaration;
  }

  [[nodiscard]] bool func_only_local(const IRFuncRef func) const noexcept {
    return ir->functions[static_cast<u32>(func)].local_only;
  }

  [[nodiscard]] bool func_has_weak_linkage(const IRFuncRef) const noexcept {
    return false;
  }

  u32 cur_func;

  [[nodiscard]] bool cur_needs_unwind_info() const noexcept { return false; }

  [[nodiscard]] bool cur_is_vararg() const noexcept { return false; }

  [[nodiscard]] u32 cur_highest_val_idx() const noexcept {
    return highest_local_val_idx;
  }

  [[nodiscard]] auto cur_args() const noexcept {
    const auto &func = ir->functions[cur_func];
    return std::views::iota(func.arg_begin_idx, func.arg_end_idx) |
           std::views::transform([](u32 val) { return IRValueRef(val); });
  }

  [[nodiscard]] static bool cur_arg_is_byval(u32) noexcept { return false; }

  [[nodiscard]] static u32 cur_arg_byval_align(u32) noexcept { return 0; }

  [[nodiscard]] static u32 cur_arg_byval_size(u32) noexcept { return 0; }

  [[nodiscard]] static bool cur_arg_is_sret(u32) noexcept { return false; }

  [[nodiscard]] auto cur_static_allocas() const noexcept {
    assert(ir->functions[cur_func].block_begin_idx !=
           ir->functions[cur_func].block_end_idx);
    const auto &block = ir->blocks[ir->functions[cur_func].block_begin_idx];
    return std::views::iota(block.inst_begin_idx, block.inst_end_idx) |
           std::views::filter([ir = ir](u32 val) {
             return ir->values[val].op == TestIR::Value::Op::alloca;
           }) |
           std::views::transform([](u32 val) { return IRValueRef(val); });
  }

  [[nodiscard]] static bool cur_has_dynamic_alloca() noexcept { return false; }

  [[nodiscard]] IRBlockRef cur_entry_block() const noexcept {
    const auto &func = ir->functions[cur_func];
    assert(ir->functions[cur_func].block_begin_idx !=
           ir->functions[cur_func].block_end_idx);

    return static_cast<IRBlockRef>(func.block_begin_idx);
  }

  [[nodiscard]] auto cur_blocks() const noexcept {
    const auto &func = ir->functions[cur_func];
    return std::views::iota(func.block_begin_idx, func.block_end_idx) |
           std::views::transform([](u32 val) { return IRBlockRef(val); });
  }

  [[nodiscard]] auto block_succs(const IRBlockRef block) const noexcept {
    const auto &info = ir->blocks[static_cast<u32>(block)];
    const auto *data = ir->value_operands.data();
    return std::ranges::subrange(data + info.succ_begin_idx,
                                 data + info.succ_end_idx) |
           std::views::transform([](u32 val) { return IRBlockRef(val); });
  }

  [[nodiscard]] auto block_insts(const IRBlockRef block) const noexcept {
    const auto &info = ir->blocks[static_cast<u32>(block)];
    return std::views::iota(info.phi_end_idx, info.inst_end_idx) |
           std::views::transform([](u32 val) { return IRInstRef(val); });
  }

  [[nodiscard]] auto block_phis(const IRBlockRef block) const noexcept {
    const auto &info = ir->blocks[static_cast<u32>(block)];
    return std::views::iota(info.inst_begin_idx, info.phi_end_idx) |
           std::views::transform([](u32 val) { return IRValueRef(val); });
  }

  [[nodiscard]] u32 block_info(IRBlockRef block) const noexcept {
    return ir->blocks[static_cast<u32>(block)].block_info;
  }

  void block_set_info(IRBlockRef block, u32 value) noexcept {
    ir->blocks[static_cast<u32>(block)].block_info = value;
  }

  [[nodiscard]] u32 block_info2(IRBlockRef block) const noexcept {
    return ir->blocks[static_cast<u32>(block)].block_info2;
  }

  void block_set_info2(IRBlockRef block, u32 value) noexcept {
    ir->blocks[static_cast<u32>(block)].block_info2 = value;
  }

  [[nodiscard]] std::string_view
      block_fmt_ref(IRBlockRef block) const noexcept {
    return ir->blocks[static_cast<u32>(block)].name;
  }

  [[nodiscard]] std::string_view value_fmt_ref(IRValueRef val) const noexcept {
    return ir->values[static_cast<u32>(val)].name;
  }

  [[nodiscard]] std::string_view inst_fmt_ref(IRInstRef inst) const noexcept {
    return ir->values[static_cast<u32>(inst)].name;
  }

  [[nodiscard]] tpde::ValLocalIdx val_local_idx(IRValueRef val) {
    assert(static_cast<u32>(val) >= ir->functions[cur_func].arg_begin_idx);
    return ValLocalIdx(u32(val) - ir->functions[cur_func].arg_begin_idx);
  }

  bool val_is_phi(IRValueRef val) const noexcept {
    return ir->values[u32(val)].type == TestIR::Value::Type::phi;
  }

  [[nodiscard]] auto inst_operands(IRInstRef inst) {
    const auto &info = ir->values[static_cast<u32>(inst)];
    const auto *data = ir->value_operands.data();
    return std::ranges::subrange(data + info.op_begin_idx,
                                 data + info.op_begin_idx + info.op_count) |
           std::views::transform([](u32 val) { return IRValueRef(val); });
  }

  [[nodiscard]] bool
      val_ignore_in_liveness_analysis(IRValueRef value) const noexcept {
    return ir->values[static_cast<u32>(value)].op == TestIR::Value::Op::alloca;
  }

  [[nodiscard]] auto inst_results(IRInstRef inst) const noexcept {
    const auto &info = ir->values[static_cast<u32>(inst)];
    bool is_def = TestIR::Value::OP_INFOS[static_cast<u32>(info.op)].is_def;
    return std::views::single(IRValueRef(inst)) | std::views::drop(!is_def);
  }

  static bool inst_fused(IRInstRef) noexcept { return false; }

  [[nodiscard]] auto val_as_phi(IRValueRef value) const noexcept {
    struct PHIRef {
      const u32 *op_begin, *block_begin;

      [[nodiscard]] u32 incoming_count() const noexcept {
        return block_begin - op_begin;
      }

      [[nodiscard]] IRValueRef
          incoming_val_for_slot(const u32 slot) const noexcept {
        assert(slot < incoming_count());
        return static_cast<IRValueRef>(op_begin[slot]);
      }

      [[nodiscard]] IRBlockRef
          incoming_block_for_slot(const u32 slot) const noexcept {
        assert(slot < incoming_count());
        return static_cast<IRBlockRef>(block_begin[slot]);
      }

      [[nodiscard]] IRValueRef
          incoming_val_for_block(const IRBlockRef block_ref) const noexcept {
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

  [[nodiscard]] u32 val_alloca_size(IRValueRef value) const noexcept {
    const auto val_idx = static_cast<u32>(value);
    assert(ir->values[val_idx].op == TestIR::Value::Op::alloca);
    const auto *data = ir->value_operands.data();
    return data[ir->values[val_idx].op_begin_idx];
  }

  [[nodiscard]] u32 val_alloca_align(IRValueRef value) const noexcept {
    const auto val_idx = static_cast<u32>(value);
    assert(ir->values[val_idx].op == TestIR::Value::Op::alloca);
    const auto *data = ir->value_operands.data();
    return data[ir->values[val_idx].op_begin_idx + 1];
  }

  void start_compile() const noexcept {}

  void end_compile() const noexcept {}

  bool switch_func(IRFuncRef func) noexcept {
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

} // namespace tpde::test
