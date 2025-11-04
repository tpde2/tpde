// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <string>
#include <vector>

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

} // namespace tpde::test
