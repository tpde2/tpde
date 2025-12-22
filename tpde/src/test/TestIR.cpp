// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// For testing of the analyzer components, we need a small IR that we can
// parse from text files

// Quick specification of the format:
// All identifiers are alphanumeric starting with a letter
// Comments are done using ; at the beginning of a line
//
// Function and block names don't have a prefix when they are defined, value
// names and func/block names when they are used are prefixed with %
//
// clang-format off
// <funcName>(%<valName>, %<valName>, ...) {
// <blockName>:
//     %<valName> = alloca <size>
// ; No operands
//     %<valName> =
// ; Force fixed assignment if there is space
//     %<valName>! =
//     %<valName> = [operation] %<valName>, %<valName>, ...
//     jump %<blockName>, %<blockName>, ...
//     <or>
//     br %<blockName>
//     <or>
//     condbr %<valName>, %<trueTarget>, %<falseTarget>
//
// <blockName>:
//     %<valName> = phi [%<blockName>, %<valName>], [%<blockName>, %<valName>], ...
//     terminate
// }
//
// <funcName>...

// clang-format on

#include <algorithm>
#include <cctype>
#include <charconv>
#include <format>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "TestIR.hpp"

namespace tpde::test {

namespace {

class TestIRParser {
  struct Resolve {
    std::string_view name;
    u32 index;
  };

  std::string_view text;
  TestIR &ir;

  std::unordered_map<std::string_view, u32> funcs;
  std::vector<Resolve> func_resolves;

  // Per-block state
  std::unordered_map<std::string_view, u32> blocks;
  std::unordered_map<std::string_view, u32> values;
  std::vector<Resolve> block_resolves;
  std::vector<Resolve> value_resolves;

  struct ValueName {
    std::string_view name;
    bool force_fixed;
  };

public:
  TestIRParser(std::string_view text, TestIR &ir) : text(text), ir(ir) {}

private:
  void skip_whitespace(bool skip_newlines = true) {
    while (!text.empty()) {
      if (text[0] == ';') {
        auto comment_len = text.find('\n');
        if (comment_len == std::string::npos) {
          comment_len = text.size() - 1;
        }
        text = text.substr(comment_len);
      } else if (std::isspace(text[0])) {
        if (text[0] == '\n' && !skip_newlines) {
          break;
        }
        text.remove_prefix(1);
      } else {
        break;
      }
    }
  }

  bool try_read(char c) {
    if (text.empty() || text[0] != c) {
      return false;
    }
    text.remove_prefix(1);
    return true;
  }

  std::string_view parse_name() {
    if (text.empty() || !std::isalpha(text[0])) {
      TPDE_LOG_ERR("unable to parse name");
      return {};
    }
    auto is_ident = [](const auto c) { return !std::isalnum(c) && c != '_'; };
    auto len = std::find_if(text.begin(), text.end(), is_ident) - text.begin();
    auto name = text.substr(0, len);
    text.remove_prefix(len);
    return name;
  }

  ValueName parse_value_name() {
    skip_whitespace();
    if (text.empty() || text[0] != '%') {
      TPDE_LOG_ERR("expected value name");
      return ValueName{"", false};
    }
    text.remove_prefix(1);
    std::string_view name = parse_name();
    bool force_fixed = false;
    if (!text.empty() && text[0] == '!') {
      force_fixed = true;
      text.remove_prefix(1);
    }
    return ValueName{name, force_fixed};
  }

  bool parse_inst() {
    skip_whitespace();
    ValueName vname{"", false};
    if (!text.empty() && text[0] == '%') {
      vname = parse_value_name();
      if (vname.name.empty()) {
        return false;
      }
      skip_whitespace();
      if (!try_read('=')) {
        TPDE_LOG_ERR("expected =");
        return false;
      }
      skip_whitespace(false);
    }

    std::string_view op_name;
    if (!text.empty() && (text[0] == '%' || text[0] == '\n')) {
      // old-style "any" instruction with operands
      op_name = "any";
    } else {
      op_name = parse_name();
      if (op_name.empty()) {
        return false;
      }
    }

    if (op_name == "phi") {
      values[vname.name] = ir.values.size();
      ir.values.emplace_back(TestIR::Value::Type::phi, vname.name);
      ir.values.back().force_fixed_assignment = vname.force_fixed;
      auto op_begin = ir.value_operands.size();
      ir.values.back().op_begin_idx = op_begin;
      skip_whitespace(false);
      std::vector<std::string_view> block_names;
      while (!text.empty() && text[0] != '\n') {
        if (!try_read('[')) {
          break;
        }
        if (!try_read('^')) {
          TPDE_LOG_ERR("expected block name");
          return false;
        }
        auto block_name = parse_name();
        if (block_name.empty()) {
          return false;
        }
        block_names.push_back(block_name);
        skip_whitespace();
        if (!try_read(',')) {
          TPDE_LOG_ERR("expected , in phi tuple");
          return false;
        }
        skip_whitespace();
        if (!try_read('%')) {
          TPDE_LOG_ERR("expected value name");
          return false;
        }
        auto value_name = parse_name();
        if (value_name.empty()) {
          return false;
        }
        value_resolves.emplace_back(value_name, ir.value_operands.size());
        ir.value_operands.push_back(0);

        if (!try_read(']')) {
          TPDE_LOG_ERR("missing ]");
          return false;
        }
        if (!try_read(',')) {
          break;
        }
        skip_whitespace();
      }
      ir.values.back().op_count = ir.value_operands.size() - op_begin;
      for (auto name : block_names) {
        block_resolves.emplace_back(name, ir.value_operands.size());
        ir.value_operands.push_back(0);
      }
      ir.values.back().op_end_idx = ir.value_operands.size();
      return true;
    }

    const TestIR::Value::OpInfo *op_info = nullptr;
    TestIR::Value::Op op = TestIR::Value::Op::none;
    for (size_t i = 0; i < std::size(TestIR::Value::OP_INFOS); ++i) {
      if (TestIR::Value::OP_INFOS[i].name == op_name) {
        op = static_cast<TestIR::Value::Op>(i);
        op_info = &TestIR::Value::OP_INFOS[i];
        break;
      }
    }
    if (!op_info) {
      TPDE_LOG_ERR("invalid op '{}'", op_name);
      return false;
    }

    values[vname.name] = ir.values.size();
    auto type = TestIR::Value::Type::normal;
    if (op_info->is_terminator) {
      type = TestIR::Value::Type::terminator;
    }
    ir.values.emplace_back(type, vname.name);
    ir.values.back().force_fixed_assignment = vname.force_fixed;
    ir.values.back().op = op;
    ir.values.back().op_begin_idx = ir.value_operands.size();

    if (op == TestIR::Value::Op::call) {
      ir.functions.back().has_call = true;

      skip_whitespace(false);
      if (!try_read('@')) {
        TPDE_LOG_ERR("expected global function after call");
        return false;
      }
      std::string_view name = parse_name();
      if (name.empty()) {
        return false;
      }
      func_resolves.emplace_back(name, ir.values.size() - 1);

      skip_whitespace(false);
      if (try_read(',')) {
        skip_whitespace();
      }
    }

    u32 op_idx = 0;
    u32 block_begin = ~0u;
    u32 imm_begin = ~0u;
    skip_whitespace(false);
    while (!text.empty() && text[0] != '\n') {
      if (op_idx < block_begin && try_read('%')) {
        std::string_view operand = parse_name();
        if (operand.empty()) {
          return false;
        }
        value_resolves.emplace_back(operand, ir.value_operands.size());
        ir.value_operands.push_back(0);
      } else if (op_idx < imm_begin && try_read('^')) {
        if (block_begin == ~0u) {
          block_begin = op_idx;
        }
        std::string_view operand = parse_name();
        if (operand.empty()) {
          return false;
        }
        block_resolves.emplace_back(operand, ir.value_operands.size());
        ir.value_operands.push_back(0);
      } else {
        if (imm_begin == ~0u) {
          imm_begin = op_idx;
        }
        int base = 10;
        if (text.starts_with("0x")) {
          base = 16;
          text.remove_prefix(2);
        }

        u32 imm = 0;
        const auto res =
            std::from_chars(text.data(), text.data() + text.size(), imm, base);
        if (res.ec != std::errc{}) {
          TPDE_LOG_ERR("invalid immediate at '{}'", vname.name);
          return false;
        }
        text.remove_prefix(res.ptr - text.data());
        ir.value_operands.push_back(imm);
      }

      ++op_idx;
      skip_whitespace(false);
      if (!try_read(',')) {
        break;
      }
      skip_whitespace();
    }

    imm_begin = imm_begin == ~0u ? op_idx : imm_begin;
    block_begin = block_begin == ~0u ? imm_begin : block_begin;
    ir.values.back().op_count = block_begin;

    if (op_info->op_count != ~0u && block_begin != op_info->op_count) {
      TPDE_LOG_ERR("operand count mismatch for {}: expected {} got {}",
                   op_name,
                   op_info->op_count,
                   block_begin);
      return false;
    }
    if (op_info->succ_count != ~0u &&
        imm_begin - block_begin != op_info->succ_count) {
      TPDE_LOG_ERR("block count mismatch for {}: expected {} got {}",
                   op_name,
                   op_info->succ_count,
                   imm_begin - block_begin);
      return false;
    }
    if (op_info->imm_count != ~0u && op_idx - imm_begin != op_info->imm_count) {
      TPDE_LOG_ERR("imm count mismatch for {}: expected {} got {}",
                   op_name,
                   op_info->imm_count,
                   op_idx - imm_begin);
      return false;
    }

    if (op_info->is_terminator) {
      auto succ_begin = ir.values.back().op_begin_idx + block_begin;
      auto succ_end = ir.values.back().op_begin_idx + imm_begin;
      ir.blocks.back().succ_begin_idx = succ_begin;
      ir.blocks.back().succ_end_idx = succ_end;
    }

    ir.values.back().op_end_idx = ir.value_operands.size();

    return true;
  }

  bool parse_block() {
    skip_whitespace();
    auto name = parse_name();
    if (!try_read(':')) {
      TPDE_LOG_ERR("expected :");
      return false;
    }
    if (blocks.contains(name)) {
      TPDE_LOG_ERR("duplicate block name {}", name);
      return false;
    }
    TPDE_LOG_TRACE("parsing block {}", name);
    blocks[name] = ir.blocks.size();
    ir.blocks.push_back(TestIR::Block{.name = std::string(name)});
    ir.blocks.back().inst_begin_idx = ir.values.size();
    ir.blocks.back().phi_end_idx = ir.values.size();
    do {
      if (!parse_inst()) {
        return false;
      }
      if (ir.values.back().type == TestIR::Value::Type::phi) {
        if (ir.blocks.back().phi_end_idx != ir.values.size() - 1) {
          TPDE_LOG_ERR("phis must be at beginning of block");
          return false;
        }
        ir.blocks.back().phi_end_idx = ir.values.size();
      }
    } while (ir.values.back().type != TestIR::Value::Type::terminator);
    ir.blocks.back().inst_end_idx = ir.values.size();
    return true;
  }

  bool parse_func() {
    blocks.clear();
    values.clear();
    block_resolves.clear();
    value_resolves.clear();

    skip_whitespace();
    std::string_view name = parse_name();
    if (name.empty()) {
      return false;
    }
    bool is_declaration = false;
    if (name == "define" || name == "declare") {
      is_declaration = (name == "declare");
      // LLVM-IR compatibility for update_test_checks
      skip_whitespace();
      if (!try_read('@')) {
        TPDE_LOG_ERR("expected @ after LLVM-IR-compatible function definition");
        return false;
      }
      name = parse_name();
    }
    TPDE_LOG_TRACE("parsing function '{}'", name);
    if (funcs.contains(name)) {
      TPDE_LOG_ERR("Duplicate function {}", name);
      return false;
    }

    const auto func_idx = ir.functions.size();
    ir.functions.push_back(TestIR::Function{.name = std::string(name)});
    ir.functions.back().arg_begin_idx = ir.values.size();
    funcs[name] = func_idx;

    skip_whitespace();
    if (!try_read('(')) {
      TPDE_LOG_ERR("expected (");
      return false;
    }
    while (true) {
      skip_whitespace();
      if (try_read(')')) {
        break;
      }
      if (!values.empty()) {
        if (!try_read(',')) {
          TPDE_LOG_ERR("expected comma");
          return false;
        }
        skip_whitespace();
      }
      auto arg_name = parse_value_name();
      if (arg_name.name.empty()) {
        return false;
      }
      values[arg_name.name] = ir.values.size();
      ir.values.emplace_back(TestIR::Value::Type::arg, arg_name.name);
      ir.values.back().force_fixed_assignment = arg_name.force_fixed;
    }

    ir.functions.back().arg_end_idx = ir.values.size();

    skip_whitespace();
    if (is_declaration || try_read('!')) {
      ir.functions.back().declaration = true;
      return true;
    }
    if (text.starts_with("local")) {
      ir.functions.back().local_only = true;
      text.remove_prefix(5);
    }
    skip_whitespace();
    if (!try_read('{')) {
      TPDE_LOG_ERR("expected {");
      return false;
    }

    ir.functions.back().block_begin_idx = ir.blocks.size();
    while (true) {
      skip_whitespace();
      if (try_read('}')) {
        break;
      }
      if (!parse_block()) {
        return false;
      }
    }
    ir.functions.back().block_end_idx = ir.blocks.size();

    for (auto &r : block_resolves) {
      auto it = blocks.find(r.name);
      if (it == blocks.end()) {
        TPDE_LOG_ERR("use of undefined block {}", r.name);
        return false;
      }
      ir.value_operands[r.index] = it->second;
    }
    for (auto &r : value_resolves) {
      auto it = values.find(r.name);
      if (it == values.end()) {
        TPDE_LOG_ERR("use of undefined value {}", r.name);
        return false;
      }
      ir.value_operands[r.index] = it->second;
    }
    // TODO: validate dominance
    return true;
  }

public:
  bool parse() {
    while (true) {
      skip_whitespace();
      if (text.empty()) {
        break;
      }
      if (!parse_func()) {
        TPDE_LOG_ERR("Failed to parse function");
        return false;
      }
    }
    for (auto &r : func_resolves) {
      auto it = funcs.find(r.name);
      if (it == funcs.end()) {
        TPDE_LOG_ERR("use of undefined function {}", r.name);
        return false;
      }
      ir.values[r.index].call_func_idx = it->second;
      const TestIR::Function &func = ir.functions[it->second];
      u32 argc = func.arg_end_idx - func.arg_begin_idx;
      if (ir.values[r.index].op_count != argc) {
        TPDE_LOG_ERR("arg count mismatch in call {}, expected {} got {}",
                     func.name,
                     argc,
                     ir.values[r.index].op_count);
        return false;
      }
    }
    return true;
  }
};

} // end anonymous namespace

bool TestIR::parse_ir(std::string_view text) {
  return TestIRParser{text, *this}.parse();
}

void TestIR::dump_debug() const {
  TPDE_LOG_DBG("Dumping IR");

  for (u32 i = 0; i < functions.size(); ++i) {
    const auto &func = functions[i];
    if (func.declaration) {
      TPDE_LOG_DBG("Extern function {}: {}", i, func.name);
    } else if (func.local_only) {
      TPDE_LOG_DBG("Local function {}: {}", i, func.name);
    } else {
      TPDE_LOG_DBG("Function {}: {}", i, func.name);
    }

    TPDE_LOG_TRACE("  Arg {}->{}", func.arg_begin_idx, func.arg_end_idx);
    for (auto arg_idx = func.arg_begin_idx; arg_idx < func.arg_end_idx;
         ++arg_idx) {
      assert(values[arg_idx].type == Value::Type::arg);
      TPDE_LOG_DBG("  Arg {}: {}", arg_idx, values[arg_idx].name);
    }

    if (func.declaration) {
      continue;
    }

    TPDE_LOG_TRACE("  Block {}->{}", func.block_begin_idx, func.block_end_idx);
    for (auto block_idx = func.block_begin_idx; block_idx < func.block_end_idx;
         ++block_idx) {
      const auto &block = blocks[block_idx];
      TPDE_LOG_DBG("  Block {}: {}", block_idx, block.name);
      TPDE_LOG_TRACE(
          "    Succ {}->{}", block.succ_begin_idx, block.succ_end_idx);
      for (auto succ = block.succ_begin_idx; succ < block.succ_end_idx;
           ++succ) {
        const auto succ_idx = value_operands[succ];
        (void)succ_idx; // only used for debug output
        TPDE_LOG_DBG("    Succ {}: {}", succ_idx, blocks[succ_idx].name);
      }

      TPDE_LOG_TRACE(
          "    Inst {}->{}", block.inst_begin_idx, block.inst_end_idx);
      for (auto val_idx = block.inst_begin_idx; val_idx < block.inst_end_idx;
           ++val_idx) {
        const auto &val = values[val_idx];

        switch (val.type) {
          using enum Value::Type;
        case normal:
        case terminator: {
          const auto &info = Value::OP_INFOS[static_cast<u32>(val.op)];
          (void)info;
          TPDE_LOG_DBG("    Value {} ({}): {}", val_idx, info.name, val.name);
          for (auto op = val.op_begin_idx; op < val.op_end_idx; ++op) {
            const auto op_idx = value_operands[op];
            (void)op_idx; // only used for debug output
            if (op - val.op_begin_idx < val.op_count) {
              TPDE_LOG_DBG("      Op {}: {}", op_idx, values[op_idx].name);
            } else {
              TPDE_LOG_DBG("      Op ${}", op_idx);
            }
          }
          break;
        }
        case arg: TPDE_UNREACHABLE("argument must not be in instruction list");
        case phi: {
          TPDE_LOG_DBG("    PHI {}", val_idx);
          for (u32 op = 0; op < val.op_count; ++op) {
            const auto block_idx =
                value_operands[val.op_begin_idx + val.op_count + op];
            (void)block_idx; // only used for debug output
            const auto inc_idx = value_operands[val.op_begin_idx + op];
            (void)inc_idx; // only used for debug output
            TPDE_LOG_DBG("      {} from {}: {} from {}",
                         inc_idx,
                         block_idx,
                         values[inc_idx].name,
                         blocks[block_idx].name);
          }
          break;
        }
        }
      }
    }
  }
}

void TestIR::print() const {
  std::cout << std::format("Printing IR\n");

  for (u32 i = 0; i < functions.size(); ++i) {
    const auto &func = functions[i];
    if (func.declaration) {
      std::cout << std::format("Extern function {}\n", func.name);
    } else if (func.local_only) {
      std::cout << std::format("Local function {}\n", func.name);
    } else {
      std::cout << std::format("Function {}\n", func.name);
    }

    for (auto arg_idx = func.arg_begin_idx; arg_idx < func.arg_end_idx;
         ++arg_idx) {
      assert(values[arg_idx].type == Value::Type::arg);
      std::cout << std::format("  Argument {}\n", values[arg_idx].name);
    }

    if (func.declaration) {
      continue;
    }

    for (auto block_idx = func.block_begin_idx; block_idx < func.block_end_idx;
         ++block_idx) {
      const auto &block = blocks[block_idx];
      std::cout << std::format("  Block {}\n", block.name);
      for (auto succ = block.succ_begin_idx; succ < block.succ_end_idx;
           ++succ) {
        const auto succ_idx = value_operands[succ];
        std::cout << std::format("    Succ {}\n", blocks[succ_idx].name);
      }

      for (auto val_idx = block.inst_begin_idx; val_idx < block.inst_end_idx;
           ++val_idx) {
        const auto &val = values[val_idx];

        switch (val.type) {
          using enum Value::Type;
        case normal: {
        case terminator:
          const auto &info = Value::OP_INFOS[static_cast<u32>(val.op)];
          std::cout << std::format("    Value {} ({})\n", val.name, info.name);
          if (val.op == Value::Op::call) {
            std::cout << std::format("      Target {}\n",
                                     functions[val.call_func_idx].name);
          }
          for (auto op = val.op_begin_idx; op < val.op_end_idx; ++op) {
            const auto op_idx = value_operands[op];
            if (op - val.op_begin_idx < val.op_count) {
              std::cout << std::format("      Op {}\n", values[op_idx].name);
            } else if (val.op_end_idx - op > info.imm_count) {
              std::cout << std::format("      Op ^{}\n", blocks[op_idx].name);
            } else {
              std::cout << std::format("      Op ${}\n", op_idx);
            }
          }
          break;
        }
        case arg: TPDE_UNREACHABLE("argument must not be in instruction list");
        case phi: {
          std::cout << std::format("    PHI {}\n", values[val_idx].name);
          for (u32 op = 0; op < val.op_count; ++op) {
            const auto block_idx =
                value_operands[val.op_begin_idx + val.op_count + op];
            const auto inc_idx = value_operands[val.op_begin_idx + op];
            std::cout << std::format("      {} from {}\n",
                                     values[inc_idx].name,
                                     blocks[block_idx].name);
          }
          break;
        }
        }
      }
    }
  }
}

} // end namespace tpde::test
