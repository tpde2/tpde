// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "CompilerConfig.hpp"
#include "base.hpp"
#include "tpde/Assembler.hpp"
#include <concepts>
#include <type_traits>

#ifdef ARG
  #error ARG is used as a temporary preprocessor macro
#endif

#define ARG(x) std::declval<x>()

namespace tpde {

class AssignmentPartRef;
class CCAssigner;
struct RegBank;

template <bool B>
concept IsTrue = (B == true);

template <typename T>
concept ValueParts = requires(T a) {
  /// Provides the number of parts for a value
  { a.count() } -> std::convertible_to<u32>;

  /// Provides the size in bytes of a value part (must be a power of two)
  { a.size_bytes(ARG(u32)) } -> std::convertible_to<u32>;

  /// Provides the bank for a value part
  { a.reg_bank(ARG(u32)) } -> std::convertible_to<RegBank>;
};

template <typename T>
concept ValRefSpecialStruct = requires(T a) {
  // Clang does not support std::is_corresponding_member.
  { a.mode } -> std::same_as<uint8_t &>;
  requires std::is_standard_layout_v<T>;
  requires offsetof(T, mode) == 0;
};

template <typename T, typename Config>
concept Compiler = CompilerConfig<Config> && requires(T a) {
  // mostly platform things
  { T::NUM_FIXED_ASSIGNMENTS } -> SameBaseAs<u32[Config::NUM_BANKS]>;

  // (func_idx)
  { a.start_func(ARG(u32)) };

  {
    a.prologue_assign_arg(
        ARG(CCAssigner *), ARG(u32), ARG(typename T::IRValueRef))
  };

  // This has to call assembler->finish_func
  // (func_idx)
  { a.finish_func(ARG(u32)) };

  // (reg_to_spill, frame_off, size)
  { a.spill_reg(ARG(typename Config::AsmReg), ARG(u32), ARG(u32)) };

  // (dst_reg, frame_off, size)
  { a.load_from_stack(ARG(typename Config::AsmReg), ARG(u32), ARG(u32)) };

  // (dst_reg, src_reg, size)
  {
    a.mov(ARG(typename Config::AsmReg), ARG(typename Config::AsmReg), ARG(u32))
  };

  // (value); might allocate register
  {
    a.gval_expr_as_reg(ARG(typename T::GenericValuePart &))
  } -> std::same_as<typename Config::AsmReg>;

  {
    a.select_fixed_assignment_reg(ARG(AssignmentPartRef),
                                  ARG(typename T::IRValueRef))
  } -> std::same_as<typename Config::AsmReg>;


  // mostly implementor stuff

  { a.cur_func_may_emit_calls() } -> std::convertible_to<bool>;

  /// Provides the personality function for the current function or
  /// an invalid SymRef otherwise.
  { a.cur_personality_func() } -> std::same_as<SymRef>;

  /// Provides a calling convention assigner for the current function.
  /// Optional, if not implemented, the default C calling convention will be
  /// used.
  { a.cur_cc_assigner() } -> std::convertible_to<CCAssigner *>;

  {
    a.try_force_fixed_assignment(ARG(typename T::IRValueRef))
  } -> std::convertible_to<bool>;

  { a.val_parts(ARG(typename T::IRValueRef)) } -> ValueParts;

  /// A compiler can provide a data structure that is a non-assignment ValueRef.
  /// This struct is used in a union, so it must be a standard-layout struct and
  /// have "uint8_t mode;" as first member, which must be >= 4.
  requires ValRefSpecialStruct<typename T::ValRefSpecial>;

  /// Provides the implementation to return special ValueRefs, e.g. for
  /// constants or globals.
  {
    a.val_ref_special(ARG(typename T::IRValueRef))
  } -> std::same_as<std::optional<typename T::ValRefSpecial>>;

  /// Provides the implementation to construct a ValuePartRef from a
  /// ValRefSpecial.
  {
    a.val_part_ref_special(ARG(typename T::ValRefSpecial &), ARG(u32))
  } -> std::same_as<typename T::ValuePart>;

  /// The compiler numbers functions and this gives the derived implementation
  /// a chance to save that mapping
  ///
  /// (func_ref, idx)
  { a.define_func_idx(ARG(typename T::IRFuncRef), ARG(u32)) };

  /// When the default variable reference handling is disabled, the
  /// implementation has to provide a few helpers to tell us what to do
  requires IsTrue<Config::DEFAULT_VAR_REF_HANDLING> || requires {
    /// Called when starting to compile a func.
    /// This may initialize variable reference assignments
    /// which should be live at the beginning of the function
    { a.setup_var_ref_assignments() };

    /// Load the address of a variable reference referred to
    /// by the AssignmentPartRef into a register
    // (dst_reg, ap_ref)
    {
      a.load_address_of_var_reference(ARG(typename Config::AsmReg),
                                      ARG(AssignmentPartRef))
    };
  };

  {
    a.compile_inst(ARG(typename T::IRInstRef), ARG(typename T::InstRange))
  } -> std::convertible_to<bool>;
};

} // namespace tpde
