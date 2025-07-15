// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <concepts>
#include <type_traits>

#include "base.hpp"

namespace tpde {

class CCAssigner;

template <typename T, typename Base>
concept SameBaseAs =
    std::is_same_v<std::remove_const_t<std::remove_reference_t<T>>, Base>;

template <typename T>
concept CompilerConfig = requires {
  { T::FRAME_INDEXING_NEGATIVE } -> SameBaseAs<bool>;
  { T::PLATFORM_POINTER_SIZE } -> SameBaseAs<u32>;
  { T::NUM_BANKS } -> SameBaseAs<u32>;
  { T::DEFAULT_VAR_REF_HANDLING } -> SameBaseAs<bool>;

  typename T::DefaultCCAssigner;
  requires std::derived_from<typename T::DefaultCCAssigner, CCAssigner>;

  typename T::Assembler;
  typename T::AsmReg;
  typename T::FunctionWriter;
};

struct CompilerConfigDefault {
  constexpr static bool DEFAULT_VAR_REF_HANDLING = true;
};

} // namespace tpde
