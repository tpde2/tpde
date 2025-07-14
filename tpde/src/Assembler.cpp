// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/Assembler.hpp"

namespace tpde {

Assembler::~Assembler() = default;

void Assembler::reset() noexcept {
  sections.clear();
  section_allocator.reset();
}

} // namespace tpde
