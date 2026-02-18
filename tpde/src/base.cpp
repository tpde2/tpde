// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tpde/base.hpp"

#include <cstdlib>
#include <stdexcept>

namespace tpde {

[[noreturn]] void fatal_error([[maybe_unused]] const char *msg) {
  TPDE_LOG_ERR("TPDE FATAL ERROR: {}", msg);
#ifdef __cpp_exceptions
  throw std::runtime_error(msg);
#else
  abort();
#endif
}

} // end namespace tpde
