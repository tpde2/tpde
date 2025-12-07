// SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include <cassert>
#include <cstdint>

#ifdef TPDE_LOGGING
  #include <spdlog/spdlog.h>
  #define TPDE_LOG(level, ...)                                                 \
    (spdlog::should_log(level) ? spdlog::log(level, __VA_ARGS__) : (void)0)
  #ifndef NDEBUG
    #define TPDE_LOG_TRACE(...) TPDE_LOG(spdlog::level::trace, __VA_ARGS__)
    #define TPDE_LOG_DBG(...) TPDE_LOG(spdlog::level::debug, __VA_ARGS__)
  #else
    #define TPDE_LOG_TRACE(...)
    #define TPDE_LOG_DBG(...)
  #endif
  #define TPDE_LOG_INFO(...) TPDE_LOG(spdlog::level::info, __VA_ARGS__)
  #define TPDE_LOG_WARN(...) TPDE_LOG(spdlog::level::warn, __VA_ARGS__)
  #define TPDE_LOG_ERR(...) TPDE_LOG(spdlog::level::err, __VA_ARGS__)
#else
  #define TPDE_LOG_TRACE(...)
  #define TPDE_LOG_DBG(...)
  #define TPDE_LOG_INFO(...)
  #define TPDE_LOG_WARN(...)
  #define TPDE_LOG_ERR(...)
#endif

#ifndef NDEBUG
  #define TPDE_UNREACHABLE(msg) assert(0 && (msg))
#else
  #define TPDE_UNREACHABLE(msg) __builtin_unreachable()
#endif
#define TPDE_FATAL(msg) ::tpde::fatal_error(msg)

#if __has_cpp_attribute(clang::lifetimebound)
  #define TPDE_LIFETIMEBOUND [[clang::lifetimebound]]
#else
  #define TPDE_LIFETIMEBOUND
#endif

#if __has_builtin(__builtin_assume_separate_storage)
  #define TPDE_NOALIAS(a, b) __builtin_assume_separate_storage(a, b)
#else
  #define TPDE_NOALIAS(a, b)
#endif

namespace tpde {
// NOTE(ts): someone's gonna hate me...
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

#ifndef NDEBUG
constexpr bool WithAsserts = true;
#else
constexpr bool WithAsserts = false;
#endif

/// Abort program with a fatal error
[[noreturn]] void fatal_error(const char *msg) noexcept;

} // namespace tpde
