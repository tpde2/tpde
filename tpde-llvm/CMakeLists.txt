# SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.23)

project(tpde_llvm)


# tpde_llvm library

if (NOT TPDE_ENABLE_ENCODEGEN)
    message(FATAL_ERROR "TPDE_ENABLE_LLVM needs TPDE_ENABLE_ENCODEGEN")
endif ()

# We first create an object library that does NOT link against LLVM, this is
# required for the plugin. Later we create a static library for use with e.g.
# executables that also includes LLVM in its public dependencies.
add_library(tpde_llvm_impl OBJECT)
set_target_properties(tpde_llvm_impl PROPERTIES
    VERSION 0.1.0
    LINKER_LANGUAGE CXX
)

target_link_libraries(tpde_llvm_impl PRIVATE tpde)

# Configure LLVM
set(TPDE_LINK_LLVM_STATIC FALSE CACHE BOOL "Should LLVM be linked statically?")
# ASan is not very compatible with dynamically loaded plugins. It is, in
# principle, possible to preload libasan.so, but this requires knowledge of the
# system-dependent path, which is not viable for tests.
if (LLVM_VERSION_MAJOR LESS 22 OR TPDE_ENABLE_ASAN)
    set(TPDE_ENABLE_LLVM_PLUGIN FALSE)
else ()
    set(TPDE_ENABLE_LLVM_PLUGIN TRUE CACHE BOOL "Whether to build LLVM plugin")
endif ()

target_include_directories(tpde_llvm_impl SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(tpde_llvm_impl PUBLIC ${LLVM_DEFINITIONS_LIST})
if (NOT LLVM_ENABLE_RTTI)
    target_compile_options(tpde_llvm_impl PUBLIC -fno-rtti)
endif ()

target_sources(tpde_llvm_impl PRIVATE
    src/JITMapper.cpp
    src/LLVMAdaptor.cpp
    src/LLVMCompiler.cpp
    src/OrcCompiler.cpp

    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/tpde-llvm/LLVMCompiler.hpp
        include/tpde-llvm/OrcCompiler.hpp

    PRIVATE
    FILE_SET priv_headers TYPE HEADERS
    BASE_DIRS src
    FILES
        src/base.hpp
        src/JITMapper.hpp
        src/LLVMAdaptor.hpp
        src/LLVMCompilerBase.hpp
)

if (TPDE_ENABLE_PCH)
    target_precompile_headers(tpde_llvm_impl PRIVATE
        src/base.hpp
        <llvm/ADT/STLExtras.h>
        <llvm/ADT/SmallString.h>
        <llvm/ADT/SmallVector.h>
        <llvm/IR/InlineAsm.h>
        <llvm/IR/Instructions.h>
        <llvm/IR/IntrinsicInst.h>
        <llvm/IR/Module.h>
        <llvm/IR/Operator.h>
        <llvm/TargetParser/Triple.h>
        <fadec-enc2.h>
        <ranges>
        <variant>
    )
endif ()

set(TPDE_CLANG_HINTS "")
if (DEFINED LLVM_DIR)
    set(TPDE_CLANG_HINTS "${LLVM_DIR}/../clang")
endif ()

function(check_clang_version validator_result_var CANDIDATE)
    execute_process(COMMAND ${CANDIDATE} --version
                    RESULT_VARIABLE CANDIDATE_VERSION_RC
                    OUTPUT_VARIABLE CANDIDATE_VERSION_OUT)
    if (CANDIDATE_VERSION_RC EQUAL 0)
        string(REGEX MATCH "clang version [0-9]+.[0-9]+" CANDIDATE_VERSION_STR ${CANDIDATE_VERSION_OUT})
    endif ()
    if (NOT CANDIDATE_VERSION_STR)
        message(VERBOSE "Unable to determine version for Clang at ${CANDIDATE}")
        set(${validator_result_var} FALSE PARENT_SCOPE)
    else ()
        string(SUBSTRING ${CANDIDATE_VERSION_STR} 14 -1 CANDIDATE_VERSION)
        message(VERBOSE "Found Clang ${CANDIDATE_VERSION} at ${CANDIDATE}")
        if (${CANDIDATE_VERSION} VERSION_GREATER "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
            message(VERBOSE "Cannot use Clang newer than used LLVM version")
            set(${validator_result_var} FALSE PARENT_SCOPE)
        elseif (NOT ${CANDIDATE_VERSION} VERSION_EQUAL "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
            # TODO: run optimization in tpde-encodegen to avoid this.
            message(WARNING "Different Clang (${CANDIDATE_VERSION}) and LLVM "
                    "(${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}) versions "
                    "might cause broken tests.")
        endif ()
    endif ()
endfunction()
find_program(TPDE_CLANG NO_CACHE REQUIRED
             NAMES clang-${LLVM_VERSION_MAJOR} clang-20 clang-19 clang-18 clang
             HINTS ${LLVM_TOOLS_BINARY_DIR}
             VALIDATOR check_clang_version)
message(STATUS "Found Clang: " ${TPDE_CLANG})

if ("x86_64" IN_LIST TPDE_TARGETS)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/encode_template_x64.bc
        COMMAND ${TPDE_CLANG} -c -emit-llvm -ffreestanding -fcf-protection=none
            -O3 -fomit-frame-pointer -fno-math-errno
            --target=x86_64-linux -march=x86-64-v4
            -o ${CMAKE_CURRENT_BINARY_DIR}/encode_template_x64.bc
            ${CMAKE_CURRENT_SOURCE_DIR}/src/encode_template.c
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/encode_template.c
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/encode_template_x64.hpp
        COMMAND tpde_encodegen
            -o ${CMAKE_CURRENT_BINARY_DIR}/encode_template_x64.hpp
            ${CMAKE_CURRENT_BINARY_DIR}/encode_template_x64.bc
        DEPENDS
            tpde_encodegen
            ${CMAKE_CURRENT_BINARY_DIR}/encode_template_x64.bc
    )

    target_compile_definitions(tpde_llvm_impl PRIVATE TPDE_LLVM_X64)
    target_sources(tpde_llvm_impl PRIVATE
        src/x64/LLVMCompilerX64.cpp

        PRIVATE
        FILE_SET priv_headers TYPE HEADERS
        BASE_DIRS src
        FILES
            src/x64/LLVMCompilerX64.hpp

        PRIVATE
        FILE_SET encodegen TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/encode_template_x64.hpp
    )
endif()

if ("aarch64" IN_LIST TPDE_TARGETS)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/encode_template_arm64.bc
        COMMAND ${TPDE_CLANG} -c -emit-llvm -ffreestanding -fcf-protection=none
            -O3 -fomit-frame-pointer -fno-math-errno
            --target=aarch64 -march=armv8.1-a
            -o ${CMAKE_CURRENT_BINARY_DIR}/encode_template_arm64.bc
            ${CMAKE_CURRENT_SOURCE_DIR}/src/encode_template.c
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/encode_template.c
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/encode_template_arm64.hpp
        COMMAND tpde_encodegen
            -o ${CMAKE_CURRENT_BINARY_DIR}/encode_template_arm64.hpp
            ${CMAKE_CURRENT_BINARY_DIR}/encode_template_arm64.bc
        DEPENDS
            tpde_encodegen
            ${CMAKE_CURRENT_BINARY_DIR}/encode_template_arm64.bc
    )

    target_compile_definitions(tpde_llvm_impl PRIVATE TPDE_LLVM_ARM64)
    target_sources(tpde_llvm_impl PRIVATE
        src/arm64/LLVMCompilerArm64.cpp

        PRIVATE
        FILE_SET priv_headers TYPE HEADERS
        BASE_DIRS src
        FILES
            src/arm64/LLVMCompilerArm64.hpp

        PRIVATE
        FILE_SET encodegen TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/encode_template_arm64.hpp
    )
endif()

# The library that does link against LLVM.
add_library(tpde_llvm)
add_library(tpde::tpde_llvm ALIAS tpde_llvm)
target_link_libraries(tpde_llvm PUBLIC tpde_llvm_impl)
if (TPDE_LINK_LLVM_STATIC)
    llvm_map_components_to_libnames(TPDE_LLVM_LIBS
        analysis core irreader irprinter jitlink orcjit passes support bitreader bitstreamreader targetparser
    )
    target_link_libraries(tpde_llvm PUBLIC ${TPDE_LLVM_LIBS})
else ()
    target_link_libraries(tpde_llvm PUBLIC LLVM)
endif ()


# tpde-llc binary

add_executable(tpde-llc tools/tpde-llc.cpp)
target_link_libraries(tpde-llc PRIVATE tpde_llvm)
# This is not really required, but is a simple way to make TPDE_LOGGING and
# spdlog available, so that the wrapper can enable logging.
target_link_libraries(tpde-llc PRIVATE tpde)

# general deps directory (for args)
target_include_directories(tpde-llc PRIVATE ../deps/)

list(APPEND TPDE_LLVM_TEST_DEPENDS tpde-llc)


# tpde-lli binary

add_executable(tpde-lli tools/tpde-lli.cpp)
target_link_libraries(tpde-lli PRIVATE tpde_llvm)
# This is not really required, but is a simple way to make TPDE_LOGGING and
# spdlog available, so that the wrapper can enable logging.
target_link_libraries(tpde-lli PRIVATE tpde)

# general deps directory (for args)
target_include_directories(tpde-lli PRIVATE ../deps/)

list(APPEND TPDE_LLVM_TEST_DEPENDS tpde-lli)


# tpde-plugin.so LLVM plugin

if (TPDE_ENABLE_LLVM_PLUGIN)
    if (LLVM_VERSION_MAJOR LESS 22)
        message(SEND_ERROR "TPDE LLVM plugin requires LLVM 22 or newer")
    endif ()
    add_library(tpde-plugin MODULE tools/tpde-plugin.cpp)
    target_link_libraries(tpde-plugin PRIVATE tpde_llvm_impl)
    set_target_properties(tpde-plugin PROPERTIES
        PREFIX ""
        SUFFIX ${LLVM_PLUGIN_EXT}
    )

    list(APPEND TPDE_LLVM_TEST_DEPENDS tpde-plugin)
endif ()


# Tests

if (TPDE_INCLUDE_TESTS)
    # configure lit.site.cfg.py
    tpde_canonicalize_bool(TPDE_ENABLE_LLVM_PLUGIN)
    configure_file(test/lit.site.cfg.py.in test/lit.site.cfg.py @ONLY)
    add_tpde_lit_testsuite(check-tpde-llvm "${CMAKE_CURRENT_BINARY_DIR}/test"
        DEPENDS ${TPDE_LLVM_TEST_DEPENDS})

    # Backwards-compatibility
    add_custom_target(tpde_llvm_filetest DEPENDS check-tpde-llvm)
endif ()


# Install
install(TARGETS tpde_llvm tpde-llc tpde-lli RUNTIME ARCHIVE FILE_SET HEADERS)
