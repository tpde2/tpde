# SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.23)

project(tpde_llvm)


# tpde_llvm library

if (NOT TPDE_ENABLE_ENCODEGEN)
    message(FATAL_ERROR "TPDE_ENABLE_LLVM needs TPDE_ENABLE_ENCODEGEN")
endif ()

add_library(tpde_llvm)
add_library(tpde::tpde_llvm ALIAS tpde_llvm)
set_target_properties(tpde_llvm PROPERTIES
    VERSION 0.1.0
    LINKER_LANGUAGE CXX
)

target_link_libraries(tpde_llvm PRIVATE tpde)

# hopscotch-map (disable warnings)
target_include_directories(tpde_llvm SYSTEM PRIVATE ../deps/hopscotch-map/include)

# Configure LLVM
set(TPDE_LINK_LLVM_STATIC FALSE CACHE BOOL "Should LLVM be linked statically?")

target_include_directories(tpde_llvm SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(tpde_llvm PUBLIC ${LLVM_DEFINITIONS})
if (TPDE_LINK_LLVM_STATIC)
    llvm_map_components_to_libnames(TPDE_LLVM_LIBS
        core irreader irprinter jitlink orcjit passes support bitreader bitstreamreader targetparser
    )
    target_link_libraries(tpde_llvm PUBLIC ${TPDE_LLVM_LIBS})
else ()
    target_link_libraries(tpde_llvm PUBLIC LLVM)
endif ()

target_sources(tpde_llvm PRIVATE
    src/JITMapper.cpp
    src/LLVMAdaptor.cpp
    src/LLVMCompiler.cpp

    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/tpde-llvm/LLVMCompiler.hpp

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
    target_precompile_headers(tpde_llvm PRIVATE
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

if (TPDE_X64)
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

    target_compile_definitions(tpde_llvm PRIVATE TPDE_LLVM_X64)
    target_sources(tpde_llvm PRIVATE
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
endif ()

if (TPDE_A64)
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

    target_compile_definitions(tpde_llvm PRIVATE TPDE_LLVM_ARM64)
    target_sources(tpde_llvm PRIVATE
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
endif ()


# tpde-llc binary

add_executable(tpde-llc tools/tpde-llc.cpp)
target_link_libraries(tpde-llc PRIVATE tpde_llvm)
# This is not really required, but is a simple way to make TPDE_LOGGING and
# spdlog available, so that the wrapper can enable logging.
target_link_libraries(tpde-llc PRIVATE tpde)

# general deps directory (for args)
target_include_directories(tpde-llc PRIVATE ../deps/)


# tpde-lli binary

add_executable(tpde-lli tools/tpde-lli.cpp)
target_link_libraries(tpde-lli PRIVATE tpde_llvm)
# This is not really required, but is a simple way to make TPDE_LOGGING and
# spdlog available, so that the wrapper can enable logging.
target_link_libraries(tpde-lli PRIVATE tpde)

# general deps directory (for args)
target_include_directories(tpde-lli PRIVATE ../deps/)


# Tests

if (TPDE_INCLUDE_TESTS)
    # configure lit.site.cfg.py
    configure_file(test/lit.site.cfg.py.in test/lit.site.cfg.py @ONLY)
    add_tpde_lit_testsuite(check-tpde-llvm "${CMAKE_CURRENT_BINARY_DIR}/test"
        DEPENDS tpde-llc tpde-lli)

    # Backwards-compatibility
    add_custom_target(tpde_llvm_filetest DEPENDS check-tpde-llvm)
endif ()
