# SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.25)

project(tpde2)

# TODO(ae): use target_compile_features(cxx_std_20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TPDE_INCLUDE_TESTS "enable test" ON)
option(TPDE_ENABLE_ENCODEGEN "enable tpde-encodegen tool (needs LLVM)" ON)
option(TPDE_ENABLE_LLVM "enable LLVM back-end (needs LLVM)" ON)
option(TPDE_ENABLE_COVERAGE "enable coverage instrumentation" OFF)
option(TPDE_BUILD_DOCS "Build documentation" OFF)

# warnings
if (MSVC)
    add_compile_options(/W4 /WX)
    # disable RTTI and exceptions
    add_compile_options(/GR- /EHsc /D_HAS_EXCEPTIONS=0)
else ()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # disable RTTI and exceptions
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti;-fno-exceptions>")

    # this is annoying
    add_compile_options(-Wno-missing-field-initializers)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # practically supported by every compiler
        add_compile_options(-Wno-nested-anon-types)
        # i want these sometimes
        add_compile_options(-Wno-old-style-cast)
        # We often add unreachable() to default cases
        add_compile_options(-Wno-covered-switch-default)
        add_compile_options(-Wsign-compare)
    endif ()

    add_compile_options(-ffunction-sections -fdata-sections)
    add_link_options(LINKER:--gc-sections)

    # for development builds use -Werror and enable sanitizers
    if (PROJECT_IS_TOP_LEVEL)
        add_compile_options(-Werror -Wno-error=deprecated-declarations)
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_compile_options(-fsanitize=address -D_GLIBCXX_ASSERTIONS)
            add_link_options(-fsanitize=address)
        endif ()
    endif ()
endif ()

if (TPDE_ENABLE_COVERAGE)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        error("coverage build only supported with Clang")
    endif ()
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
endif ()

if (TPDE_ENABLE_ENCODEGEN OR TPDE_ENABLE_LLVM)
    # LLVM versions we currently support. The first version should be the
    # "primary" versions where all tests pass (i.e., no XFAIL due to different
    # code generation).
    set(TPDE_LLVM_SUPPORTED_VERSIONS 20.1 19.1)

    # allow overriding the LLVM version used
    if (DEFINED LLVM_DIR)
        find_package(LLVM CONFIG)
    endif ()
    foreach (VERSION ${TPDE_LLVM_SUPPORTED_VERSIONS})
        if (NOT LLVM_FOUND)
            find_package(LLVM ${VERSION} CONFIG)
        endif ()
    endforeach ()
    if (NOT LLVM_FOUND)
        message(FATAL_ERROR "Unable to find suitable LLVM version "
                "(supported versions: ${TPDE_LLVM_SUPPORTED_VERSIONS})")
    endif()

    message(STATUS "Found LLVM ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    if (NOT "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}" IN_LIST TPDE_LLVM_SUPPORTED_VERSIONS)
        message(WARNING "Found LLVM, but version is unsupported "
                "(supported versions: ${TPDE_LLVM_SUPPORTED_VERSIONS})")
    endif ()

    if (LLVM_VERSION_MAJOR LESS 20)
        # See https://github.com/llvm/llvm-project/pull/123578
        add_compile_options(-Wno-error=cpp)
    endif ()
endif()

if (TPDE_INCLUDE_TESTS)
    find_program(LIT NAMES ${LLVM_DEFAULT_EXTERNAL_LIT} lit llvm-lit REQUIRED)
    message(STATUS "Found lit: ${LIT}")
endif ()

function(add_tpde_lit_target target)
    cmake_parse_arguments(ARG "" "" "DEPENDS" ${ARGN})
    add_custom_target(${target}
        COMMAND ${LIT} -sv ${ARG_UNPARSED_ARGUMENTS}
        USES_TERMINAL)
    if (ARG_DEPENDS)
        add_dependencies(${target} ${ARG_DEPENDS})
    endif ()
endfunction()

function(add_tpde_lit_testsuite target)
    cmake_parse_arguments(ARG "" "" "DEPENDS" ${ARGN})
    add_tpde_lit_target(${target} ${ARG_UNPARSED_ARGUMENTS} DEPENDS ${ARG_DEPENDS})
    set_property(GLOBAL APPEND PROPERTY TPDE_LIT_TESTSUITES ${ARG_UNPARSED_ARGUMENTS})
    set_property(GLOBAL APPEND PROPERTY TPDE_LIT_DEPENDS ${ARG_DEPENDS})
endfunction()

add_subdirectory(tpde)
if (TPDE_ENABLE_ENCODEGEN)
    add_subdirectory(tpde-encodegen)
endif ()
if (TPDE_ENABLE_LLVM)
    add_subdirectory(tpde-llvm)
endif ()

if (TPDE_INCLUDE_TESTS)
    get_property(TPDE_LIT_TESTSUITES GLOBAL PROPERTY TPDE_LIT_TESTSUITES)
    get_property(TPDE_LIT_DEPENDS GLOBAL PROPERTY TPDE_LIT_DEPENDS)
    add_tpde_lit_target(check-tpde ${TPDE_LIT_TESTSUITES} DEPENDS ${TPDE_LIT_DEPENDS})
endif ()

if (TPDE_BUILD_DOCS)
    add_subdirectory(docs)
endif ()
