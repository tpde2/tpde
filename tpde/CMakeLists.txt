# SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.23)

project(tpde)


# tpde library

set(TPDE_LOGGING "ON" CACHE STRING "Enable logging with spdlog (DebugOnly/ON(default)/OFF)")
set_property(CACHE TPDE_LOGGING PROPERTY STRINGS DebugOnly ON OFF)

option(TPDE_X64 "enable x86-64 support" ON)
option(TPDE_A64 "enable AArch64 support" ON)

add_library(tpde)
add_library(tpde::tpde ALIAS tpde)
set_target_properties(tpde PROPERTIES
    VERSION 0.1.0
    LINKER_LANGUAGE CXX
)

# spdlog
if (((TPDE_LOGGING STREQUAL "DebugOnly") AND (CMAKE_BUILD_TYPE STREQUAL "Debug")) OR (TPDE_LOGGING STREQUAL "ON"))
    set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "TPDE compiles without exceptions")
    set(SPDLOG_USE_STD_FORMAT ON CACHE BOOL "TPDE uses C++20")
    add_subdirectory(../deps/spdlog ${CMAKE_CURRENT_BINARY_DIR}/deps/spdlog)

    target_compile_definitions(tpde PUBLIC TPDE_LOGGING)
    target_link_libraries(tpde PUBLIC spdlog::spdlog)
endif ()

target_sources(tpde PRIVATE
    src/Assembler.cpp
    src/AssemblerElf.cpp
    src/base.cpp
    src/ElfMapper.cpp
    src/StringTable.cpp
    src/ValueAssignment.cpp
    src/util/SmallVector.cpp

    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/tpde/Analyzer.hpp
        include/tpde/Assembler.hpp
        include/tpde/base.hpp
        include/tpde/Compiler.hpp
        include/tpde/CompilerBase.hpp
        include/tpde/AssemblerElf.hpp
        include/tpde/ElfMapper.hpp
        include/tpde/FunctionWriter.hpp
        include/tpde/IRAdaptor.hpp
        include/tpde/RegisterFile.hpp
        include/tpde/ScratchReg.hpp
        include/tpde/StringTable.hpp
        include/tpde/AssignmentPartRef.hpp
        include/tpde/ValuePartRef.hpp
        include/tpde/util/SmallBitSet.hpp
        include/tpde/util/SmallVector.hpp
        include/tpde/util/misc.hpp
)

# fadec
if (TPDE_X64)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(FADEC_DECODE ON BOOL "fadec decode for asserts")
    else ()
        set(FADEC_DECODE OFF BOOL "no fadec decode")
    endif ()
    set(FADEC_ENCODE OFF BOOL "no fadec encode")
    set(FADEC_ENCODE2 ON BOOL "fadec encode2 support")
    add_subdirectory(../deps/fadec ${CMAKE_CURRENT_BINARY_DIR}/deps/fadec)

    target_link_libraries(tpde PUBLIC fadec::fadec)
    target_sources(tpde PRIVATE
        src/x64/AssemblerElfX64.cpp

        PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/tpde/x64/AssemblerElfX64.hpp
            include/tpde/x64/CompilerX64.hpp
            include/tpde/x64/FunctionWriterX64.hpp
    )
endif ()

# disarm
if (TPDE_A64)
    add_subdirectory(../deps/disarm ${CMAKE_CURRENT_BINARY_DIR}/deps/disarm)

    target_link_libraries(tpde PUBLIC disarm64::disarm64)
    target_sources(tpde PRIVATE
        src/arm64/AssemblerElfA64.cpp

        PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/tpde/arm64/AssemblerElfA64.hpp
            include/tpde/arm64/CompilerA64.hpp
            include/tpde/arm64/FunctionWriterA64.hpp
    )
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(tpde PUBLIC TPDE_DEBUG)
    target_compile_definitions(tpde PUBLIC TPDE_ASSERTS)
endif ()


# tpde_test executable

add_executable(tpde_test)
target_link_libraries(tpde_test PUBLIC tpde)
target_include_directories(tpde_test PRIVATE ../deps/)
target_sources(tpde_test PRIVATE
    src/test/TestIR.cpp
    src/test/TestIRCompiler.cpp
    src/test/TestIRCompilerA64.cpp
    src/test/test_main.cpp

    PRIVATE
    FILE_SET HEADERS
    FILES
        src/test/TestIR.hpp
        src/test/TestIRCompiler.hpp
        src/test/TestIRCompilerA64.hpp
)


# Tests

if (TPDE_INCLUDE_TESTS)
    # configure lit.site.cfg.py
    configure_file(test/filetest/lit.site.cfg.py.in test/filetest/lit.site.cfg.py @ONLY)
    add_tpde_lit_testsuite(check-tpde-core "${CMAKE_CURRENT_BINARY_DIR}/test/filetest"
        DEPENDS tpde_test)

    # Backwards-compatibility
    add_custom_target(tpde_filetest DEPENDS check-tpde-core)
endif()
