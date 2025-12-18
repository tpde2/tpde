# SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.23)

project(tpde_encodegen)

add_executable(tpde_encodegen)
add_executable(tpde::tpde_encodegen ALIAS tpde_encodegen)

# Configure LLVM
set(TPDE_LINK_LLVM_STATIC FALSE CACHE BOOL "Should LLVM be linked statically?")

target_include_directories(tpde_encodegen SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(tpde_encodegen PRIVATE ${LLVM_DEFINITIONS})
target_compile_options(tpde_encodegen PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti;-fno-exceptions>")
if (TPDE_LINK_LLVM_STATIC)
    set(LLVM_COMPONENTS
        core CodeGen irreader irprinter passes mc support targetparser asmparser asmprinter bitreader bitstreamreader
    )

    if (x86_64 IN_LIST TPDE_TARGETS)
        list(APPEND LLVM_COMPONENTS X86)
    endif()
    if (aarch64 IN_LIST TPDE_TARGETS)
        list(APPEND LLVM_COMPONENTS AArch64)
    endif()

    llvm_map_components_to_libnames(TPDE_ENCODEGEN_LLVM_LIBS ${LLVM_COMPONENTS})
    target_link_libraries(tpde_encodegen PRIVATE ${TPDE_ENCODEGEN_LLVM_LIBS})
else ()
    target_link_libraries(tpde_encodegen PRIVATE LLVM)
endif ()

target_sources(tpde_encodegen PRIVATE
    src/main.cpp
    src/encode_gen.cpp
    src/arm64/Target.cpp
    src/x64/Target.cpp

    PRIVATE
    FILE_SET HEADERS
    BASE_DIRS src
    FILES
        src/encode_gen.hpp
        src/Target.hpp
        src/arm64/Target.hpp
        src/x64/Target.hpp
)

if (TPDE_ENABLE_PCH)
    target_precompile_headers(tpde_encodegen PRIVATE
        <llvm/ADT/DenseMap.h>
        <llvm/ADT/SmallVector.h>
        <llvm/CodeGen/AsmPrinter.h>
        <llvm/CodeGen/LivePhysRegs.h>
        <llvm/CodeGen/MachineConstantPool.h>
        <llvm/CodeGen/MachineFunction.h>
        <llvm/CodeGen/MachineInstr.h>
        <llvm/CodeGen/MachineModuleInfo.h>
        <llvm/CodeGen/MachineRegisterInfo.h>
        <llvm/IR/Constants.h>
        <llvm/IR/Function.h>
        <llvm/IR/Instructions.h>
        <llvm/IR/LLVMContext.h>
        <llvm/IR/Module.h>
        <llvm/IR/Verifier.h>
        <llvm/IRReader/IRReader.h>
        <llvm/MC/MCContext.h>
        <llvm/MC/MCObjectWriter.h>
        <llvm/MC/MCStreamer.h>
        <llvm/MC/TargetRegistry.h>
        <llvm/Target/TargetLoweringObjectFile.h>
        <llvm/Target/TargetMachine.h>
        <iostream>
        <format>
        <set>
        <string>
        <unordered_set>
    )
endif ()
