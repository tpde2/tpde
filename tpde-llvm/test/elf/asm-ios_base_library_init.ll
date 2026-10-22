; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 < %s | llvm-readelf -s - | FileCheck %s
; RUN: tpde-llc --target=aarch64 < %s | llvm-readelf -s - | FileCheck %s

; CHECK: Symbol table '.symtab'
; CHECK: NOTYPE  GLOBAL DEFAULT   UND _ZSt21ios_base_library_initv

module asm ".globl _ZSt21ios_base_library_initv"
