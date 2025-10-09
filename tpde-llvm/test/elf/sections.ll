; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 < %s | llvm-readelf -Ss - | FileCheck %s
; RUN: tpde-llc --target=aarch64 < %s | llvm-readelf -Ss - | FileCheck %s

; See also llvm/test/CodeGen/X86/elf-unique-sections-by-flags.ll
; We don't implement all cases correctly.

; CHECK: Section Headers:
; CHECK-DAG: [{{ *}}[[SEC1_RX_IDX:[0-9]+]]] sec1 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0+}} AX 0 0 {{[0-9]+}}
; CHECK-DAG: [{{ *}}[[SEC1_RO_IDX:[0-9]+]]] sec1 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0+}} A 0 0 {{[0-9]+}}
; CHECK-DAG: [{{ *}}[[SEC1_RW_IDX:[0-9]+]]] sec1 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0+}} WA 0 0 {{[0-9]+}}
; CHECK-DAG: [{{ *}}[[SEC2_RW_IDX:[0-9]+]]] sec2 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0+}} WA 0 0 {{[0-9]+}}
; CHECK-DAG: [{{ *}}[[SEC2_RWT_IDX:[0-9]+]]] sec2 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0+}} WAT 0 0 {{[0-9]+}}

; CHECK: Symbol table '.symtab'
; CHECK-DAG: {{[0-9]+}} FUNC    GLOBAL DEFAULT [[SEC1_RX_IDX]]      fsec1
; CHECK-DAG:          4 OBJECT  GLOBAL DEFAULT [[SEC1_RO_IDX]]      csec1
; CHECK-DAG:          4 OBJECT  GLOBAL DEFAULT [[SEC1_RW_IDX]]      gsec1
; CHECK-DAG:          4 OBJECT  GLOBAL DEFAULT [[SEC2_RW_IDX]]      gsec2
; CHECK-DAG:          4 TLS     GLOBAL DEFAULT [[SEC2_RWT_IDX]]     tsec2

define void @fsec1() section "sec1" {
  ret void
}

@csec1 = constant i32 1, align 4, section "sec1"
@gsec1 = global i32 1, align 4, section "sec1"

@tsec2 = thread_local global i32 10, section "sec2", align 4
@gsec2 = global i32 10, section "sec2", align 4
