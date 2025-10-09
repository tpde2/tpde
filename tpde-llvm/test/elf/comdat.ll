; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 < %s | llvm-readelf -Ssg - | FileCheck %s
; RUN: tpde-llc --target=aarch64 < %s | llvm-readelf -Ssg - | FileCheck %s

; CHECK: Section Headers:
; CHECK-DAG: [{{ *}}[[C1_GROUP:[0-9]+]]] .group         GROUP    {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0*4}}     {{[0-9]+}} {{[0-9]+}} 4
; CHECK-DAG: [{{ *}}[[C1_DATA:[0-9]+]]]  .data{{[^ ]*}} PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0+}}  WAG 0 0 {{[0-9]+}}
; CHECK-DAG: [{{ *}}[[C2_GROUP:[0-9]+]]] .group         GROUP    {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0*4}}     {{[0-9]+}} {{[0-9]+}} 4
; CHECK-DAG: [{{ *}}[[C2_TEXT:[0-9]+]]]  .text{{[^ ]*}} PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0+}}  AXG 0 0 {{[0-9]+}}
; CHECK-DAG: [{{ *}}[[C2_DATA:[0-9]+]]]  .data{{[^ ]*}} PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{0+}}  WAG 0 0 {{[0-9]+}}

; CHECK: Symbol table '.symtab'
; CHECK-DAG:          4 OBJECT  GLOBAL DEFAULT [[C1_DATA]]      c1
; CHECK-DAG: {{[0-9]+}} FUNC    GLOBAL DEFAULT [[C2_TEXT]]      c2a
; CHECK-DAG:          4 OBJECT  GLOBAL DEFAULT [[C2_DATA]]      c2b

; CHECK: COMDAT group section [{{ *}}[[C1_GROUP]]] `.group' [c1] contains [[#]] sections:
; CHECK: [{{ *}}[[C1_DATA]]]

; CHECK: COMDAT group section [{{ *}}[[C2_GROUP]]] `.group' [c2] contains [[#]] sections:
; CHECK-DAG: [{{ *}}[[C2_DATA]]]
; CHECK-DAG: [{{ *}}[[C2_TEXT]]]

$c1 = comdat any

@c1 = global i32 1, align 4, comdat

$c2 = comdat any
define void @c2a() comdat($c2) { ret void }
@c2b = global i32 1, align 4, comdat($c2)
