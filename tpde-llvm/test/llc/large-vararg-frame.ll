; SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
; RUN: python3 %S/Inputs/large-vararg-frame.py > %t.ll
; RUN: tpde-llc --target=aarch64-unknown-linux-gnu %t.ll -o %t.o
; RUN: llvm-objdump -d %t.o | FileCheck %s
;
; CHECK: mov x16, #{{(4176|0x1050)}}
; CHECK-NEXT: add x17, x17, x16
