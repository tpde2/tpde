; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; REQUIRES: tpde-plugin
; RUN: llc -load-pass-plugin=%tpde-plugin -mtriple=x86_64 -filetype=obj -o %t < %s 2>&1 | FileCheck %s -check-prefixes=ERR
; RUN: %objdump < %t | FileCheck %s -check-prefixes=X64

; ERR: warning: TPDE compilation failed

define void @asm() {
; X64-LABEL: <asm>:
; X64: nop
  call void asm "nop", ""()
  ret void
}
