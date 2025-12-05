; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; REQUIRES: tpde-plugin
; RUN: llc -load-pass-plugin=%tpde-plugin -mtriple=x86_64 -filetype=obj < %s | %objdump | FileCheck %s -check-prefixes=X64
; RUN: llc -load-pass-plugin=%tpde-plugin -mtriple=aarch64 -filetype=obj < %s | %objdump | FileCheck %s -check-prefixes=ARM64
; RUN: not llc -load-pass-plugin=%tpde-plugin -mtriple=aarch64 -filetype=asm < %s 2>&1 | FileCheck %s -check-prefix=ERR

; ERR: error: TPDE only support compiling to object files

define i8 @ret_i8(i8 %a) {
; X64-LABEL: <ret_i8>:
; X64:         ret
;
; ARM64-LABEL: <ret_i8>:
; ARM64:         ret
  ret i8 %a
}
