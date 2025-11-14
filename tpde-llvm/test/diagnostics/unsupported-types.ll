; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: not tpde-llc --target=x86_64 %s | FileCheck %s
; RUN: not tpde-llc --target=aarch64 %s | FileCheck %s

; CHECK: unsupported type: bfloat
; CHECK-NEXT: type with incompatible layout at function/call: bfloat
declare void @f(bfloat)

; CHECK: unsupported type: bfloat
; CHECK-NEXT: Failed to compile function bfloat
define void @bfloat() {
  call void @f(bfloat 1.0)
  ret void
}
