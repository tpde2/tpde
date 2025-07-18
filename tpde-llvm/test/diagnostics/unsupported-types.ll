; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: not tpde-llc --target=x86_64 %s | FileCheck %s
; RUN: not tpde-llc --target=aarch64 %s | FileCheck %s

; CHECK: unsupported type: x86_fp80
; CHECK-NEXT: type with incompatible layout at function/call: x86_fp80
declare void @f(x86_fp80)

; CHECK: unsupported type: x86_fp80
; CHECK-NEXT: Failed to compile function call_arg_x86_fp80
define void @call_arg_x86_fp80() {
  call void @f(x86_fp80 0xK3FFF8000000000000000)
  ret void
}
