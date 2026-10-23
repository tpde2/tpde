; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: not tpde-llc --target=x86_64 %s | FileCheck %s
; RUN: not tpde-llc --target=aarch64 %s | FileCheck %s

; CHECK: Failed to compile instruction ret [10 x i32] zeroinitializer
; CHECK: Failed to compile function bad_ret
define [10 x i32] @bad_ret() {
  ret [10 x i32] zeroinitializer
}

; CHECK: Failed to compile instruction %x = call [10 x i32] @bad_ret()
; CHECK: Failed to compile function bad_call_ret
define void @bad_call_ret() {
  %x = call [10 x i32] @bad_ret()
  ret void
}
