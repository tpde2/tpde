; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: not tpde-llc --target=x86_64 %s | FileCheck %s
; RUN: not tpde-llc --target=aarch64 %s | FileCheck %s

; CHECK: type with incompatible layout at function/call: { x86_fp80, i64 }
; CHECK-NEXT: Failed to compile function f_x86_fp80_3
define void @f_x86_fp80_3({x86_fp80, i64}) {
  ret void
}

; CHECK: type with incompatible layout at function/call: [2 x x86_fp80]
; CHECK-NEXT: Failed to compile function f_x86_fp80_4
define void @f_x86_fp80_4([2 x x86_fp80]) {
  ret void
}

; CHECK: unsupported type: half
; CHECK-NEXT: Failed to compile function f_half_1
define void @f_half_1(ptr %p) {
  %l = load half, ptr %p
  ret void
}

; CHECK: unsupported type: half
; CHECK-NEXT: type with incompatible layout at function/call: half
; CHECK-NEXT: Failed to compile function f_half_2
define half @f_half_2() {
  ret half zeroinitializer
}

; CHECK: unsupported type: half
; CHECK-NEXT: type with incompatible layout at function/call: half
; CHECK-NEXT: Failed to compile function f_half_3
define void @f_half_3(half) {
  ret void
}

; CHECK: unsupported type: [2 x half]
; CHECK-NEXT: type with incompatible layout at function/call: [2 x half]
; CHECK-NEXT: Failed to compile function f_half_4
define void @f_half_4([2 x half]) {
  ret void
}

; CHECK: unsupported type: bfloat
; CHECK-NEXT: Failed to compile function f_bfloat_1
define void @f_bfloat_1(ptr %p) {
  %l = load bfloat, ptr %p
  ret void
}

; CHECK: unsupported type: bfloat
; CHECK-NEXT: type with incompatible layout at function/call: bfloat
; CHECK-NEXT: Failed to compile function f_bfloat_2
define bfloat @f_bfloat_2() {
  ret bfloat zeroinitializer
}

; CHECK: unsupported type: bfloat
; CHECK-NEXT: type with incompatible layout at function/call: bfloat
; CHECK-NEXT: Failed to compile function f_bfloat_3
define void @f_bfloat_3(bfloat) {
  ret void
}

; CHECK: unsupported type: [2 x bfloat]
; CHECK-NEXT: type with incompatible layout at function/call: [2 x bfloat]
; CHECK-NEXT: Failed to compile function f_bfloat_4
define void @f_bfloat_4([2 x bfloat]) {
  ret void
}

; CHECK: type with incompatible layout at function/call: <4 x i1>
; CHECK-NEXT: Failed to compile function f_v4i1_1
define <4 x i1> @f_v4i1_1() {
  ret <4 x i1> zeroinitializer
}

; CHECK: type with incompatible layout at function/call: <4 x i1>
; CHECK-NEXT: Failed to compile function f_v4i1_2
define void @f_v4i1_2(<4 x i1>) {
  ret void
}

; CHECK: type with incompatible layout at function/call: [2 x <4 x i1>]
; CHECK-NEXT: Failed to compile function f_v4i1_3
define void @f_v4i1_3([2 x <4 x i1>]) {
  ret void
}

; CHECK: type with incompatible layout at function/call: <4 x i12>
; CHECK-NEXT: Failed to compile function f_v4i12_1
define <4 x i12> @f_v4i12_1() {
  ret <4 x i12> zeroinitializer
}

; CHECK: type with incompatible layout at function/call: <4 x i12>
; CHECK-NEXT: Failed to compile function f_v4i12_2
define void @f_v4i12_2(<4 x i12>) {
  ret void
}

; CHECK: type with incompatible layout at function/call: [2 x <4 x i12>]
; CHECK-NEXT: Failed to compile function f_v4i12_3
define void @f_v4i12_3([2 x <4 x i12>]) {
  ret void
}

; CHECK: type with incompatible layout at function/call: <4 x i64>
; CHECK-NEXT: Failed to compile function f_v4i64_1
define <4 x i64> @f_v4i64_1() {
  ret <4 x i64> zeroinitializer
}

; CHECK: type with incompatible layout at function/call: <4 x i64>
; CHECK-NEXT: Failed to compile function f_v4i64_2
define void @f_v4i64_2(<4 x i64>) {
  ret void
}

; CHECK: type with incompatible layout at function/call: [2 x <4 x i64>]
; CHECK-NEXT: Failed to compile function f_v4i64_3
define void @f_v4i64_3([2 x <4 x i64>]) {
  ret void
}

; CHECK: type with incompatible layout at function/call: [2 x <4 x i25>]
; CHECK-NEXT: unsupported type: [2 x <4 x i25>]
; CHECK-NEXT: Failed to compile function f_call_v4i25_2
define void @f_call_v4i25_2(ptr %f) {
  call void %f([2 x <4 x i25>] zeroinitializer)
  ret void
}
