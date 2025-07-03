; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: not tpde-llc --target=x86_64 %s | FileCheck %s
; RUN: not tpde-llc --target=aarch64 %s | FileCheck %s

; CHECK: unsupported type: <16 x i3>
; CHECK-NEXT: Failed to compile function f_v16i3
define void @f_v16i3(ptr %p) {
  %l = load <16 x i3>, ptr %p
  store <16 x i3> %l, ptr %p
  ret void
}

; CHECK: unsupported type: <16 x i12>
; CHECK-NEXT: Failed to compile function f_v16i12
define void @f_v16i12(ptr %p) {
  %l = load <16 x i12>, ptr %p
  store <16 x i12> %l, ptr %p
  ret void
}

; CHECK: unsupported type: <2 x i128>
; CHECK-NEXT: Failed to compile function f_v2i128
define void @f_v2i128(ptr %p) {
  %l = load <2 x i128>, ptr %p
  store <2 x i128> %l, ptr %p
  ret void
}

; CHECK: unsupported type: <65 x i1>
; CHECK-NEXT: Failed to compile function f_v65i1
define void @f_v65i1(ptr %p) {
  %l = load <65 x i1>, ptr %p
  store <65 x i1> %l, ptr %p
  ret void
}

; CHECK: unsupported type: <128 x i1>
; CHECK-NEXT: Failed to compile function f_v128i1
define void @f_v128i1(ptr %p) {
  %l = load <128 x i1>, ptr %p
  store <128 x i1> %l, ptr %p
  ret void
}

; CHECK: unsupported type: <8 x half>
; CHECK-NEXT: Failed to compile function f_v8half
define void @f_v8half(ptr %p) {
  %l = load <8 x half>, ptr %p
  store <8 x half> %l, ptr %p
  ret void
}

; CHECK: unsupported type: <8 x bfloat>
; CHECK-NEXT: Failed to compile function f_v8bfloat
define void @f_v8bfloat(ptr %p) {
  %l = load <8 x bfloat>, ptr %p
  store <8 x bfloat> %l, ptr %p
  ret void
}

; CHECK: unsupported type: <2 x ptr addrspace(1)>
; CHECK-NEXT: Failed to compile function f_v2p1
define void @f_v2p1(ptr %p) {
  %l = load <2 x ptr addrspace(1)>, ptr %p
  store <2 x ptr addrspace(1)> %l, ptr %p
  ret void
}

; CHECK: unsupported type: <2 x ptr addrspace(256)>
; CHECK-NEXT: Failed to compile function f_v2p256
define void @f_v2p256(ptr %p) {
  %l = load <2 x ptr addrspace(256)>, ptr %p
  store <2 x ptr addrspace(256)> %l, ptr %p
  ret void
}
