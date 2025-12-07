; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: not tpde-llc --target=x86_64 %s | FileCheck %s
; RUN: not tpde-llc --target=aarch64 %s | FileCheck %s

; CHECK: unsupported type: <65537 x i8>
; CHECK-NEXT: Failed to compile function f_v65537i8
define void @f_v65537i8(ptr %p) {
  %l = load <65537 x i8>, ptr %p
  store <65537 x i8> %l, ptr %p
  ret void
}

; CHECK: unsupported type: [65537 x i8]
; CHECK-NEXT: Failed to compile function f_a65537i8
define void @f_a65537i8(ptr %p) {
  %l = load [65537 x i8], ptr %p
  store [65537 x i8] %l, ptr %p
  ret void
}

; CHECK: unsupported type: { [40000 x i8], [40000 x i8] }
; CHECK-NEXT: Failed to compile function f_sa40000i8a40000i8
define void @f_sa40000i8a40000i8(ptr %p) {
  %l = load {[40000 x i8], [40000 x i8]}, ptr %p
  store {[40000 x i8], [40000 x i8]} %l, ptr %p
  ret void
}
