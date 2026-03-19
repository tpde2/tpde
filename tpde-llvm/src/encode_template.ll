; SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; Clang only exposes 2 return registers with AAPCS, but LLVM exposes 8.
; Therefore, write these functions in LLVM IR.
define {i128, i1} @of_add_i128(i128 %a, i128 %b) { %r = call {i128, i1} @llvm.sadd.with.overflow(i128 %a, i128 %b) ret {i128, i1} %r}
define {i128, i1} @of_add_u128(i128 %a, i128 %b) { %r = call {i128, i1} @llvm.uadd.with.overflow(i128 %a, i128 %b) ret {i128, i1} %r}
define {i128, i1} @of_sub_i128(i128 %a, i128 %b) { %r = call {i128, i1} @llvm.ssub.with.overflow(i128 %a, i128 %b) ret {i128, i1} %r}
define {i128, i1} @of_sub_u128(i128 %a, i128 %b) { %r = call {i128, i1} @llvm.usub.with.overflow(i128 %a, i128 %b) ret {i128, i1} %r}
define {i128, i1} @of_mul_i128(i128 %a, i128 %b) { %r = call {i128, i1} @llvm.smul.with.overflow(i128 %a, i128 %b) ret {i128, i1} %r}
define {i128, i1} @of_mul_u128(i128 %a, i128 %b) { %r = call {i128, i1} @llvm.umul.with.overflow(i128 %a, i128 %b) ret {i128, i1} %r}
