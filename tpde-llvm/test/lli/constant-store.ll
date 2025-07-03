; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-lli %s | FileCheck %s

@fmt_x32 = private constant [6 x i8] c"%08x\0A\00", align 1
@fmt_x64 = private constant [9 x i8] c"%016llx\0A\00", align 1
@fmt_x64_x64 = private constant [13 x i8] c"%016x %016x\0A\00", align 1
declare i32 @printf(ptr, ...)

@buf64 = internal global [2 x i64] zeroinitializer

define i32 @main() {
; CHECK: 0807060504030201
  store <8 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>, ptr @buf64
  %test1_ld = load i64, ptr @buf64
  %test1_p = call i32 (ptr, ...) @printf(ptr @fmt_x64, i64 %test1_ld)

; CHECK: 04030201
  store <4 x i8> <i8 1, i8 2, i8 3, i8 4>, ptr @buf64
  %test2_ld = load i32, ptr @buf64
  %test2_p = call i32 (ptr, ...) @printf(ptr @fmt_x64, i32 %test2_ld)

; CHECK: [[TEST3:[0-9a-f]{16}]] [[TEST3]]
; CHECK: 0000000000000000
  store <2 x ptr> <ptr @buf64, ptr null>, ptr @buf64
  %test3_ld1 = load i64, ptr @buf64
  %test3_exp = ptrtoint ptr @buf64 to i64
  %test3_p1 = call i32 (ptr, ...) @printf(ptr @fmt_x64_x64, i64 %test3_ld1, i64 %test3_exp)
  %test3_idx2 = getelementptr i64, ptr @buf64, i32 1
  %test3_ld2 = load i64, ptr %test3_idx2
  %test3_p2 = call i32 (ptr, ...) @printf(ptr @fmt_x64, i64 %test3_ld2)

  ret i32 0
}
