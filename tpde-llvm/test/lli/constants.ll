; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-lli %s | FileCheck %s

@fmt_x64 = private constant [7 x i8] c"%016x\0A\00", align 1
@fmt_x64_x64 = private constant [13 x i8] c"%016x %016x\0A\00", align 1
declare i32 @printf(ptr, ...)

@consti24 = internal constant i24 19
@const18 = internal constant i64 mul (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64), i64 18)
@const_struct_size = internal constant i64 ptrtoint (ptr getelementptr ({i64, i32, i32}, ptr null, i32 1) to i64)
@const_ptrtoint64 = internal constant i64 ptrtoint (ptr @const_ptrtoint64 to i64)
@const_inttoptr64 = internal constant ptr inttoptr (i64 32 to ptr)
@const_inttoptr32 = internal constant ptr inttoptr (i32 33 to ptr)
@const_inttoptr128 = internal constant ptr inttoptr (i128 34 to ptr)
@const_inttoptr128_large = internal constant ptr inttoptr (i128 18446744073709551616 to ptr)

define i32 @main() {
; CHECK: 0000000000000013
  %consti24_ld = load i24, ptr @consti24
  %consti24_ext = zext i24 %consti24_ld to i64
  %consti24_p = call i32 (ptr, ...) @printf(ptr @fmt_x64, i64 %consti24_ext)
; CHECK: 0000000000000012
  %const18_ld = load i64, ptr @const18
  %const18_p = call i32 (ptr, ...) @printf(ptr @fmt_x64, i64 %const18_ld)
; CHECK: 0000000000000010
  %const_struct_size_ld = load i64, ptr @const_struct_size
  %const_struct_size_p = call i32 (ptr, ...) @printf(ptr @fmt_x64, i64 %const_struct_size_ld)
; CHECK: [[CONST_PTRTOINT64:[0-9a-f]{16}]] [[CONST_PTRTOINT64]]
  %const_ptrtoint64_ld = load i64, ptr @const_ptrtoint64
  %const_ptrtoint64_real = ptrtoint ptr @const_ptrtoint64 to i64
  %const_ptrtoint64_p = call i32 (ptr, ...) @printf(ptr @fmt_x64_x64, i64 %const_ptrtoint64_ld, i64 %const_ptrtoint64_real)
; CHECK: 0000000000000020
  %const_inttoptr64_ld = load ptr, ptr @const_inttoptr64
  %const_inttoptr64_cast = ptrtoint ptr %const_inttoptr64_ld to i64
  %const_inttoptr64_p = call i32 (ptr, ...) @printf(ptr @fmt_x64, i64 %const_inttoptr64_cast)
; CHECK: 0000000000000021
  %const_inttoptr32_ld = load ptr, ptr @const_inttoptr32
  %const_inttoptr32_cast = ptrtoint ptr %const_inttoptr32_ld to i64
  %const_inttoptr32_p = call i32 (ptr, ...) @printf(ptr @fmt_x64, i64 %const_inttoptr32_cast)
; CHECK: 0000000000000022
  %const_inttoptr128_ld = load ptr, ptr @const_inttoptr128
  %const_inttoptr128_cast = ptrtoint ptr %const_inttoptr128_ld to i64
  %const_inttoptr128_p = call i32 (ptr, ...) @printf(ptr @fmt_x64, i64 %const_inttoptr128_cast)
; CHECK: 0000000000000000
  %const_inttoptr128_large_ld = load ptr, ptr @const_inttoptr128_large
  %const_inttoptr128_large_cast = ptrtoint ptr %const_inttoptr128_large_ld to i64
  %const_inttoptr128_large_p = call i32 (ptr, ...) @printf(ptr @fmt_x64, i64 %const_inttoptr128_large_cast)

  ret i32 0
}
