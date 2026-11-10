; SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; LLVM represents the scaled halfword load with LDRHHroX. Its extension
; operand is a boolean, and the encoder must retain the element-size shift.
; RUN: %tpde_encodegen %s -o %t.hpp
; RUN: FileCheck %s --input-file=%t.hpp
target triple = "aarch64-unknown-linux-gnu"

; CHECK-LABEL: ::encode_load_index(
; CHECK: ASMD(LDRHr_lsl, {{.*}}, 1);
define i16 @load_index(ptr %base, i64 %index) {
  %p = getelementptr i16, ptr %base, i64 %index
  %v = load i16, ptr %p
  ret i16 %v
}
