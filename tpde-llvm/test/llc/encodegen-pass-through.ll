; SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; Exercise encodegen itself: the final return move is removed after redirecting
; the result to its source. That source must retain its parameter reference.
; RUN: %tpde_encodegen %s -o %t.hpp
; RUN: FileCheck %s --input-file=%t.hpp
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: ::encode_identity(
; CHECK-NOT: param_0.reset()
; CHECK: try_salvage_or_materialize(param_0, scratch_di, 0, result_0.part_size());
define i64 @identity(i64 %x) {
  ret i64 %x
}

; Both halves depend on the input: computing the sign must not consume the
; reference still needed by the pass-through low half.
; CHECK-LABEL: ::encode_widen(
; CHECK-NOT: param_0.reset()
; CHECK: try_salvage_or_materialize(param_0, scratch_ax, 0, result_0.part_size());
define i128 @widen(i64 %x) {
  %r = sext i64 %x to i128
  ret i128 %r
}

; CHECK-LABEL: ::encode_duplicate(
; CHECK-NOT: param_0.reset()
; CHECK: GenericValuePart result_source_0{{.*}}gval_as_reg(param_0)
; CHECK: try_salvage_or_materialize(result_source_0,
; CHECK: try_salvage_or_materialize(param_0,
define { i64, i64 } @duplicate(i64 %x) {
  %lo = insertvalue { i64, i64 } poison, i64 %x, 0
  %r = insertvalue { i64, i64 } %lo, i64 %x, 1
  ret { i64, i64 } %r
}
