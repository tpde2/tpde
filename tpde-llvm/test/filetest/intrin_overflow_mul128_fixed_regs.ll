; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; Regression test: the x64 encoding of llvm.smul.with.overflow.i128 needs up
; to 10 GP registers at once. Combined with 5 pre-existing fixed register
; assignments (values kept in a register across the whole function, one per
; callee-saved GP register), only 9 GP registers used to remain available,
; which made the register allocator abort with
; "TPDE FATAL ERROR: ran out of registers for scratch registers" instead of
; compiling successfully.
; RUN: tpde-llc --target=x86_64 %s -o %t.o

declare {i128, i1} @llvm.smul.with.overflow.i128(i128, i128)
declare void @use(i64, i64, i64, i64, i64)

define void @trigger_fixed_reg_pressure_smul128(i64 %x, i128 %a, i128 %b) {
entry:
  %v1 = add i64 %x, 1
  %v2 = add i64 %x, 2
  %v3 = add i64 %x, 3
  %v4 = add i64 %x, 4
  %v5 = add i64 %x, 5
  ; %v1..%v5 are live across the block boundary below, so the register
  ; allocator tries to give each of them a fixed (permanently-held) GP
  ; register, i.e. all 5 callee-saved GP registers (rbx, r12-r15).
  %r = call {i128, i1} @llvm.smul.with.overflow.i128(i128 %a, i128 %b)
  br label %exit

exit:
  call void @use(i64 %v1, i64 %v2, i64 %v3, i64 %v4, i64 %v5)
  ret void
}
