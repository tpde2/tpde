; SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;
; Widening returns must materialize the pass-through low half at its actual
; width, even when instruction selection elides its register-to-register move.
; RUN: tpde-llc --target=x86_64-unknown-linux-gnu %s -o %t.x86.o
; RUN: tpde-llc --target=aarch64-unknown-linux-gnu %s -o %t.arm.o
; RUN: tpde-lli %s

define i128 @signed_widen(i64 %x) {
  %wide = sext i64 %x to i128
  ret i128 %wide
}

define i128 @unsigned_widen(i64 %x) {
  %wide = zext i64 %x to i128
  ret i128 %wide
}

@saved = global i128 0

; Keep the source live after widening, so its register cannot simply be
; consumed by the result. Verify the full stored value as well as the source.
define i1 @widen_live_source(i64 %x) {
  %wide = zext i64 %x to i128
  store volatile i128 %wide, ptr @saved
  %narrow = trunc i64 %x to i32
  %source_ok = icmp eq i32 %narrow, 1985229328
  %stored = load volatile i128, ptr @saved
  %result_ok = icmp eq i128 %stored, 18364758544493064720
  %ok = and i1 %source_ok, %result_ok
  ret i1 %ok
}

define i32 @main() {
  %signed = call i128 @signed_widen(i64 -81985529216486896)
  %unsigned = call i128 @unsigned_widen(i64 -81985529216486896)
  %signed_ok = icmp eq i128 %signed, -81985529216486896
  %unsigned_ok = icmp eq i128 %unsigned, 18364758544493064720
  %ok = and i1 %signed_ok, %unsigned_ok
  %copy_ok = call i1 @widen_live_source(i64 -81985529216486896)
  %all_ok = and i1 %ok, %copy_ok
  %failed = xor i1 %all_ok, true
  %status = zext i1 %failed to i32
  ret i32 %status
}
