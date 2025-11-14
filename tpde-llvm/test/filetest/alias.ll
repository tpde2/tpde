; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; TODO: don't depend on linker and convert to readelf test.
; RUN: tpde-llc %s -o %t.o && clang -o %t %t.o && %t
; REQUIRES: os-linux

@alias = dso_local unnamed_addr alias void (), ptr @alias_impl

define void @alias_impl() {
entry:
  ret void
}

define i32 @main() {
  entry:
  call void @alias()
  ret i32 0
}
