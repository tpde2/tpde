; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: not tpde-lli %s 2>&1 | FileCheck %s
; RUN: not tpde-lli --orc %s 2>&1 | FileCheck %s --check-prefix=CHECK-ORC

; CHECK: unresolved symbol undefined
; CHECK-ORC: Symbols not found: [ undefined ]

declare void @undefined()
define i32 @main() {
  call void @undefined()
  ret i32 0
}
