; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: tpde-llc --target=x86_64 -o /dev/null --time-trace=%t.json
; RUN: FileCheck --input-file=%t.json %s

; CHECK: "traceEvents"

define void @func() {
  ret void
}
