; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: echo "ack  ack" | tpde-llc --target=x86_64 -o /dev/null --perf-control=/dev/stdout,/dev/stdin %s > %t
; RUN: FileCheck --input-file=%t %s

; CHECK: enable
; CHECK-NEXT: disable

define void @func() {
  ret void
}
