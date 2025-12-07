; NOTE: Do not autogenerate
; SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: not tpde-llc --target=x86_64 %s | FileCheck %s
; RUN: not tpde-llc --target=aarch64 %s | FileCheck %s

@tls1 = thread_local global i32 0

; CHECK: thread-local global with unsupported uses: ptr @tls1
@glob1 = global ptr @tls1

@tls2 = thread_local global i32 0

; CHECK: thread-local global with unsupported uses: ptr @tls2
@glob2 = global {ptr, ptr, ptr} {ptr null, ptr @tls2, ptr @glob1}
