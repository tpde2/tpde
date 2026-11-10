# SPDX-FileCopyrightText: 2026 Contributors to TPDE <https://tpde.org>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# 522 named stack arguments occupy 4176 bytes, beyond an unshifted ADD
# immediate and not a multiple of 4096. This needs register materialization.
parameters = ", ".join(f"i64 %a{i}" for i in range(530))
print(f"define void @large({parameters}, ...) {{ ret void }}")
