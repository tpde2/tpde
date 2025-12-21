# SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import lit.formats
import os

from lit.llvm import llvm_config

config.name = 'TPDE'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.tir']

config.test_source_root = os.path.dirname(__file__)
config.environment["FILECHECK_OPTS"] = "--enable-var-scope --dump-input-filter=all --allow-unused-prefixes=false"

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.tpde_obj_root, append_path=True)
# Abort on ASan errors so that also tests running with "not" fail.
llvm_config.with_environment('ASAN_OPTIONS', "abort_on_error=1", append_path=True)
config.substitutions.append(("%tpde_test", "tpde_test"))
config.substitutions.append(('%objdump', 'llvm-objdump -d -r --no-show-raw-insn --symbolize-operands --no-addresses --x86-asm-syntax=intel -'))
