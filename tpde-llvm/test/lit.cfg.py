# SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import lit.formats

from lit.llvm import llvm_config

config.name = 'TPDE-LLVM'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.ll', '.cpp', '.test']

config.test_source_root = os.path.dirname(__file__)
config.environment["FILECHECK_OPTS"] = "--enable-var-scope --dump-input-filter=all --allow-unused-prefixes=false"

# Tweak the PATH to include the tools dir and TPDE binaries.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.tpde_llvm_bin_dir, append_path=True)
config.substitutions.append(('tpde-llc', 'tpde-llc --regular-exit'))
config.substitutions.append(('%objdump', 'llvm-objdump -d -r --no-show-raw-insn --symbolize-operands --no-addresses --x86-asm-syntax=intel -'))

config.available_features.add(f'llvm{config.llvm_version}')
config.available_features.add(f'os-{config.system_name.lower()}')

# TODO(ts): arch config
