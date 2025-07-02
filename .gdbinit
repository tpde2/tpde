# SPDX-FileCopyrightText: 2024 Tobias Schwarz <tobias.schwarz@tum.de>
#
# SPDX-License-Identifier: CC0-1.0

python
import sys

sys.path.append('tpde/gdb')
import PrettyPrinters
PrettyPrinters.register_tpde_printers()

end

# intel syntax
set disassembly-flavor intel
