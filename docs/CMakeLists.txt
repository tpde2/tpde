# SPDX-FileCopyrightText: 2025 Contributors to TPDE <https://tpde.org>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Doxygen REQUIRED)

file(GLOB_RECURSE TPDE_HEADERS ${PROJECT_SOURCE_DIR}/tpde/include/*)
file(GLOB_RECURSE TPDE_LLVM_HEADERS ${PROJECT_SOURCE_DIR}/tpde-llvm/include/*)
file(GLOB_RECURSE MARKDOWN_SOURCES ${PROJECT_SOURCE_DIR}/docs/*.md)

CONFIGURE_FILE(Doxyfile.in Doxyfile @ONLY)

add_custom_command(
	OUTPUT html
	COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
	# touch directory for dependency
	COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/html
	DEPENDS ${TPDE_HEADERS} ${TPDE_LLVM_HEADERS} ${MARKDOWN_SOURCES} Doxyfile.in
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_custom_target(docs ALL DEPENDS html)
