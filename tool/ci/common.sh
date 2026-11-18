# shellcheck shell=bash
# Sourced by tool/check and the CI jobs: the C flags rayo is built with, and
# helpers. Everything rayo compiles is strict C89, with -ffp-contract=off: no
# compiler fuses a multiply and an add into one rounding on its own (clang
# does by default, and with -mfma emits FMA instructions), so every build
# gives the same floats whatever its compiler and optimisation flags
# (tool/ci/builds.sh).
set -euo pipefail

RAYO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RAYO_ROOT"

C89_FLAGS=(-std=c89 -pedantic -Werror -Wall -Wextra -Wconversion -Wshadow
  -Wdeclaration-after-statement -Wvla -ffp-contract=off)
INCLUDES=(-Iinclude -Isrc)
# Hardening (docs/SECURITY.md), for every build but -fsyntax-only (lint.sh): a
# canary in each function with a buffer, the C library's checked copies
# where sizes are known (-U first: some compilers define it already), and
# no format string that is not a literal. Shared libraries also link with
# HARDEN_LDFLAGS: relocations read-only after loading, bound at once, and a
# stack that cannot be executed.
HARDEN_FLAGS=(-fstack-protector-strong -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2 -Wformat
  -Wformat-security)
HARDEN_LDFLAGS=(-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)

lib_sources() { find src -name '*.c' | LC_ALL=C sort; }
c_files() { find include src tests tools examples -name '*.[ch]' | LC_ALL=C sort; }

# Whether linking with --wrap works here (GNU ld, lld; not Apple's ld64).
can_wrap() {
  [ "$(uname -s)" != Darwin ]
}

# build_tests CC OUT [flags...]: the C test runner, with allocation counting
# where the linker can wrap malloc.
build_tests() {
  local cc=$1 out=$2
  shift 2
  # Never empty: bash 3.2 (macOS) treats an empty array as unset under set -u.
  local wrap=(-lm)
  if can_wrap; then
    # shellcheck disable=SC2054
    wrap+=(-DRAYO_TEST_COUNT_ALLOCS -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc)
  fi
  # shellcheck disable=SC2046
  "$cc" "${C89_FLAGS[@]}" "${HARDEN_FLAGS[@]}" "${INCLUDES[@]}" -Itests "$@" $(lib_sources) \
    tests/*.c -lm -lpthread "${wrap[@]}" -o "$out"
}

# build_tool CC SOURCE OUT [flags...]: a program (examples/*/*.c) with the
# timing helpers of tools/.
build_tool() {
  local cc=$1 src=$2 out=$3
  shift 3
  # shellcheck disable=SC2046
  "$cc" "${C89_FLAGS[@]}" "${HARDEN_FLAGS[@]}" "${INCLUDES[@]}" -Itools "$@" $(lib_sources) \
    tools/bench.c "$src" -lm -lpthread -o "$out"
}

# run_tests BINARY: the whole suite with each kernel set the build has.
run_tests() {
  local bin=$1 k
  for k in generic simd best; do
    echo "-- kernels: $k"
    RAYO_TEST_KERNELS=$k "$bin"
  done
}
