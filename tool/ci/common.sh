# Sourced by tool/check and the CI jobs: the C flags rayo is built with, and
# helpers. Everything rayo compiles is strict C89.
set -euo pipefail

RAYO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RAYO_ROOT"

C89_FLAGS=(-std=c89 -pedantic -Werror -Wall -Wextra -Wconversion -Wshadow
  -Wdeclaration-after-statement -Wvla)
INCLUDES=(-Iinclude -Isrc)

lib_sources() { find src -name '*.c' | LC_ALL=C sort; }
c_files() { find include src tests tools -name '*.[ch]' | LC_ALL=C sort; }

# Whether linking with --wrap works here (GNU ld, lld; not Apple's ld64).
can_wrap() {
  [ "$(uname -s)" != Darwin ]
}

# build_tests CC OUT [flags...]: the C test runner, with allocation counting
# where the linker can wrap malloc.
build_tests() {
  local cc=$1 out=$2
  shift 2
  local wrap=()
  if can_wrap; then
    wrap=(-DRAYO_TEST_COUNT_ALLOCS -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc)
  fi
  # shellcheck disable=SC2046
  "$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -Itests "$@" $(lib_sources) tests/*.c -lm \
    "${wrap[@]}" -o "$out"
}

# build_tool CC NAME OUT [flags...]: a tool from tools/.
build_tool() {
  local cc=$1 name=$2 out=$3
  shift 3
  # shellcheck disable=SC2046
  "$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -Itools "$@" $(lib_sources) tools/bench.c \
    "tools/$name.c" -lm -o "$out"
}

# run_tests BINARY: the whole suite with each kernel set the build has.
run_tests() {
  local bin=$1 k
  for k in generic simd best; do
    echo "-- kernels: $k"
    RAYO_TEST_KERNELS=$k "$bin"
  done
}
