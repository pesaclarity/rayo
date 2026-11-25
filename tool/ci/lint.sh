#!/usr/bin/env bash
# Format, and every C file strict C89 with gcc and clang (whichever exist).
source "$(dirname "$0")/common.sh"

if command -v clang-format >/dev/null; then
  echo "== clang-format"
  # shellcheck disable=SC2046
  clang-format --dry-run --Werror $(c_files)
fi
for cc in gcc clang; do
  command -v "$cc" >/dev/null || continue
  echo "== C89 with $cc"
  # tests/erf is C99 on purpose: it calls the C library's erf (tool/ci/erf.sh).
  for f in $(find src tests tools examples wasm -name '*.c' -not -path 'tests/erf/*' | LC_ALL=C sort); do
    "$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -Itests -Itools -DRAYO_TEST_COUNT_ALLOCS \
      -fsyntax-only "$f"
  done
  # The SIMD files with their instruction sets on, where this host can.
  if [ "$(uname -m)" = x86_64 ]; then
    for f in src/kernels/*.c; do
      "$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -msse4.1 -fsyntax-only "$f"
      "$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -mavx2 -mfma -fsyntax-only "$f"
    done
  fi
done
