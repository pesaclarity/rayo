#!/usr/bin/env bash
# rayo_erff against the C library's erf on every float in [0, 4]
# (tests/erf/erf_check.c, C99). The library file itself stays C89.
#   tool/ci/erf.sh [CC]
source "$(dirname "$0")/common.sh"

cc=${1:-cc}
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
echo "== erf: every float in [0, 4], $cc ($(uname -m))"
"$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -O2 -c src/core/mathx.c -o "$out/mathx.o"
"$cc" -std=c99 -O2 -Wall -Wextra -Werror "${INCLUDES[@]}" tests/erf/erf_check.c "$out/mathx.o" \
  -lm -o "$out/erf_check"
"$out/erf_check"
