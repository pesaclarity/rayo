#!/usr/bin/env bash
# rayo as WebAssembly with wasi-sdk (docs/WASM.md): the library and the ABI
# of wasm/rayo_wasm.c as reactor modules with no imports, a generic one and a
# SIMD128 one, and the C tests as WASI programs.
#   WASI_SDK=/opt/wasi-sdk wasm/build.sh [OUT]   (default: wasm/out)
source "$(dirname "$0")/../tool/ci/common.sh"

cc="${WASI_SDK:?set WASI_SDK to a wasi-sdk (25 or later)}/bin/clang"
out=${1:-wasm/out}
mkdir -p "$out"
flags=(--target=wasm32-wasip1 "${C89_FLAGS[@]}" "${INCLUDES[@]}" -O2 -ffp-contract=off
  -DRAYO_NO_THREADS)
# shellcheck disable=SC2054
module=(-mexec-model=reactor -Wl,--strip-all -Wl,-z,stack-size=1048576
  -Wl,--initial-memory=16777216 -Wl,--max-memory=268435456)
for v in "" -simd; do
  simd=()
  if [ -n "$v" ]; then simd=(-msimd128); fi
  # shellcheck disable=SC2046
  "$cc" "${flags[@]}" "${simd[@]+"${simd[@]}"}" "${module[@]}" $(lib_sources) wasm/rayo_wasm.c \
    -lm -o "$out/rayo$v.wasm"
  # shellcheck disable=SC2046
  "$cc" "${flags[@]}" "${simd[@]+"${simd[@]}"}" -Itests -Wl,-z,stack-size=8388608 $(lib_sources) \
    tests/*.c -lm -o "$out/tests$v.wasm"
done
ls -l "$out"
