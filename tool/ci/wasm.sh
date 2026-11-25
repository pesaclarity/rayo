#!/usr/bin/env bash
# rayo as WebAssembly (docs/WASM.md): wasm/build.sh's generic and SIMD128
# builds, the C tests under wasmtime with each kernel set (the SIMD128
# kernel against the generic one, the golden models), and the Ask reader
# through the JavaScript host in Node against TFLite's outputs.
#   WASI_SDK=... [WASMTIME=wasmtime] [NODE=node] tool/ci/wasm.sh
source "$(dirname "$0")/common.sh"

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
wasm/build.sh "$out"
for v in "" -simd; do
  echo "== C tests, rayo$v.wasm"
  for k in generic simd best; do
    echo "-- kernels: $k"
    "${WASMTIME:-wasmtime}" run --dir=. --env "RAYO_TEST_KERNELS=$k" "$out/tests$v.wasm"
  done
done
echo "== The Ask reader in ${NODE:-node}, through wasm/rayo.mjs"
"${NODE:-node}" examples/wasm/ask_reader.mjs "$out/rayo.wasm" "$out/rayo-simd.wasm"
