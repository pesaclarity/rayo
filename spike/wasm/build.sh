#!/bin/sh
# Builds rayo.wasm (generic) and rayo-simd.wasm (-msimd128, autovectorised)
# with wasi-sdk.  WASI_SDK=/path/to/wasi-sdk spike/wasm/build.sh
set -e
cd "$(dirname "$0")/../.."
CC="${WASI_SDK:?set WASI_SDK}/bin/clang"
OUT=spike/wasm/out; mkdir -p "$OUT"
SRC="$(ls src/*.c src/*/*.c) spike/wasm/rayo_wasm.c"
EXPORTS="rw_alloc rw_free rw_error rw_open rw_bind rw_input rw_run rw_output rw_kernels rw_close"
EX=""; for e in $EXPORTS; do EX="$EX -Wl,--export=$e"; done
COMMON="--target=wasm32-wasip1 -std=c89 -O2 -ffp-contract=off -Iinclude -Isrc -DRAYO_NO_THREADS \
  -mexec-model=reactor -Wl,--strip-all -Wl,-z,stack-size=1048576 -Wl,--initial-memory=16777216 -Wl,--max-memory=268435456 $EX"
$CC $COMMON $SRC -lm -o $OUT/rayo.wasm
$CC $COMMON -msimd128 $SRC -lm -o $OUT/rayo-simd.wasm
ls -l $OUT
