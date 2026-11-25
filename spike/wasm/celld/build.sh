#!/bin/sh
# Assembles dist/ for celld dev: rayo.mjs + the question table + the Worker, and rayo-simd.wasm.
# build.sh <ask_reader_decodes.json>
set -e; cd "$(dirname "$0")"; mkdir -p dist
cp ../out/${WASM:-rayo-simd.wasm} dist/rayo.wasm
{ sed 's/^export function/function/; s/^export const/const/' ../rayo.mjs
  node -e 'const d=require(process.argv[1]);const q={};for(const r of d.readings)q[r.question]=r.input;console.log("const QUESTIONS = "+JSON.stringify(q)+";")' "$1"
  cat worker.src.mjs; } > dist/index.mjs
