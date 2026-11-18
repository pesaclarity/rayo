#!/usr/bin/env bash
# The same floats from two builds: tests/builds/outputs.c built with gcc -O2
# and with clang -O3 and the host's SIMD flags (-mavx2 -mfma on x86_64), then
# every output of every test model, with each kernel set, compared bit for
# bit. rayo is compiled with -ffp-contract=off (common.sh), so no compiler
# fuses a multiply and an add on its own.
#   tool/ci/builds.sh
source "$(dirname "$0")/common.sh"

for cc in gcc clang; do
  command -v "$cc" >/dev/null || { echo "builds.sh: needs $cc" >&2; exit 1; }
done
if [ "$(uname -m)" = x86_64 ]; then simd=(-mavx2 -mfma); else simd=(-O3); fi
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
models=(tests/models/*.weights tests/ask/*.weights examples/classifier/classifier.weights)

echo "== the same floats from gcc -O2 and clang -O3 ${simd[*]} ($(uname -m))"
build_tool gcc tests/builds/outputs.c "$out/a" -O2
build_tool clang tests/builds/outputs.c "$out/b" -O3 "${simd[@]}"
"$out/a" "${models[@]}" >"$out/a.txt"
"$out/b" "${models[@]}" >"$out/b.txt"
if ! cmp -s "$out/a.txt" "$out/b.txt"; then
  diff "$out/a.txt" "$out/b.txt" | head -20 || true
  echo "builds.sh: the two builds give different floats" >&2
  exit 1
fi
echo "$(grep -c ', run ' "$out/a.txt") runs of ${#models[@]} models, $(grep -o '[0-9a-f]\{8\}' "$out/a.txt" | wc -l) floats: the same bits"
