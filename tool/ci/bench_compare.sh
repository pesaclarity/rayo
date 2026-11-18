#!/usr/bin/env bash
# Before and after: the Ask reader and writer and a language model of
# SmolLM2-135M's shape (synthetic weights, q4 and int8), built from BASE (a
# commit) and from this tree with the same compiler and flags, run in turn
# on this machine, each the median of ROUNDS rounds (default 3). Prints a
# Markdown table (and appends it to SUMMARY, if set), with the CPU and the
# instructions it has. Never a check: shared runners' numbers are
# indicative. Needs python3 with NumPy.
#   tool/ci/bench_compare.sh BASE [CC] [flags...]
source "$(dirname "$0")/common.sh"

base=${1:?usage: bench_compare.sh BASE [CC] [flags...]}
cc=${2:-cc}
shift 2 || shift $#
rounds=${ROUNDS:-3}
out="$(mktemp -d)"
cleanup() {
  git worktree remove --force "$out/base-tree" >/dev/null 2>&1 || true
  rm -rf "$out"
}
trap cleanup EXIT

git worktree add --detach "$out/base-tree" "$base" >/dev/null
for tree in base head; do
  dir=$RAYO_ROOT
  [ "$tree" = base ] && dir=$out/base-tree
  mkdir -p "$out/$tree"
  # Each tree's library and Ask programs; the language-model program is
  # this tree's for both (it only chooses kernels and prints), on each
  # tree's library.
  (
    cd "$dir" || exit 1
    build_tool "$cc" examples/ask/ask_reader_check.c "$out/$tree/reader" -O3 "$@"
    build_tool "$cc" examples/ask/ask_writer_bench.c "$out/$tree/writer" -O3 "$@"
    # shellcheck disable=SC2046
    "$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -I"$RAYO_ROOT/tools" -O3 "$@" $(lib_sources) \
      "$RAYO_ROOT/tools/bench.c" "$RAYO_ROOT/tools/slm_bench.c" -lm -lpthread -o "$out/$tree/slm"
  )
done
python3 tool/make_synthetic_slm.py smollm2-135m q4 "$out/q4.weights" >/dev/null
python3 tool/make_synthetic_slm.py smollm2-135m int8 "$out/int8.weights" >/dev/null

# name|program|arguments (the model's path as M4 or M8).
cases=(
  "reader best|reader|tests/ask/ask-reader.weights tests/ask/ask_reader_decodes.json 100"
  "reader simd|reader|tests/ask/ask-reader.weights tests/ask/ask_reader_decodes.json 100 simd"
  "writer best|writer|tests/ask/ask-writer.weights tests/ask/ask_writer_decodes.json 10"
  "writer simd|writer|tests/ask/ask-writer.weights tests/ask/ask_writer_decodes.json 10 simd"
  "q4 1 thread|slm|M4 128 64 1 --kv int8"
  "q4 2 threads|slm|M4 128 64 2 --kv int8"
  "q4 1 thread simd|slm|M4 128 64 1 --kv int8 --kernels simd"
  "int8 1 thread|slm|M8 128 64 1 --kv int8"
  "int8 2 threads|slm|M8 128 64 2 --kv int8"
  "int8 1 thread simd|slm|M8 128 64 1 --kv int8 --kernels simd"
  "int8 900 tokens|slm|M8 896 32 1 --kv int8"
)
for r in $(seq "$rounds"); do
  for c in "${cases[@]}"; do
    IFS='|' read -r name prog args <<<"$c"
    args=${args//M4/$out/q4.weights}
    args=${args//M8/$out/int8.weights}
    for tree in base head; do
      # shellcheck disable=SC2086
      "$out/$tree/$prog" $args >"$out/run.txt" 2>&1 || echo "failed" >>"$out/run.txt"
      sed "s/^/$tree|$name|/" "$out/run.txt" >>"$out/raw.txt"
    done
  done
  echo "round $r of $rounds"
done

{
  echo "### Before and after: $(uname -sm), $cc $*"
  echo
  model=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)
  feats=$(grep -m1 -E '^(Features|flags)' /proc/cpuinfo 2>/dev/null || true)
  has() { if echo "$feats" | grep -qw "$1"; then echo yes; else echo no; fi; }
  if [ "$(uname -m)" = aarch64 ]; then
    echo "CPU: ${model:-unknown}, $(getconf _NPROCESSORS_ONLN) cores; dotprod (asimddp): $(has asimddp), i8mm: $(has i8mm), fp16 conversion: yes"
  else
    echo "CPU: ${model:-unknown}, $(getconf _NPROCESSORS_ONLN) cores; avx2: $(has avx2), avx_vnni: $(has avx_vnni), f16c: $(has f16c)"
  fi
  echo
  echo "Base $(git rev-parse --short "$base"), head $(git rev-parse --short HEAD); medians of $rounds rounds."
  echo
  python3 - "$out/raw.txt" <<'EOF'
import re
import statistics
import sys

runs = {}
kern = {}
for line in open(sys.argv[1], encoding="utf-8"):
    tree, name, text = line.rstrip("\n").split("|", 2)
    d = runs.setdefault((name, tree), {})
    m = re.search(r"per question median ([0-9.]+) ms", text)
    if m:
        d.setdefault("ms per question", []).append(float(m.group(1)))
    m = re.search(r"per reply median ([0-9.]+) ms", text)
    if m:
        d.setdefault("ms per reply", []).append(float(m.group(1)))
    m = re.match(r"\S+ (\S+): (labels|\d+ replies)", text)
    if m:
        kern[(name, tree)] = m.group(1)
    m = re.search(r"kernels=(\S+)", text)
    if m:
        kern[(name, tree)] = m.group(1)
    for key, label in (("prompt_tok_s", "prompt tok/s"), ("decode_tok_s", "decode tok/s"),
                       ("peak_rss_mb", "peak RSS MB"), ("peak_private_mb", "peak private MB")):
        m = re.search(key + r"=([0-9.]+)", text)
        if m:
            d.setdefault(label, []).append(float(m.group(1)))

def cell(name, label):
    b, h = runs.get((name, "base"), {}), runs.get((name, "head"), {})
    if label not in b or label not in h:
        return ""
    x, y = statistics.median(b[label]), statistics.median(h[label])
    change = (y / x - 1.0) * 100.0 if x else 0.0
    return "%.4g → %.4g (%s%.0f%%)" % (x, y, "+" if change >= 0 else "", change)


print("| case | kernels (base / head) | ms a call | prompt tok/s | decode tok/s | peak RSS MB | peak private MB |")
print("| --- | --- | --- | --- | --- | --- | --- |")
names = []
for name, _ in runs:
    if name not in names:
        names.append(name)
for name in names:
    k = "%s / %s" % (kern.get((name, "base"), "?"), kern.get((name, "head"), "?"))
    ms = cell(name, "ms per question") or cell(name, "ms per reply")
    print("| %s | %s | %s | %s | %s | %s | %s |" % (name, k, ms, cell(name, "prompt tok/s"),
          cell(name, "decode tok/s"), cell(name, "peak RSS MB"), cell(name, "peak private MB")))
print()
print("ms a call: the Ask reader per question, the writer per reply (lower is better); "
      "tok/s: higher is better. simd: the build's baseline kernels (NEON on arm64, what a "
      "phone without the dot product runs).")
EOF
} >"$out/table.md"
cat "$out/table.md"
if [ -n "${SUMMARY:-}" ]; then cat "$out/table.md" >>"$SUMMARY"; fi
