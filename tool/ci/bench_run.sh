#!/usr/bin/env bash
# The numbers the README's benchmark table shows, as one JSON file: the Ask
# reader and writer, a language model of SmolLM2-135M's shape with synthetic
# weights (q4 and int8, 1 and 2 threads), this machine's library size, and
# what the machine is. Each is the median of several runs.
#   tool/ci/bench_run.sh OUT.json [CC] [flags...]
# RUNS (default 5) and SLM_RUNS (default 3) set how many runs. Needs python3
# with NumPy (tool/make_synthetic_slm.py writes the models).
source "$(dirname "$0")/common.sh"

json=${1:?usage: bench_run.sh OUT.json [CC] [flags...]}
cc=${2:-cc}
shift 2 || shift $#
runs=${RUNS:-5}
slm_runs=${SLM_RUNS:-3}
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
raw="$out/raw"
mkdir -p "$raw"

build_tool "$cc" examples/ask/ask_reader_check.c "$out/reader" -O3 "$@"
build_tool "$cc" examples/ask/ask_writer_bench.c "$out/writer" -O3 "$@"
build_tool "$cc" tools/slm_bench.c "$out/slm" -O3 "$@"

echo "== Ask reader and writer, $runs runs"
for i in $(seq "$runs"); do
  "$out/reader" tests/ask/ask-reader.weights tests/ask/ask_reader_decodes.json 100 \
    | sed -n 1,3p >>"$raw/reader.txt"
  "$out/writer" tests/ask/ask-writer.weights tests/ask/ask_writer_decodes.json 10 \
    | sed -n 1,3p >>"$raw/writer.txt"
  echo "run $i"
done

for w in q4 int8; do
  python3 tool/make_synthetic_slm.py smollm2-135m "$w" "$out/slm.weights"
  for t in 1 2; do
    echo "== SmolLM2-135M shape, $w, $t threads, $slm_runs runs"
    for i in $(seq "$slm_runs"); do
      "$out/slm" "$out/slm.weights" 128 64 "$t" | tee -a "$raw/slm-$w.txt"
    done
  done
  rm -f "$out/slm.weights" "$out/slm.tokenizer.json"
done

# The library as a program links it: a stripped shared object of rayo_* only.
printf '{\n  global:\n    rayo_*;\n  local:\n    *;\n};\n' >"$out/rayo.map"
# shellcheck disable=SC2046
"$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -O3 "$@" -fPIC -shared -fvisibility=hidden \
  -ffunction-sections -fdata-sections $(lib_sources) -lm -lpthread -Wl,--gc-sections \
  -Wl,--version-script="$out/rayo.map" -o "$out/librayo.so"
strip --strip-unneeded "$out/librayo.so"
wc -c <"$out/librayo.so" | tr -d ' ' >"$raw/so_bytes.txt"

{
  echo "cc: $("$cc" --version | head -1)"
  echo "cflags: -O3 $*"
  echo "uname: $(uname -sm)"
  echo "cores: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)"
} >"$raw/machine.txt"
grep -m1 '^flags' /proc/cpuinfo >"$raw/cpuflags.txt" 2>/dev/null ||
  grep -m1 '^Features' /proc/cpuinfo >"$raw/cpuflags.txt" 2>/dev/null || true
lscpu >"$raw/lscpu.txt" 2>/dev/null || true
grep -m1 '^model name' /proc/cpuinfo >>"$raw/lscpu.txt" 2>/dev/null || true

python3 tool/ci/bench_results.py collect "$raw" "$json"
