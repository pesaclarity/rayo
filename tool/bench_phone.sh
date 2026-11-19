#!/usr/bin/env bash
# rayo's language-model benchmark on an Android phone over adb: builds
# tools/slm_bench.c with the NDK for the phone's ABI, pushes it and the model
# to /data/local/tmp/rayo-bench, runs it for each thread count, and writes one
# JSON file with the phone's details and every run: the model's load time,
# prompt (prefill) and decode tokens a second, and the peak RSS and private
# memory. docs/PERFORMANCE.md has the steps for a 2 GB Android Go phone.
#
#   tool/bench_phone.sh MODEL.weights [--out DIR] [--threads "1 2"] [--prompt 128]
#                       [--decode 64] [--runs 5] [--kv int8|float32] [--context 1024]
#                       [--abi arm64-v8a|armeabi-v7a] [--label TEXT] [--keep]
#
# Needs adb (USB debugging on, one phone attached, or ANDROID_SERIAL set),
# the Android NDK (ANDROID_NDK_HOME, or ANDROID_HOME/ndk/<version>) and
# python3. --keep leaves the files on the phone.
set -euo pipefail
cd "$(dirname "$0")/.."

model=${1:?usage: tool/bench_phone.sh MODEL.weights [options]}
shift
out=bench-phone threads="1 2" prompt=128 decode=64 runs=5 kv=int8 context=1024
abi="" label="" keep=0
while [ $# -gt 0 ]; do
  case $1 in
    --out) out=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --prompt) prompt=$2; shift 2 ;;
    --decode) decode=$2; shift 2 ;;
    --runs) runs=$2; shift 2 ;;
    --kv) kv=$2; shift 2 ;;
    --context) context=$2; shift 2 ;;
    --abi) abi=$2; shift 2 ;;
    --label) label=$2; shift 2 ;;
    --keep) keep=1; shift ;;
    *) echo "bench_phone: unknown option $1" >&2; exit 2 ;;
  esac
done
[ -f "$model" ] || { echo "bench_phone: no model file $model" >&2; exit 2; }
command -v adb >/dev/null || { echo "bench_phone: adb not found" >&2; exit 1; }
adb get-state >/dev/null 2>&1 || { echo "bench_phone: no phone (adb devices)" >&2; exit 1; }

ndk=${ANDROID_NDK_HOME:-}
if [ -z "$ndk" ] && [ -n "${ANDROID_HOME:-}" ] && [ -d "$ANDROID_HOME/ndk" ]; then
  ndk=$(find "$ANDROID_HOME/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -1)
fi
[ -n "$ndk" ] && [ -d "$ndk" ] || { echo "bench_phone: set ANDROID_NDK_HOME" >&2; exit 1; }
host=$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64
bin="$ndk/toolchains/llvm/prebuilt/$host/bin"

prop() { adb shell getprop "$1" | tr -d '\r'; }
[ -n "$abi" ] || abi=$(prop ro.product.cpu.abi)
case $abi in
  arm64-v8a) cc="$bin/aarch64-linux-android24-clang" flags=() ;;
  armeabi-v7a) cc="$bin/armv7a-linux-androideabi24-clang" flags=(-mfpu=neon) ;;
  x86_64) cc="$bin/x86_64-linux-android24-clang" flags=(-msse4.2) ;;
  *) echo "bench_phone: ABI $abi is not one rayo builds for" >&2; exit 1 ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
echo "== building slm_bench for $abi"
source tool/ci/common.sh
cd "$RAYO_ROOT"
build_tool "$cc" tools/slm_bench.c "$work/slm_bench" -O2 ${flags[@]+"${flags[@]}"}

dir=/data/local/tmp/rayo-bench
name=$(basename "$model")
echo "== pushing to $dir"
adb shell mkdir -p "$dir"
adb push "$work/slm_bench" "$dir/" >/dev/null
adb push "$model" "$dir/$name" >/dev/null
adb shell chmod 755 "$dir/slm_bench"

mkdir -p "$out"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
device="$out/device-$stamp.json"
{
  echo "{"
  echo "  \"model\": \"$(prop ro.product.model)\","
  echo "  \"manufacturer\": \"$(prop ro.product.manufacturer)\","
  echo "  \"android\": \"$(prop ro.build.version.release)\","
  echo "  \"sdk\": \"$(prop ro.build.version.sdk)\","
  echo "  \"abi\": \"$abi\","
  echo "  \"low_ram\": \"$(prop ro.config.low_ram)\","
  echo "  \"soc\": \"$(prop ro.soc.model) $(prop ro.board.platform)\","
  echo "  \"mem_total_kb\": \"$(adb shell grep MemTotal /proc/meminfo | tr -s ' ' | cut -d' ' -f2 | tr -d '\r')\","
  echo "  \"cpus\": \"$(adb shell cat /sys/devices/system/cpu/possible | tr -d '\r')\","
  echo "  \"max_khz\": \"$(adb shell 'cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq 2>/dev/null' | tr -d '\r' | tr '\n' ' ')\","
  echo "  \"battery\": \"$(adb shell dumpsys battery | grep -E 'level|temperature' | tr -d '\r' | tr -s ' ' | tr '\n' ' ')\""
  echo "}"
} >"$device"
cat "$device"

results=()
for t in $threads; do
  echo "== $name, $t thread(s), $prompt prompt tokens, $decode decoded, $runs runs, $kv cache"
  # The phone rests between runs, so one run's heat does not slow the next.
  sleep 10
  adb shell "cd $dir && ./slm_bench $name $prompt $decode $t --kv $kv --context $context \
    --runs $runs --json run-$t.json --label '$label'"
  adb pull "$dir/run-$t.json" "$work/run-$t.json" >/dev/null
  results+=("$work/run-$t.json")
done
[ "$keep" = 1 ] || adb shell rm -rf "$dir"

file="$out/phone-$stamp.json"
python3 - "$file" "$device" "$label" "${results[@]}" <<'PY'
import json, sys
out, device, label, *runs = sys.argv[1:]
data = {"label": label, "device": json.load(open(device)),
        "runs": [json.load(open(r)) for r in runs]}
json.dump(data, open(out, "w"), indent=2)
print(f"\n{'threads':>7} {'load ms':>8} {'prefill tok/s':>14} {'decode tok/s':>13} "
      f"{'peak RSS MB':>12} {'private MB':>11}")
for r in data["runs"]:
    print(f"{r['threads']:>7} {r['load_ms']:>8.1f} {r['prefill_tok_s']:>14.1f} "
          f"{r['decode_tok_s']:>13.2f} {r['peak_rss_mb']:>12.1f} {r['peak_private_mb']:>11.1f}")
PY
rm -f "$device"
echo "bench_phone: $file"
