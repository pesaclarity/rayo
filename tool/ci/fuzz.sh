#!/usr/bin/env bash
# rayo's fuzz targets (tests/fuzz/README.md).
#   tool/ci/fuzz.sh run [SECONDS] [TARGET...]   libFuzzer with ASan and UBSan (clang), each
#                                              target SECONDS (default 60) from its corpus
#   tool/ci/fuzz.sh replay [CC] [flags...]     every corpus file and tests/malformed through
#                                              each target, without libFuzzer (any compiler)
#   tool/ci/fuzz.sh afl OUT                     the targets for AFL++ (afl-clang-fast), in OUT
# FUZZ_OUT keeps the builds and the new corpus entries (default: a temporary directory).
source "$(dirname "$0")/common.sh"

TARGETS=(model graph tokenizer_table tokenizer_json tokenize)
mode=${1:-run}
shift || true

out="${FUZZ_OUT:-}"
if [ -z "$out" ]; then
  out="$(mktemp -d)"
  trap 'rm -rf "$out"' EXIT
fi
mkdir -p "$out"
export RAYO_FUZZ_DIR="$RAYO_ROOT/tests/fuzz"

# build CC OUT TARGET [flags...]
build() {
  local cc=$1 bin=$2 target=$3
  shift 3
  # shellcheck disable=SC2046
  "$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -Itests/fuzz "$@" $(lib_sources) tests/fuzz/common.c \
    "tests/fuzz/fuzz_$target.c" -lm -lpthread -o "$bin"
}

case "$mode" in
  run)
    seconds=${1:-60}
    shift || true
    [ $# -gt 0 ] && TARGETS=("$@")
    for t in "${TARGETS[@]}"; do
      echo "== fuzz $t, ${seconds}s"
      build clang "$out/fuzz_$t" "$t" -O1 -g -fsanitize=fuzzer,address,undefined \
        -fno-sanitize-recover=all -fno-omit-frame-pointer
      mkdir -p "$out/corpus/$t"
      dict=()
      if [ -f "tests/fuzz/$t.dict" ]; then dict=(-dict="tests/fuzz/$t.dict"); fi
      # New inputs go to $out/corpus/$t; the seeds in the repository are only read.
      status=0
      "$out/fuzz_$t" "$out/corpus/$t" "tests/fuzz/corpus/$t" "${dict[@]+"${dict[@]}"}" \
        -max_total_time="$seconds" -timeout=20 -rss_limit_mb=2560 -malloc_limit_mb=2048 \
        -max_len=1048576 -print_final_stats=1 >"$out/fuzz_$t.log" 2>&1 || status=$?
      grep -E '^(#[0-9]+.*DONE|stat::number_of_executed_units|stat::peak_rss_mb|SUMMARY)' \
        "$out/fuzz_$t.log" || true
      if [ "$status" -ne 0 ]; then
        tail -n 60 "$out/fuzz_$t.log"
        echo "fuzz $t: failed (the input is crash-*, timeout-* or oom-* in $PWD)"
        exit 1
      fi
    done
    ;;
  replay)
    cc=${1:-cc}
    shift || true
    for t in "${TARGETS[@]}"; do
      build "$cc" "$out/replay_$t" "$t" -O1 -g "$@" tests/fuzz/driver.c
      files=(tests/fuzz/corpus/"$t"/*)
      if [ -d "tests/malformed/$t" ]; then files+=(tests/malformed/"$t"/*); fi
      printf '%-16s ' "$t"
      "$out/replay_$t" "${files[@]}"
    done
    ;;
  afl)
    dest=${1:?tool/ci/fuzz.sh afl OUT}
    mkdir -p "$dest"
    for t in "${TARGETS[@]}"; do
      build "${AFL_CC:-afl-clang-fast}" "$dest/fuzz_$t" "$t" -O1 -g tests/fuzz/driver.c
    done
    echo "afl-fuzz -i tests/fuzz/corpus/model -o findings -- $dest/fuzz_model"
    ;;
  *)
    echo "tool/ci/fuzz.sh run|replay|afl" >&2
    exit 2
    ;;
esac
