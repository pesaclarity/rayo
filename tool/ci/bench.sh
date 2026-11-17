#!/usr/bin/env bash
# The Ask reader and writer timed with each kernel set this build has; with
# SUMMARY set (a file, such as $GITHUB_STEP_SUMMARY), a Markdown table too.
#   tool/ci/bench.sh [CC] [flags...]
source "$(dirname "$0")/common.sh"

cc=${1:-cc}
shift || true
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
build_tool "$cc" ask_reader_check "$out/reader" -O3 "$@"
build_tool "$cc" ask_writer_bench "$out/writer" -O3 "$@"
rows=()
for k in generic simd best; do
  r=$("$out/reader" tests/ask/ask-reader.weights tests/ask/ask_reader_decodes.json 100 "$k")
  w=$("$out/writer" tests/ask/ask-writer.weights tests/ask/ask_writer_decodes.json 10 "$k")
  echo "$r"
  echo "$w"
  kern=$(echo "$r" | head -1 | cut -d' ' -f2 | tr -d ':')
  rq=$(echo "$r" | sed -n 's/.*median \([0-9.]*\) ms, p90 \([0-9.]*\) ms.*/\1 | \2/p')
  wq=$(echo "$w" | sed -n 's/per reply median \([0-9.]*\) ms, p90 \([0-9.]*\) ms.*/\1 | \2/p')
  rows+=("| $kern | $rq | $wq |")
done
if [ -n "${SUMMARY:-}" ]; then
  {
    echo "### Bench: $(uname -sm), $cc $*"
    echo
    echo "| kernels | reader median ms | reader p90 ms | writer median ms/reply | writer p90 ms/reply |"
    echo "| --- | --- | --- | --- | --- |"
    printf '%s\n' "${rows[@]}"
  } >>"$SUMMARY"
fi
