#!/usr/bin/env bash
# The chat example, built with make as docs/RUNNING-SLMS.md says, and a
# scripted conversation with the tiny test model (tests/models/llama*): two
# turns, a reset, greedy and sampled, which must end cleanly and report
# its numbers. With RUNNER set (such as qemu-arm), the program runs under it.
#   tool/ci/chat.sh [CC] [CFLAGS...]
source "$(dirname "$0")/common.sh"

cc=${1:-cc}
shift || true
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
echo "== chat example: $cc $*"
make -s clean
make -s CC="$cc" CFLAGS="-O2 ${*:-}" ${LIBS:+LIBS="$LIBS"}
for args in "--temp 0" "--temp 0.8 --seed 7 --threads 2"; do
  # shellcheck disable=SC2086
  printf 'hello there\nand the cat?\n/reset\nhi\n/quit\n' |
    ${RUNNER:-} examples/chat/chat tests/models/llama-q4.weights \
      --tokenizer tests/models/llama.tokenizer.json --max-tokens 8 $args \
      >"$out/reply.txt" 2>"$out/log.txt"
  cat "$out/log.txt"
  grep -q "reply: .* tokens/s" "$out/log.txt"
  grep -q "started again" "$out/log.txt"
done
# One raw prompt, with the float32 model.
${RUNNER:-} examples/chat/chat tests/models/llama.weights -p "the cat" --max-tokens 4 --temp 0 \
  >/dev/null 2>"$out/log.txt"
grep -q "reply: *4 tokens" "$out/log.txt"
make -s clean
echo "chat: ok"
