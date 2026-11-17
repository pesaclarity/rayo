#!/usr/bin/env bash
# The C tests under AddressSanitizer (with leak checking) and
# UndefinedBehaviorSanitizer, every kernel set, no recovery.
#   tool/ci/sanitize.sh [CC] [flags...]
source "$(dirname "$0")/common.sh"

cc=${1:-cc}
shift || true
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
echo "== C tests under ASan, LSan and UBSan: $cc $*"
build_tests "$cc" "$out/tests" -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
  -fno-omit-frame-pointer "$@"
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1
run_tests "$out/tests"
