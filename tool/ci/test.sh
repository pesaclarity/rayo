#!/usr/bin/env bash
# The C tests with compiler CC (default cc) and extra flags (such as
# -mavx2 -mfma), each kernel set in turn.
#   tool/ci/test.sh [CC] [flags...]
source "$(dirname "$0")/common.sh"

cc=${1:-cc}
shift || true
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
echo "== C tests: $cc $* ($(uname -m))"
build_tests "$cc" "$out/tests" -O2 "$@"
run_tests "$out/tests"
