#!/usr/bin/env bash
# The examples still build and give what they promise: examples/classifier
# against NumPy (expected.json), and the model file is what make_model.py
# writes.
#   tool/ci/examples.sh [CC]
source "$(dirname "$0")/common.sh"

cc=${1:-cc}
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
echo "== examples: classifier, $cc"
python3 examples/classifier/make_model.py
git diff --exit-code --stat -- examples/classifier
# shellcheck disable=SC2046
"$cc" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -O2 $(lib_sources) examples/classifier/classify.c -lm \
  -o "$out/classify"
python3 - "$out/classify" <<'PY'
import json, subprocess, sys
want = json.load(open("examples/classifier/expected.json"))
for case in want["cases"]:
    got = subprocess.run([sys.argv[1], "examples/classifier/classifier.weights",
                          *map(str, case["ids"])], check=True, capture_output=True, text=True)
    p = [float(x) for x in got.stdout.split()[:len(want["labels"])]]
    worst = max(abs(a - b) for a, b in zip(p, case["probabilities"]))
    print(f"  ids {case['ids']}: {p}, within {worst:.1e} of NumPy")
    assert worst < 1e-5, worst
PY
