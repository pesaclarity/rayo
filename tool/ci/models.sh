#!/usr/bin/env bash
# The test models are exactly what tool/make_test_models.py and
# tool/make_llama_test_model.py write. Their expected outputs must match to
# 1e-12 (1e-5 for the language model's logits, which are rounded to six
# places): NumPy's float sums can round differently on another CPU (its SIMD
# paths), so they are not compared byte for byte.
source "$(dirname "$0")/common.sh"

echo "== test models"
before="$(mktemp -d)"
trap 'rm -rf "$before"' EXIT
cp tests/models/expected.json tests/models/llama_expected.json "$before/"
python3 tool/make_test_models.py >/dev/null
python3 tool/make_llama_test_model.py >/dev/null
git diff --exit-code --stat -- tests/models ':!tests/models/expected.json' \
  ':!tests/models/llama_expected.json'
python3 - "$before" <<'PY'
import json, sys
def walk(a, b, tol, at="$"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), at
        for k in a: walk(a[k], b[k], tol, f"{at}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), at
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, tol, f"{at}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= tol * max(1.0, abs(a)), f"{at}: {a} != {b}"
    else:
        assert a == b, f"{at}: {a} != {b}"
for name, tol in (("expected.json", 1e-12), ("llama_expected.json", 1e-5)):
    walk(json.load(open(f"{sys.argv[1]}/{name}")), json.load(open(f"tests/models/{name}")), tol)
PY
cp "$before/expected.json" "$before/llama_expected.json" tests/models/
