#!/usr/bin/env bash
# The test models are exactly what tool/make_test_models.py writes. Their
# expected outputs must match to 1e-12: NumPy's float sums can round
# differently on another CPU (its SIMD paths), so they are not compared
# byte for byte.
source "$(dirname "$0")/common.sh"

echo "== test models"
before="$(mktemp)"
trap 'rm -f "$before"' EXIT
cp tests/models/expected.json "$before"
python3 tool/make_test_models.py >/dev/null
git diff --exit-code --stat -- tests/models ':!tests/models/expected.json'
python3 - "$before" tests/models/expected.json <<'PY'
import json, sys
def walk(a, b, at="$"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), at
        for k in a: walk(a[k], b[k], f"{at}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), at
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{at}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= 1e-12 * max(1.0, abs(a)), f"{at}: {a} != {b}"
    else:
        assert a == b, f"{at}: {a} != {b}"
walk(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
PY
cp "$before" tests/models/expected.json
