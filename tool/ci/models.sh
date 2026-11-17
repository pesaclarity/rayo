#!/usr/bin/env bash
# The test models are exactly what tool/make_test_models.py writes.
source "$(dirname "$0")/common.sh"

echo "== test models"
python3 tool/make_test_models.py >/dev/null
git diff --exit-code --stat -- tests/models
