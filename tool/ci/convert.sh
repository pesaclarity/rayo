#!/usr/bin/env bash
# rayo-convert (convert/): its tests, then the released Ask reader and writer
# converted from their TFLite files (tests/ask/*.tflite) and run by the C tests
# (every reader label, the writer's next token at all 2243 steps) and timed by
# the writer's bench, which decodes with the key/value cache the conversion
# writes.
#   tool/ci/convert.sh [CC] [flags...]
source "$(dirname "$0")/common.sh"

cc=${1:-cc}
shift || true
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
py=${PYTHON:-python3}
if ! "$py" -c "import tflite, numpy, pytest, onnx, jinja2" 2>/dev/null; then
  echo "== a venv with convert/requirements-ci.txt"
  if command -v uv >/dev/null; then
    uv venv -q "$out/venv"
    VIRTUAL_ENV="$out/venv" uv pip install -q -r convert/requirements-ci.txt
  else
    "$py" -m venv "$out/venv"
    "$out/venv/bin/python" -m pip install -q -r convert/requirements-ci.txt
  fi
  py="$out/venv/bin/python"
fi
echo "== rayo-convert tests ($cc)"
(cd convert && CC="$cc" PYTHONPATH=. "$py" -m pytest -q -p no:cacheprovider)
echo "== the Ask models, converted from TFLite"
mkdir -p "$out/ask"
cp tests/ask/*.json "$out/ask/"
convert=("$py" -m rayo_convert.cli)
PYTHONPATH=convert "${convert[@]}" tests/ask/ask-reader.tflite -o "$out/ask/ask-reader.weights" \
  --name ask-reader --split-input question=tokens:96,shapes:96 --outputs slots,intent
PYTHONPATH=convert "${convert[@]}" tests/ask/ask-writer.tflite -o "$out/ask/ask-writer.weights" \
  --name ask-writer --rename output_0=logits
build_tests "$cc" "$out/tests" -O2 "$@"
RAYO_TEST_ASK="$out/ask" "$out/tests" ask
build_tool "$cc" examples/ask/ask_writer_bench.c "$out/bench" -O3 "$@"
"$out/bench" "$out/ask/ask-writer.weights" tests/ask/ask_writer_decodes.json 3 | head -3
