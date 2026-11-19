#!/usr/bin/env bash
# The release's prebuilt model: SmolLM2-135M-Instruct (Hugging Face,
# Apache-2.0) converted by rayo-convert to 4-bit weights with float16
# scales, with its tokeniser's table and chat template in the file. Writes
# to OUT:
#
#   smollm2-135m-instruct-q4.weights          the model
#   smollm2-135m-instruct-q4.weights.sha256   its SHA-256 (sha256sum -c)
#   smollm2-135m-instruct-q4.LICENSE.txt      the model's licence, Apache-2.0
#   smollm2-135m-instruct-q4.NOTICE.md        attribution, source, changes
#
# then runs it once with the chat program (greedy) as a smoke test.
#
#   tool/release/smollm2.sh OUT [--revision REV] [--from DIR] [--context N]
#
# Needs rayo-convert with its chat extra (pip install "./convert[chat]"),
# curl, a C compiler and make. --revision pins the Hugging Face commit (the
# default, main, is resolved and recorded in the notice); --from converts a
# directory that already holds the files (no download).
set -euo pipefail
cd "$(dirname "$0")/../.."

repo=HuggingFaceTB/SmolLM2-135M-Instruct
out=${1:?usage: smollm2.sh OUT [--revision REV] [--from DIR] [--context N]}
shift
revision=main from="" context=2048
while [ $# -gt 0 ]; do
  case $1 in
    --revision) revision=$2; shift 2 ;;
    --from) from=$2; shift 2 ;;
    --context) context=$2; shift 2 ;;
    *) echo "smollm2.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
name=smollm2-135m-instruct-q4
mkdir -p "$out"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

if [ -n "$from" ]; then
  src=$from
  commit="(converted from a local directory: $from)"
else
  src=$work/src
  mkdir -p "$src"
  echo "== download $repo at $revision"
  # The commit `revision` names, from the resolve endpoint's header, so the
  # notice records exactly what was converted.
  commit=$(curl -fsSI "https://huggingface.co/$repo/resolve/$revision/config.json" |
    tr -d '\r' | awk -F': ' 'tolower($1) == "x-repo-commit" {print $2}')
  [ -n "$commit" ] || { echo "smollm2.sh: no commit for $repo@$revision" >&2; exit 1; }
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
    model.safetensors; do
    curl -fsSL --retry 3 -o "$src/$f" "https://huggingface.co/$repo/resolve/$commit/$f"
  done
fi

convert_version=$(rayo-convert --version | awk '{print $2}')
echo "== convert with rayo-convert $convert_version"
meta=$(printf '{"source": "https://huggingface.co/%s", "source_commit": "%s", "licence": "Apache-2.0", "converted_by": "rayo-convert %s"}' \
  "$repo" "$commit" "$convert_version")
rayo-convert "$src" -o "$work/$name.weights" --weights q4 --scales float16 \
  --context "$context" --name "$name" --metadata "$meta"
mv "$work/$name.weights" "$out/$name.weights"
(cd "$out" && sha256sum "$name.weights" >"$name.weights.sha256")
sha=$(cut -d' ' -f1 "$out/$name.weights.sha256")
bytes=$(wc -c <"$out/$name.weights" | tr -d ' ')
cp tool/release/smollm2/LICENSE-Apache-2.0.txt "$out/$name.LICENSE.txt"
cat >"$out/$name.NOTICE.md" <<NOTICE
# $name.weights

**SmolLM2-135M-Instruct**, by Hugging Face (HuggingFaceTB), converted to
rayo's format.

- Source: https://huggingface.co/$repo
- Source commit: $commit
- Licence: Apache License 2.0 (\`$name.LICENSE.txt\`). The model's licence,
  not rayo's: the weights are Hugging Face's work, and rayo's AGPL-3.0 and
  its linking exception do not apply to them.
- Changes (Apache-2.0, section 4 b): converted by rayo-convert
  $convert_version with \`--weights q4 --scales float16 --context $context\`:
  the matrices quantised to 4-bit groups of 32 with float16 scales, the
  graph rewritten for rayo, the tokeniser's table and chat template written
  into the file. The outputs differ from the original float weights' by
  the quantisation.
- File: $bytes bytes, SHA-256 \`$sha\`.

Citation, as the model card asks:

> Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, et al. *SmolLM2: When Smol
> Goes Big — Data-Centric Training of a Small Language Model.* 2025.
> https://arxiv.org/abs/2502.02737
NOTICE

echo "== smoke test: one greedy reply"
make -s examples/chat/chat
printf 'What is a budget?\n/quit\n' |
  examples/chat/chat "$out/$name.weights" --temp 0 --max-tokens 24 2>"$work/log.txt" |
  tee "$work/reply.txt"
cat "$work/log.txt"
grep -q "reply: .* tokens/s" "$work/log.txt"
echo "smollm2.sh: $out/$name.weights, $bytes bytes, sha256 $sha"
