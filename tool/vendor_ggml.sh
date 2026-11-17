#!/usr/bin/env bash
# Copies the CPU part of ggml (MIT) at a pinned commit into third_party/ggml,
# and records the commit and every file's SHA-256 in third_party/ggml/PIN and
# third_party/ggml/SHA256SUMS.
#
# The build hook compiles only these vendored files, so a build needs no
# network: F-Droid and every other build compile exactly what is checked in.
# Run this only to move to another ggml commit, then review the diff.
#
# Usage: tool/vendor_ggml.sh [ggml checkout]
#   Without an argument it clones the pinned commit from GitHub.
set -euo pipefail

repo_url="https://github.com/ggml-org/ggml.git"
tag="v1.0.0"
commit="417a024b4fb6bc3116d41af7c067166a039bdf12"

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/third_party/ggml"

if [ $# -ge 1 ]; then
  src="$1"
else
  src="$(mktemp -d)"
  trap 'rm -rf "$src"' EXIT
  git clone -q --depth 1 --branch "$tag" "$repo_url" "$src"
fi
got="$(git -C "$src" rev-parse HEAD)"
if [ "$got" != "$commit" ]; then
  echo "ggml checkout is at $got, expected $commit" >&2
  exit 1
fi

rm -rf "$dest"
mkdir -p "$dest/include" "$dest/src/ggml-cpu/arch"

# Public headers the CPU build needs.
for h in ggml.h ggml-alloc.h ggml-backend.h ggml-cpp.h ggml-cpu.h ggml-opt.h gguf.h; do
  cp "$src/include/$h" "$dest/include/"
done
# ggml-base: the tensor library, graph allocator, backend glue, quantisation
# and GGUF. No GPU backends.
for f in ggml.c ggml.cpp ggml-alloc.c ggml-backend.cpp ggml-backend-meta.cpp \
  ggml-backend-impl.h ggml-common.h ggml-impl.h ggml-opt.cpp ggml-quants.c \
  ggml-quants.h ggml-threading.cpp ggml-threading.h gguf.cpp; do
  cp "$src/src/$f" "$dest/src/"
done
# ggml-cpu: the CPU kernels, with the ARM (NEON) and x86 (SSE/AVX) variants.
# Left out: AMX, KleidiAI and SpacemiT, which we do not build.
cp "$src"/src/ggml-cpu/*.c "$src"/src/ggml-cpu/*.cpp "$src"/src/ggml-cpu/*.h \
  "$dest/src/ggml-cpu/"
rm -f "$dest/src/ggml-cpu/hbm.cpp" "$dest/src/ggml-cpu/hbm.h"
cp -r "$src/src/ggml-cpu/tiled" "$dest/src/ggml-cpu/"
for a in arm x86; do
  mkdir -p "$dest/src/ggml-cpu/arch/$a"
  cp "$src/src/ggml-cpu/arch/$a"/*.c "$src/src/ggml-cpu/arch/$a"/*.cpp \
    "$dest/src/ggml-cpu/arch/$a/"
done
# The llamafile sgemm (MIT, Mozilla): fast float32/float16 matrix products.
mkdir -p "$dest/src/ggml-cpu/llamafile"
cp "$src"/src/ggml-cpu/llamafile/sgemm.cpp "$src"/src/ggml-cpu/llamafile/sgemm.h \
  "$dest/src/ggml-cpu/llamafile/"
# ggml-cpu includes this header by path; the AMX sources themselves are not built.
mkdir -p "$dest/src/ggml-cpu/amx"
cp "$src/src/ggml-cpu/amx/amx.h" "$dest/src/ggml-cpu/amx/"
cp "$src/LICENSE" "$dest/LICENSE"
# What CMake would generate from ggml-version.h.in.
printf '#pragma once\n\n#define GGML_VERSION "%s"\n#define GGML_COMMIT  "%s"\n' \
  "${tag#v}" "${commit:0:7}" >"$dest/src/ggml-version.h"

cat >"$dest/PIN" <<EOF
ggml (MIT), the CPU part only, vendored by tool/vendor_ggml.sh.
repository: $repo_url
tag: $tag
commit: $commit
EOF
(cd "$dest" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z |
  xargs -0 sha256sum >SHA256SUMS)
echo "Vendored ggml $tag ($commit) into third_party/ggml: $(du -sh "$dest" | cut -f1)."
