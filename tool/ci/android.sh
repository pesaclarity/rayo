#!/usr/bin/env bash
# The library for each Android ABI with the NDK (ANDROID_NDK_HOME), strict
# C89 at -Werror, as the build hook builds it; fails if a stripped .so is
# over BUDGET_KB (default 200: 150 until the language-model path, whose
# tokeniser, sampler, generator, Unicode tables and kernels add about 60 KB).
# The chat example is built for the phones' ABIs too, as
# docs/RUNNING-SLMS.md does. With SUMMARY set, a Markdown table too; with
# SIZES_JSON set, the sizes as JSON (for the benchmark table).
source "$(dirname "$0")/common.sh"

ndk="${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME}"
bin="$(echo "$ndk"/toolchains/llvm/prebuilt/*/bin)"
budget=${BUDGET_KB:-200}
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
printf '{\n  global:\n    rayo_*;\n  local:\n    *;\n};\n' >"$out/rayo.map"
rows=()
sizes=()
status=0
for abi in "aarch64-linux-android24:arm64-v8a:" "armv7a-linux-androideabi24:armeabi-v7a:-mfpu=neon" \
  "x86_64-linux-android24:x86_64:-msse4.2"; do
  IFS=: read -r target name flags <<<"$abi"
  # shellcheck disable=SC2046,SC2086
  "$bin/clang" --target="$target" "${C89_FLAGS[@]}" "${INCLUDES[@]}" $flags -O3 -fPIC -shared \
    -fvisibility=hidden -ffunction-sections -fdata-sections $(lib_sources) -lm \
    -Wl,--gc-sections -Wl,--no-undefined -Wl,--version-script="$out/rayo.map" \
    -Wl,-z,max-page-size=16384 -o "$out/librayo-$name.so"
  "$bin/llvm-strip" --strip-unneeded "$out/librayo-$name.so"
  size=$(wc -c <"$out/librayo-$name.so")
  kb=$(( (size + 1023) / 1024 ))
  verdict=ok
  if [ "$kb" -gt "$budget" ]; then verdict="over $budget KB"; status=1; fi
  echo "$name: $size bytes ($kb KB) $verdict"
  rows+=("| $name | $size | $kb | $verdict |")
  sizes+=("$name=$size")
  # The tests build for the ABI too (they run on the arm64 and armv7 jobs).
  # shellcheck disable=SC2046,SC2086
  "$bin/clang" --target="$target" "${C89_FLAGS[@]}" "${INCLUDES[@]}" -Itests $flags -O2 \
    $(lib_sources) tests/*.c -lm -o "$out/tests-$name"
  if [ "$name" != x86_64 ]; then
    make -s clean
    make -s CC="$bin/clang --target=$target" CFLAGS="-O2 $flags" LIBS=-lm
    echo "$name: chat example $(wc -c <examples/chat/chat) bytes"
    make -s clean
  fi
done
if [ -n "${SUMMARY:-}" ]; then
  {
    echo "### Android library sizes (stripped, budget $budget KB)"
    echo
    echo "| ABI | bytes | KB | |"
    echo "| --- | --- | --- | --- |"
    printf '%s\n' "${rows[@]}"
  } >>"$SUMMARY"
fi
if [ -n "${SIZES_JSON:-}" ]; then
  python3 tool/ci/bench_results.py sizes "$SIZES_JSON" "${sizes[@]}"
fi
exit $status
