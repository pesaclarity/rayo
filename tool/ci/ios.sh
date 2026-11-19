#!/usr/bin/env bash
# The C library for iOS on macOS (Xcode): strict C89 for the device (arm64)
# and the simulator, a static library of each joined into an xcframework
# (a compile check of what an app links; no package of it is released yet,
# docs/ROADMAP.md), then the C tests built for the simulator and run in one.
#   tool/ci/ios.sh
source "$(dirname "$0")/common.sh"

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
min=13.0
for sdk in iphoneos iphonesimulator; do
  case $sdk in
    iphoneos) target=arm64-apple-ios$min ;;
    iphonesimulator) target=arm64-apple-ios$min-simulator ;;
  esac
  echo "== $sdk ($target): strict C89, the library"
  mkdir -p "$out/$sdk/obj"
  for f in $(lib_sources); do
    xcrun --sdk "$sdk" clang -target "$target" "${C89_FLAGS[@]}" "${HARDEN_FLAGS[@]}" \
      "${INCLUDES[@]}" -O2 -c "$f" -o "$out/$sdk/obj/$(echo "$f" | tr / _).o"
  done
  xcrun --sdk "$sdk" libtool -static -o "$out/$sdk/librayo.a" "$out/$sdk"/obj/*.o
  mkdir -p "$out/$sdk/include"
  cp -R include/rayo "$out/$sdk/include/"
  ls -l "$out/$sdk/librayo.a"
  echo "== $sdk: the quickstart links"
  xcrun --sdk "$sdk" clang -target "$target" "${C89_FLAGS[@]}" -Iinclude -O2 \
    examples/c/quickstart.c "$out/$sdk/librayo.a" -o "$out/$sdk/quickstart"
done

echo "== xcframework"
xcodebuild -create-xcframework \
  -library "$out/iphoneos/librayo.a" -headers "$out/iphoneos/include" \
  -library "$out/iphonesimulator/librayo.a" -headers "$out/iphonesimulator/include" \
  -output "$out/rayo.xcframework"
ls "$out/rayo.xcframework"

echo "== the C tests in a simulator"
sim=arm64-apple-ios$min-simulator
# shellcheck disable=SC2046
xcrun --sdk iphonesimulator clang -target "$sim" "${C89_FLAGS[@]}" "${HARDEN_FLAGS[@]}" \
  "${INCLUDES[@]}" -Itests -O2 $(lib_sources) tests/*.c -lm -o "$out/tests"
udid=$(xcrun simctl list devices available -j | python3 -c '
import json, sys
devices = json.load(sys.stdin)["devices"]
for runtime in sorted(devices, reverse=True):
    if "iOS" in runtime:
        for d in devices[runtime]:
            if d["name"].startswith("iPhone"):
                print(d["udid"]); sys.exit()
')
[ -n "$udid" ] || { echo "ios.sh: no iPhone simulator" >&2; exit 1; }
xcrun simctl boot "$udid" 2>/dev/null || true
xcrun simctl bootstatus "$udid" -b >/dev/null
export SIMCTL_CHILD_RAYO_TEST_MODELS="$RAYO_ROOT/tests/models"
export SIMCTL_CHILD_RAYO_TEST_ASK="$RAYO_ROOT/tests/ask"
export SIMCTL_CHILD_RAYO_TEST_MALFORMED="$RAYO_ROOT/tests/malformed"
for k in generic simd best; do
  echo "-- kernels: $k"
  SIMCTL_CHILD_RAYO_TEST_KERNELS=$k xcrun simctl spawn "$udid" "$out/tests"
done
xcrun simctl shutdown "$udid" || true
echo "ios: ok"
