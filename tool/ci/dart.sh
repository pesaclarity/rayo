#!/usr/bin/env bash
# The Dart package: format, analysis and the tests (the build hook compiles
# the C library first).
source "$(dirname "$0")/common.sh"

echo "== Dart"
dart pub get
dart format --output=none --set-exit-if-changed lib test hook example
dart analyze --fatal-infos
dart test
# The library the build hook compiled, built hardened (Linux: an ELF).
if [ "$(uname -s)" = Linux ]; then
  mapfile -t libs < <(find .dart_tool -name 'librayo*.so')
  [ ${#libs[@]} -gt 0 ] || { echo "dart.sh: no librayo.so in .dart_tool" >&2; exit 1; }
  tool/ci/elf_check.sh "${libs[@]}"
fi
