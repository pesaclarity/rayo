#!/usr/bin/env bash
# The Dart package: format, analysis and the tests (the build hook compiles
# the C library first).
source "$(dirname "$0")/common.sh"

echo "== Dart"
dart pub get
dart format --output=none --set-exit-if-changed lib test hook
dart analyze --fatal-infos
dart test
