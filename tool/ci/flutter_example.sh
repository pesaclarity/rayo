#!/usr/bin/env bash
# examples/flutter_chat: format, flutter analyze, and its tests (widget
# tests, and a tiny test model through rayo, built by the hook). Needs
# flutter on PATH.
#   tool/ci/flutter_example.sh
source "$(dirname "$0")/common.sh"

echo "== Flutter example: $(flutter --version 2>/dev/null | head -1)"
cd examples/flutter_chat
flutter pub get
dart format --output=none --set-exit-if-changed lib test
flutter analyze --fatal-infos
flutter test
