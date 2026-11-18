#!/usr/bin/env bash
# Static analysis of the library, which must find nothing: the clang static
# analyzer (clang --analyze, as scan-build runs it) on every C file, with the
# SIMD kernels' instruction sets on and with the plain C89 platform; and
# cppcheck's warning, portability and performance checks. A finding that is
# not a bug is suppressed where it is, with the reason (docs/SECURITY.md).
#   tool/ci/analyze.sh
source "$(dirname "$0")/common.sh"

status=0
for flags in "" "-DRAYO_NO_PLATFORM" "-mavx2 -mfma"; do
  if [ "$flags" = "-mavx2 -mfma" ] && [ "$(uname -m)" != x86_64 ]; then continue; fi
  echo "== clang --analyze ${flags:-(defaults)}"
  for f in $(lib_sources); do
    # shellcheck disable=SC2086
    if ! clang --analyze "${C89_FLAGS[@]}" "${INCLUDES[@]}" $flags -Xclang -analyzer-werror \
      -Xclang -analyzer-output=text -o /dev/null "$f"; then
      status=1
    fi
  done
done
if command -v cppcheck >/dev/null; then
  for defs in "-D__unix__ -D__linux__ -D__GNUC__" "-DRAYO_NO_PLATFORM"; do
    echo "== cppcheck $(cppcheck --version) $defs"
    # shellcheck disable=SC2086
    if ! cppcheck --std=c89 --enable=warning,portability,performance --inline-suppr \
      --error-exitcode=1 --quiet --suppress=missingIncludeSystem "${INCLUDES[@]}" $defs src; then
      status=1
    fi
  done
else
  echo "analyze.sh: cppcheck is not installed" >&2
  status=1
fi
exit $status
