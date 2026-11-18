#!/usr/bin/env bash
# Checks that ELF shared libraries were built hardened (docs/SECURITY.md):
# position-independent; relocations read-only after loading (RELRO) and
# bound at once (BIND_NOW); no text relocations; a stack that is not
# executable; no segment both writable and executable; stack canaries
# (__stack_chk_fail) and the C library's checked copies (_FORTIFY_SOURCE,
# a __*_chk symbol) in use.
#   [READELF=llvm-readelf] tool/ci/elf_check.sh LIBRARY...
set -euo pipefail

readelf=${READELF:-readelf}
status=0
for lib in "$@"; do
  fail=()
  headers=$("$readelf" -hW "$lib")
  segments=$("$readelf" -lW "$lib")
  dynamic=$("$readelf" -dW "$lib")
  symbols=$("$readelf" --dyn-syms -W "$lib")
  grep -Eq 'Type:[[:space:]]+DYN' <<<"$headers" || fail+=("not position-independent")
  grep -q 'GNU_RELRO' <<<"$segments" || fail+=("no RELRO")
  grep -Eq 'BIND_NOW|FLAGS_1.*NOW' <<<"$dynamic" || fail+=("no BIND_NOW")
  if grep -q 'TEXTREL' <<<"$dynamic"; then fail+=("text relocations"); fi
  stack=$(grep 'GNU_STACK' <<<"$segments" || true)
  if [ -z "$stack" ] || grep -Eq ' RWE ' <<<"$stack"; then fail+=("an executable stack"); fi
  # A LOAD segment's flags are its last field but the alignment: R, W, E.
  if awk '$1 == "LOAD" && $(NF - 1) ~ /W/ && $(NF - 1) ~ /E/ { found = 1 } END { exit !found }' \
    <<<"$segments"; then
    fail+=("a writable and executable segment")
  fi
  grep -q '__stack_chk_fail' <<<"$symbols" || fail+=("no stack canaries")
  grep -Eq '__[a-z0-9_]+_chk' <<<"$symbols" || fail+=("no _FORTIFY_SOURCE")
  if [ ${#fail[@]} -eq 0 ]; then
    echo "$lib: hardened (PIC, RELRO, BIND_NOW, no TEXTREL, NX stack, no W+X, canaries, fortify)"
  else
    printf '%s: %s\n' "$lib" "$(IFS=,; echo "${fail[*]}")" >&2
    status=1
  fi
done
exit $status
