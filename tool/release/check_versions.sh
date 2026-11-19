#!/usr/bin/env bash
# Checks that a release tag matches every version in the repository, so the
# tag, the C library, package:rayo, rayo-convert, the changelog, the release
# notes and the README's `ref:` all say the same.
#
#   tool/release/check_versions.sh v0.1.0
set -euo pipefail
cd "$(dirname "$0")/../.."

tag=${1:?usage: check_versions.sh vX.Y.Z}
v=${tag#v}
[[ $tag == v* && $v =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "not a vX.Y.Z tag: $tag" >&2; exit 2; }
IFS=. read -r major minor patch <<<"$v"
fail=0
check() { # what, ok
  if [ "$2" = 1 ]; then echo "  ok: $1"; else echo "  MISMATCH: $1" >&2; fail=1; fi
}
has() { grep -qE "$1" "$2" && echo 1 || echo 0; }

echo "== versions for $tag"
check "include/rayo/version.h is $v" "$(
  grep -q "RAYO_VERSION_MAJOR $major\$" include/rayo/version.h &&
    grep -q "RAYO_VERSION_MINOR $minor\$" include/rayo/version.h &&
    grep -q "RAYO_VERSION_PATCH $patch\$" include/rayo/version.h && echo 1 || echo 0
)"
check "pubspec.yaml is $v" "$(has "^version: $v\$" pubspec.yaml)"
check "pubspec.yaml has no publish_to" "$(grep -q '^publish_to:' pubspec.yaml && echo 0 || echo 1)"
check "convert/pyproject.toml is $v" "$(has "^version = \"$v\"\$" convert/pyproject.toml)"
check "convert/rayo_convert/__init__.py is $v" "$(has "^__version__ = \"$v\"\$" convert/rayo_convert/__init__.py)"
check "CHANGELOG.md has '## $v - YYYY-MM-DD'" "$(has "^## $v - [0-9]{4}-[0-9]{2}-[0-9]{2}\$" CHANGELOG.md)"
check "docs/releases/$tag.md exists" "$([ -f "docs/releases/$tag.md" ] && echo 1 || echo 0)"
# Every git ref the README tells people to depend on is this tag.
refs=$(grep -oE 'ref: v[0-9]+\.[0-9]+\.[0-9]+' README.md docs/*.md | sort -u || true)
bad=$(echo "$refs" | grep -v "ref: $tag\$" | grep -v '^$' || true)
check "every 'ref: v...' in README.md and docs/ is $tag" "$([ -z "$bad" ] && echo 1 || echo 0)"
[ -z "$bad" ] || echo "$bad" >&2
[ "$fail" = 0 ] || { echo "check_versions: $tag does not match the repository" >&2; exit 1; }
echo "check_versions: $tag matches"
