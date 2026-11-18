#!/usr/bin/env bash
# Adds a run's benchmark results to the `bench` branch, never to main: the run's JSON in
# results/, history.csv, README.md and the table (table.html, table.svg, and the PNGs
# tool/ci/bench_render.cjs draws), as one commit. Makes the branch, with no history of
# its own, the first time. Tries again from the branch's new tip if another run pushed
# first.
#   tool/ci/bench_publish.sh INCOMING_DIR WORK_DIR
# Environment: GH_TOKEN (contents: write), GITHUB_REPOSITORY, GITHUB_SHA, and optionally
# BENCH_BRANCH (default bench), RUN_URL, NODE_PATH for Playwright, and BENCH_REMOTE (a
# repository to push to instead of GitHub's, to try it).
source "$(dirname "$0")/common.sh"

incoming=$(cd "${1:?usage: bench_publish.sh INCOMING_DIR WORK_DIR}" && pwd)
work=${2:?usage: bench_publish.sh INCOMING_DIR WORK_DIR}
branch=${BENCH_BRANCH:-bench}
repo=${GITHUB_REPOSITORY:?}
sha=${GITHUB_SHA:?}
url=${BENCH_REMOTE:-"https://x-access-token:${GH_TOKEN:?}@github.com/${repo}.git"}
date=$(date -u +%Y-%m-%dT%H:%M:%SZ)

for attempt in 1 2 3 4; do
  rm -rf "$work"
  mkdir -p "$work"
  git -C "$work" init -q
  git -C "$work" remote add origin "$url"
  if git -C "$work" fetch -q --depth 1 origin "refs/heads/$branch" 2>/dev/null; then
    git -C "$work" checkout -q -B "$branch" FETCH_HEAD
  else
    echo "no $branch branch yet: making it"
    git -C "$work" checkout -q --orphan "$branch"
  fi
  python3 tool/ci/bench_results.py publish --incoming "$incoming" --out "$work" \
    --commit "$sha" --date "$date" --repo "$repo" ${RUN_URL:+--run-url "$RUN_URL"}
  node tool/ci/bench_render.cjs "$work" || echo "::warning::the PNG tables were not drawn"
  git -C "$work" add -A
  if git -C "$work" diff --cached --quiet; then
    echo "nothing changed"
    exit 0
  fi
  git -C "$work" -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit -q -m "bench: ${sha:0:7}" -m "Results of $repo@$sha${RUN_URL:+ ($RUN_URL)}."
  if git -C "$work" push -q origin "HEAD:refs/heads/$branch"; then
    echo "pushed to $branch"
    exit 0
  fi
  echo "push failed (attempt $attempt): trying again from the branch's tip"
  sleep $((attempt * 5))
done
exit 1
