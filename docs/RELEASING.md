# Releasing rayo

How a maintainer makes a release: the checks, the signed tag, the GitHub release with its
assets, and `package:rayo` on pub.dev. The signing key and the pub.dev account stay on the
maintainer's machine; no workflow holds them.

The steps below are for **v0.1.0**; for a later version, change the numbers.

## 1. Before the tag

- [ ] `LICENSE-EXCEPTION` has been reviewed by a lawyer, and its "DRAFT" box removed (with
      the matching notes in `docs/LICENSING.md` and the README).
- [ ] The release pull request is merged into `main`, and CI is green on `main`.
- [ ] The versions agree: `tool/release/check_versions.sh v0.1.0` passes. It checks
      `include/rayo/version.h`, `pubspec.yaml` (and that it has no `publish_to`),
      `convert/pyproject.toml`, `convert/rayo_convert/__init__.py`, the `## 0.1.0 - DATE`
      heading of `CHANGELOG.md`, these release notes (`docs/releases/v0.1.0.md`), and that every
      `ref: v...` in the README and docs is `ref: v0.1.0`.
- [ ] `tool/check` passes on your computer.

```sh
git switch main && git pull --ff-only
tool/release/check_versions.sh v0.1.0
tool/check
```

## 2. The signed tag

```sh
git tag -s v0.1.0 -m "rayo 0.1.0"
git tag -v v0.1.0
git push origin v0.1.0
```

The tag is what `ref: v0.1.0` in the README's git dependency and in the Flutter docs resolves
to, so it must be exactly `v0.1.0`, on the merge commit of the release pull request.

Pushing it starts the **Release assets** workflow (`.github/workflows/release.yml`). It
checks the tag against the versions again, runs the C tests and the examples, converts
SmolLM2-135M-Instruct to `smollm2-135m-instruct-q4.weights` with rayo-convert, writes its
SHA-256, licence (Apache-2.0) and notice, and keeps all of them as the run's
`release-assets` artifact. It does not create or edit the release.

## 3. The GitHub release

Wait for the workflow, download its assets, check them, and create the release with them:

```sh
run=$(gh run list -R pesaclarity/rayo --workflow release.yml --branch v0.1.0 \
  --json databaseId --jq '.[0].databaseId')
gh run watch "$run" -R pesaclarity/rayo --exit-status
rm -rf dist && gh run download "$run" -R pesaclarity/rayo -n release-assets -D dist
(cd dist && sha256sum -c SHA256SUMS && sha256sum -c smollm2-135m-instruct-q4.weights.sha256)

gh release create v0.1.0 -R pesaclarity/rayo --verify-tag \
  --title "rayo 0.1.0" --notes-file docs/releases/v0.1.0.md \
  dist/smollm2-135m-instruct-q4.weights \
  dist/smollm2-135m-instruct-q4.weights.sha256 \
  dist/smollm2-135m-instruct-q4.LICENSE.txt \
  dist/smollm2-135m-instruct-q4.NOTICE.md \
  dist/SHA256SUMS
```

GitHub adds the source archives itself. If the workflow cannot reach Hugging Face, build the
model on your computer instead (the same steps; the result is the same bytes, since the
conversion is deterministic) and attach it the same way:

```sh
python3 -m venv .venv && . .venv/bin/activate && pip install "./convert[chat]"
tool/release/smollm2.sh dist      # add --revision COMMIT to pin a Hugging Face commit
(cd dist && sha256sum -- *.weights *.LICENSE.txt *.NOTICE.md > SHA256SUMS)
```

Then put the model's size and SHA-256 (from `dist/smollm2-135m-instruct-q4.weights.sha256`)
into the release's description, under "Prebuilt model".

## 4. pub.dev

`package:rayo` is published from your computer, with your pub.dev account. The first time,
log in (`dart pub login`); then, from a clean checkout of the tag:

```sh
git switch --detach v0.1.0
git status --short                 # must print nothing
dart pub get
dart pub publish --dry-run         # must report no warnings
dart pub publish
```

`--dry-run` lists every file the package holds (the Dart library, the C sources and headers
its hook compiles, the licences and `example/`; `.pubignore` leaves out the rest) and must end
with "Package has 0 warnings."

After the first publish, on pub.dev's admin page for the package:

- add it to the `pesaclarity` verified publisher, if there is one (Admin, Transfer to
  publisher);
- optionally turn on automated publishing from GitHub Actions for later releases (Admin,
  Automated publishing, with a `v{{version}}` tag pattern), which needs a workflow with
  `id-token: write` and no stored secret.

Within a few minutes pub.dev scores the package. The homepage and documentation links point
at `https://pesaclarity.github.io/docs/rayo/`, which must be live, and the repository check
clones `main`, whose `pubspec.yaml` must have no `publish_to`.

## 5. After the release

- In `CHANGELOG.md`, keep the empty `## Unreleased` heading on top for the next changes.
- On the website, sync the docs again so they show the release (`npm run sync -- --update`
  in pesaclarity.github.io).
- Phone numbers: run `tool/bench_phone.sh` on a 2 GB Android Go phone
  ([Performance](PERFORMANCE.md#measuring-an-android-phone)) and replace the README's
  "Phone results: coming" row.
