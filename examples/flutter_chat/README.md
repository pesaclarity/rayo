# rayo_flutter_chat

A minimal Flutter app on `package:rayo`: it maps a language model from a
file, runs one prompt in a background isolate and shows the reply. The
runner is `lib/runner.dart`; the page is `lib/main.dart`.

The repository keeps only `lib/`, `test/` and `pubspec.yaml`. Make the
platform folders once, then run it:

```sh
cd examples/flutter_chat
flutter create --platforms=android,ios,linux,macos .
flutter run --dart-define=RAYO_MODEL=/path/to/smollm2.weights
```

rayo's build hook compiles the C library for the target while the app
builds (the NDK on Android, Xcode on iOS and macOS, clang or gcc on Linux):
nothing is downloaded.

The model is a file on the device: convert one with rayo-convert
(`docs/GETTING-STARTED.md`, or the `smollm2-135m-q4.weights` asset of a
rayo release), copy it into the app's files (`adb push` to
`/data/local/tmp/` for a quick try, or download it on first start) and
type its path in the first field.

```sh
flutter analyze
flutter test      # widget tests, and a tiny test model through rayo itself
```
