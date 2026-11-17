import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings/rayo_bindings.dart' as c;
import 'native_text.dart';

/// Whether the native library loads here (it does on every target the
/// build hook compiles for). Never throws, so an app can fall back.
bool rayoAvailable() {
  try {
    c.rayo_version_string();
    return true;
  } on Object {
    return false;
  }
}

/// The library's version, such as `0.1.0`.
String rayoVersion() => cString(c.rayo_version_string());

/// Lower-case hex SHA-256 of [bytes], computed natively.
String rayoSha256(Uint8List bytes) => using((arena) {
  final data = arena<Uint8>(bytes.isEmpty ? 1 : bytes.length);
  data.asTypedList(bytes.length).setAll(0, bytes);
  final out = arena<Uint8>(32);
  c.rayo_sha256(data, bytes.length, out);
  return [
    for (final b in out.asTypedList(32)) b.toRadixString(16).padLeft(2, '0'),
  ].join();
});
