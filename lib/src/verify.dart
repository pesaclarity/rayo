import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings/rayo_bindings.dart' as c;
import 'errors.dart';

/// What a model file must match before rayo reads it (rayo/verify.h,
/// docs/SECURITY.md): its SHA-256, and or its Ed25519 signature.
class RayoIntegrity {
  /// [sha256]: the file's SHA-256, lower- or upper-case hex. [publicKey]
  /// (32 bytes) and [signature] (64 bytes): an Ed25519 signature of the
  /// whole file ("pure" Ed25519, RFC 8032), as tool/sign_model.py or
  /// `openssl pkeyutl -sign -rawin` writes it. Give a digest, a key and a
  /// signature, or all three; every one given is checked.
  RayoIntegrity({this.sha256, this.publicKey, this.signature}) {
    if (sha256 == null && publicKey == null && signature == null) {
      throw ArgumentError(
        'RayoIntegrity needs a sha256, or a key and a signature',
      );
    }
    if ((publicKey == null) != (signature == null)) {
      throw ArgumentError('a publicKey needs a signature, and the reverse');
    }
    if (publicKey != null && publicKey!.length != 32) {
      throw ArgumentError.value(publicKey!.length, 'publicKey', '32 bytes');
    }
    if (signature != null && signature!.length != 64) {
      throw ArgumentError.value(signature!.length, 'signature', '64 bytes');
    }
    if (sha256 != null && !RegExp(r'^[0-9a-fA-F]{64}$').hasMatch(sha256!)) {
      throw ArgumentError.value(sha256, 'sha256', '64 hex digits');
    }
  }

  final String? sha256;
  final Uint8List? publicKey;
  final Uint8List? signature;

  /// The C struct, with its bytes, in [arena].
  Pointer<c.RayoIntegrity> toC(Arena arena) {
    final want = arena<c.RayoIntegrity>();
    want.ref.sha256 = nullptr;
    want.ref.publicKey = nullptr;
    want.ref.signature = nullptr;
    final digest = sha256;
    if (digest != null) {
      final p = arena<Uint8>(32);
      for (var i = 0; i < 32; i++) {
        p[i] = int.parse(digest.substring(2 * i, 2 * i + 2), radix: 16);
      }
      want.ref.sha256 = p;
    }
    if (publicKey != null) want.ref.publicKey = _bytes(arena, publicKey!);
    if (signature != null) want.ref.signature = _bytes(arena, signature!);
    return want;
  }

  static Pointer<Uint8> _bytes(Arena arena, Uint8List b) {
    final p = arena<Uint8>(b.length);
    p.asTypedList(b.length).setAll(0, b);
    return p;
  }
}

/// The `rayo_status` of a failed integrity check (RAYO_ERR_INTEGRITY).
const rayoErrIntegrity = 7;

/// Checks [bytes] against [want]; throws a [RayoException] whose status is
/// [rayoErrIntegrity] when they do not match. To load the bytes that were
/// checked, pass [want] to `RayoModel.load` or `RayoModel.map` instead.
void rayoVerify(Uint8List bytes, RayoIntegrity want) => using((arena) {
  final data = arena<Uint8>(bytes.isEmpty ? 1 : bytes.length);
  data.asTypedList(bytes.length).setAll(0, bytes);
  final msg = arena<c.RayoMessage>();
  check(c.rayo_verify(data.cast(), bytes.length, want.toC(arena), msg), msg);
});

/// Checks the file at [path] against [want], as [rayoVerify].
void rayoVerifyFile(String path, RayoIntegrity want) => using((arena) {
  final msg = arena<c.RayoMessage>();
  check(
    c.rayo_verify_file(
      path.toNativeUtf8(allocator: arena).cast(),
      want.toC(arena),
      msg,
    ),
    msg,
  );
});
