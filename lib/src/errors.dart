import 'dart:ffi';

import 'bindings/rayo_bindings.dart' as c;
import 'native_text.dart';

/// A rayo call that failed: a damaged model, a size that does not fit, a
/// closed object.
class RayoException implements Exception {
  RayoException(this.status, this.message);

  /// The C library's `rayo_status`, 0 for errors raised in Dart.
  final int status;
  final String message;

  @override
  String toString() => 'RayoException: $message';
}

/// Throws when [status] is not RAYO_OK, with the message in [msg].
void check(int status, [Pointer<c.RayoMessage>? msg]) {
  if (status == 0) return;
  final kind = cString(c.rayo_status_str(status));
  final detail = msg == null ? '' : messageText(msg);
  throw RayoException(status, detail.isEmpty ? kind : '$kind: $detail');
}
