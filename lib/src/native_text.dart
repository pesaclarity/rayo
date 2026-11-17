import 'dart:convert';
import 'dart:ffi';

import 'bindings/rayo_bindings.dart' as c;

/// The NUL-terminated UTF-8 text at [p] ('' for a null pointer).
String cString(Pointer<Uint8> p) {
  if (p == nullptr) return '';
  var n = 0;
  while (p[n] != 0) {
    n++;
  }
  return utf8.decode(p.asTypedList(n));
}

/// A rayo_message's text.
String messageText(Pointer<c.RayoMessage> msg) {
  final bytes = <int>[];
  for (var i = 0; i < 256; i++) {
    final b = msg.ref.text[i];
    if (b == 0) break;
    bytes.add(b);
  }
  return utf8.decode(bytes, allowMalformed: true);
}
