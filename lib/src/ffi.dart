// Bindings to src/rayo.h. The library is the code asset that hook/build.dart
// compiles; its asset id is this file's.
// ignore_for_file: non_constant_identifier_names
@DefaultAsset('package:rayo/src/ffi.dart')
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

final class RayoModelHandle extends Opaque {}

final class RayoSessionHandle extends Opaque {}

@Native<Pointer<Utf8> Function()>(symbol: 'rayo_version', isLeaf: true)
external Pointer<Utf8> rayo_version();

@Native<Void Function(Pointer<Uint8>, Size, Pointer<Uint8>)>(
  symbol: 'rayo_sha256',
  isLeaf: true,
)
external void rayo_sha256(Pointer<Uint8> data, int size, Pointer<Uint8> out);

@Native<
  Pointer<RayoModelHandle> Function(Pointer<Uint8>, Size, Pointer<Utf8>, Size)
>(symbol: 'rayo_model_load')
external Pointer<RayoModelHandle> rayo_model_load(
  Pointer<Uint8> data,
  int size,
  Pointer<Utf8> err,
  int errLen,
);

@Native<Void Function(Pointer<RayoModelHandle>)>(symbol: 'rayo_model_free')
external void rayo_model_free(Pointer<RayoModelHandle> model);

@Native<Pointer<Utf8> Function(Pointer<RayoModelHandle>)>(
  symbol: 'rayo_model_info',
  isLeaf: true,
)
external Pointer<Utf8> rayo_model_info(Pointer<RayoModelHandle> model);

@Native<
  Pointer<RayoSessionHandle> Function(
    Pointer<RayoModelHandle>,
    Int,
    Pointer<Utf8>,
    Size,
  )
>(symbol: 'rayo_session_new')
external Pointer<RayoSessionHandle> rayo_session_new(
  Pointer<RayoModelHandle> model,
  int threads,
  Pointer<Utf8> err,
  int errLen,
);

@Native<Void Function(Pointer<RayoSessionHandle>)>(symbol: 'rayo_session_free')
external void rayo_session_free(Pointer<RayoSessionHandle> session);

@Native<
  Int Function(
    Pointer<RayoSessionHandle>,
    Pointer<Utf8>,
    Int64,
    Pointer<Utf8>,
    Size,
  )
>(symbol: 'rayo_session_bind', isLeaf: true)
external int rayo_session_bind(
  Pointer<RayoSessionHandle> session,
  Pointer<Utf8> name,
  int value,
  Pointer<Utf8> err,
  int errLen,
);

@Native<
  Int Function(
    Pointer<RayoSessionHandle>,
    Pointer<Utf8>,
    Pointer<Void>,
    Size,
    Pointer<Utf8>,
    Size,
  )
>(symbol: 'rayo_session_set_input', isLeaf: true)
external int rayo_session_set_input(
  Pointer<RayoSessionHandle> session,
  Pointer<Utf8> name,
  Pointer<Void> data,
  int bytes,
  Pointer<Utf8> err,
  int errLen,
);

@Native<Int Function(Pointer<RayoSessionHandle>, Pointer<Utf8>, Size)>(
  symbol: 'rayo_session_run',
)
external int rayo_session_run(
  Pointer<RayoSessionHandle> session,
  Pointer<Utf8> err,
  int errLen,
);

@Native<
  Pointer<Float> Function(
    Pointer<RayoSessionHandle>,
    Pointer<Utf8>,
    Pointer<Int64>,
    Pointer<Utf8>,
    Size,
  )
>(symbol: 'rayo_session_output', isLeaf: true)
external Pointer<Float> rayo_session_output(
  Pointer<RayoSessionHandle> session,
  Pointer<Utf8> name,
  Pointer<Int64> ne,
  Pointer<Utf8> err,
  int errLen,
);

@Native<Void Function(Pointer<RayoSessionHandle>)>(
  symbol: 'rayo_session_reset',
  isLeaf: true,
)
external void rayo_session_reset(Pointer<RayoSessionHandle> session);
