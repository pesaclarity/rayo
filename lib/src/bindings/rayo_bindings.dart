// Raw bindings to include/rayo/*.h. The library is the code asset that
// hook/build.dart compiles; its asset id is this file's.
// ignore_for_file: non_constant_identifier_names, public_member_api_docs
@DefaultAsset('package:rayo/src/bindings/rayo_bindings.dart')
library;

import 'dart:ffi';

final class RayoModelHandle extends Opaque {}

final class RayoSessionHandle extends Opaque {}

/// `rayo_message`.
final class RayoMessage extends Struct {
  @Array(256)
  external Array<Uint8> text;
}

/// `rayo_shape`.
final class RayoShape extends Struct {
  @Int32()
  external int rank;

  @Array(4)
  external Array<Int64> dims;
}

/// `rayo_tensor_view`.
final class RayoTensorView extends Struct {
  external Pointer<Uint8> name;

  @Int32()
  external int dtype;

  external RayoShape shape;

  external Pointer<Void> data;

  @Size()
  external int bytes;
}

/// `rayo_memory_info`.
final class RayoMemoryInfo extends Struct {
  @Size()
  external int weights;

  @Size()
  external int arena;

  @Size()
  external int overhead;
}

/// `rayo_dtype`.
const rayoDtypeF32 = 0;
const rayoDtypeI32 = 1;
const rayoDtypeI8 = 2;

@Native<Pointer<Uint8> Function()>(isLeaf: true)
external Pointer<Uint8> rayo_version_string();

@Native<Pointer<Uint8> Function(Int32)>(isLeaf: true)
external Pointer<Uint8> rayo_status_str(int status);

@Native<Void Function(Pointer<Uint8>, Size, Pointer<Uint8>)>(isLeaf: true)
external void rayo_sha256(Pointer<Uint8> data, int size, Pointer<Uint8> out);

@Native<
  Int32 Function(
    Pointer<Void>,
    Size,
    Pointer<Pointer<RayoModelHandle>>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_model_create(
  Pointer<Void> data,
  int size,
  Pointer<Pointer<RayoModelHandle>> out,
  Pointer<RayoMessage> msg,
);

@Native<Void Function(Pointer<RayoModelHandle>)>()
external void rayo_model_destroy(Pointer<RayoModelHandle> model);

@Native<Pointer<Uint8> Function(Pointer<RayoModelHandle>)>(isLeaf: true)
external Pointer<Uint8> rayo_model_name(Pointer<RayoModelHandle> model);

@Native<Pointer<Uint8> Function(Pointer<RayoModelHandle>)>(isLeaf: true)
external Pointer<Uint8> rayo_model_metadata_json(
  Pointer<RayoModelHandle> model,
);

@Native<Int32 Function(Pointer<RayoModelHandle>)>(isLeaf: true)
external int rayo_model_input_count(Pointer<RayoModelHandle> model);

@Native<Int32 Function(Pointer<RayoModelHandle>)>(isLeaf: true)
external int rayo_model_output_count(Pointer<RayoModelHandle> model);

@Native<
  Int32 Function(Pointer<RayoModelHandle>, Int32, Pointer<RayoTensorView>)
>(isLeaf: true)
external int rayo_model_input_info(
  Pointer<RayoModelHandle> model,
  int index,
  Pointer<RayoTensorView> out,
);

@Native<
  Int32 Function(Pointer<RayoModelHandle>, Int32, Pointer<RayoTensorView>)
>(isLeaf: true)
external int rayo_model_output_info(
  Pointer<RayoModelHandle> model,
  int index,
  Pointer<RayoTensorView> out,
);

@Native<Void Function(Pointer<RayoModelHandle>, Pointer<RayoMemoryInfo>)>(
  isLeaf: true,
)
external void rayo_model_memory(
  Pointer<RayoModelHandle> model,
  Pointer<RayoMemoryInfo> out,
);

@Native<
  Int32 Function(
    Pointer<RayoModelHandle>,
    Pointer<Pointer<RayoSessionHandle>>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_session_create(
  Pointer<RayoModelHandle> model,
  Pointer<Pointer<RayoSessionHandle>> out,
  Pointer<RayoMessage> msg,
);

@Native<Void Function(Pointer<RayoSessionHandle>)>()
external void rayo_session_destroy(Pointer<RayoSessionHandle> session);

@Native<
  Int32 Function(
    Pointer<RayoSessionHandle>,
    Pointer<Uint8>,
    Int64,
    Pointer<RayoMessage>,
  )
>(isLeaf: true)
external int rayo_session_bind(
  Pointer<RayoSessionHandle> session,
  Pointer<Uint8> name,
  int value,
  Pointer<RayoMessage> msg,
);

@Native<
  Int32 Function(Pointer<RayoSessionHandle>, Int32, Pointer<RayoTensorView>)
>(isLeaf: true)
external int rayo_session_input(
  Pointer<RayoSessionHandle> session,
  int index,
  Pointer<RayoTensorView> out,
);

@Native<Int32 Function(Pointer<RayoSessionHandle>, Pointer<RayoMessage>)>()
external int rayo_session_run(
  Pointer<RayoSessionHandle> session,
  Pointer<RayoMessage> msg,
);

@Native<
  Int32 Function(Pointer<RayoSessionHandle>, Int32, Pointer<RayoTensorView>)
>(isLeaf: true)
external int rayo_session_output(
  Pointer<RayoSessionHandle> session,
  int index,
  Pointer<RayoTensorView> out,
);

@Native<Void Function(Pointer<RayoSessionHandle>)>(isLeaf: true)
external void rayo_session_reset(Pointer<RayoSessionHandle> session);

@Native<Pointer<Uint8> Function(Pointer<RayoSessionHandle>)>(isLeaf: true)
external Pointer<Uint8> rayo_session_kernels(
  Pointer<RayoSessionHandle> session,
);

@Native<Void Function(Pointer<RayoSessionHandle>, Pointer<RayoMemoryInfo>)>(
  isLeaf: true,
)
external void rayo_session_memory(
  Pointer<RayoSessionHandle> session,
  Pointer<RayoMemoryInfo> out,
);
