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
  external Array<Int32> dims;
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

  @Size()
  external int states;
}

/// `rayo_dtype`.
const rayoDtypeF32 = 0;
const rayoDtypeI32 = 1;
const rayoDtypeI8 = 2;

@Native<Pointer<Uint8> Function()>(isLeaf: true)
external Pointer<Uint8> rayo_version_string();

@Native<Pointer<Uint8> Function(Int32)>(isLeaf: true)
external Pointer<Uint8> rayo_status_str(int status);

@Native<Pointer<Uint8> Function(Int32)>(isLeaf: true)
external Pointer<Uint8> rayo_op_builtin(int index);

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
    Int32,
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

// Language models (rayo/generate.h, rayo/text.h, rayo_model_load).

final class RayoTokenizerHandle extends Opaque {}

final class RayoGeneratorHandle extends Opaque {}

/// `rayo_model_options`.
final class RayoModelOptions extends Struct {
  external Pointer<Void> ops;

  @Int()
  external int opCount;

  @Int()
  external int readFile;

  @Int()
  external int access;

  external Pointer<RayoIntegrity> integrity;
}

/// `rayo_integrity` (rayo/verify.h).
final class RayoIntegrity extends Struct {
  external Pointer<Uint8> sha256;

  external Pointer<Uint8> publicKey;

  external Pointer<Uint8> signature;
}

@Native<
  Int32 Function(
    Pointer<Void>,
    Size,
    Pointer<RayoModelOptions>,
    Pointer<Pointer<RayoModelHandle>>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_model_create_with(
  Pointer<Void> data,
  int size,
  Pointer<RayoModelOptions> options,
  Pointer<Pointer<RayoModelHandle>> out,
  Pointer<RayoMessage> msg,
);

@Native<
  Int32 Function(
    Pointer<Void>,
    Size,
    Pointer<RayoIntegrity>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_verify(
  Pointer<Void> data,
  int size,
  Pointer<RayoIntegrity> want,
  Pointer<RayoMessage> msg,
);

@Native<
  Int32 Function(Pointer<Uint8>, Pointer<RayoIntegrity>, Pointer<RayoMessage>)
>()
external int rayo_verify_file(
  Pointer<Uint8> path,
  Pointer<RayoIntegrity> want,
  Pointer<RayoMessage> msg,
);

/// `rayo_sampler_options`.
final class RayoSamplerOptions extends Struct {
  @Float()
  external double temperature;

  @Int32()
  external int topK;

  @Float()
  external double topP;

  @Float()
  external double minP;

  @Float()
  external double repetitionPenalty;

  @Uint32()
  external int seed;
}

/// `rayo_generator_options`.
final class RayoGeneratorOptions extends Struct {
  external RayoSamplerOptions sampling;

  @Int()
  external int threads;

  @Int32()
  external int maxTokens;

  @Int32()
  external int context;

  @Int32()
  external int chunk;

  @Int()
  external int cache;
}

/// `rayo_cache_type`.
const rayoCacheFloat32 = 0;
const rayoCacheInt8 = 1;

/// `rayo_process_memory_info`.
final class RayoProcessMemoryInfo extends Struct {
  @Size()
  external int privateBytes;

  @Size()
  external int fileBytes;

  @Size()
  external int residentBytes;

  @Size()
  external int peakResidentBytes;
}

@Native<Int32 Function(Pointer<RayoProcessMemoryInfo>)>()
external int rayo_process_memory(Pointer<RayoProcessMemoryInfo> out);

@Native<Int Function()>(isLeaf: true)
external int rayo_cpu_default_threads();

@Native<
  Int32 Function(
    Pointer<RayoModelHandle>,
    Pointer<Pointer<RayoTokenizerHandle>>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_tokenizer_from_model(
  Pointer<RayoModelHandle> model,
  Pointer<Pointer<RayoTokenizerHandle>> out,
  Pointer<RayoMessage> msg,
);

@Native<
  Int32 Function(
    Pointer<Uint8>,
    Pointer<RayoModelOptions>,
    Pointer<Pointer<RayoModelHandle>>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_model_load(
  Pointer<Uint8> path,
  Pointer<RayoModelOptions> options,
  Pointer<Pointer<RayoModelHandle>> out,
  Pointer<RayoMessage> msg,
);

@Native<Int Function()>(isLeaf: true)
external int rayo_cpu_fast_cores();

@Native<
  Int32 Function(
    Pointer<Uint8>,
    Pointer<Pointer<RayoTokenizerHandle>>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_tokenizer_load(
  Pointer<Uint8> path,
  Pointer<Pointer<RayoTokenizerHandle>> out,
  Pointer<RayoMessage> msg,
);

@Native<Void Function(Pointer<RayoTokenizerHandle>)>()
external void rayo_tokenizer_destroy(Pointer<RayoTokenizerHandle> tokenizer);

@Native<
  Pointer<Uint8> Function(Pointer<RayoTokenizerHandle>, Int32, Pointer<Size>)
>(isLeaf: true)
external Pointer<Uint8> rayo_tokenizer_piece(
  Pointer<RayoTokenizerHandle> tokenizer,
  int id,
  Pointer<Size> size,
);

@Native<
  Int32 Function(
    Pointer<RayoModelHandle>,
    Pointer<RayoTokenizerHandle>,
    Pointer<RayoGeneratorOptions>,
    Pointer<Pointer<RayoGeneratorHandle>>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_generator_create(
  Pointer<RayoModelHandle> model,
  Pointer<RayoTokenizerHandle> tokenizer,
  Pointer<RayoGeneratorOptions> options,
  Pointer<Pointer<RayoGeneratorHandle>> out,
  Pointer<RayoMessage> msg,
);

@Native<Void Function(Pointer<RayoGeneratorHandle>)>()
external void rayo_generator_destroy(Pointer<RayoGeneratorHandle> generator);

@Native<
  Int32 Function(
    Pointer<RayoGeneratorHandle>,
    Pointer<Uint8>,
    Pointer<Uint8>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_generator_chat(
  Pointer<RayoGeneratorHandle> generator,
  Pointer<Uint8> system,
  Pointer<Uint8> user,
  Pointer<RayoMessage> msg,
);

@Native<
  Int32 Function(
    Pointer<RayoGeneratorHandle>,
    Pointer<Uint8>,
    Size,
    Pointer<RayoMessage>,
  )
>()
external int rayo_generator_prompt(
  Pointer<RayoGeneratorHandle> generator,
  Pointer<Uint8> text,
  int size,
  Pointer<RayoMessage> msg,
);

@Native<
  Int32 Function(
    Pointer<RayoGeneratorHandle>,
    Pointer<Int32>,
    Pointer<RayoMessage>,
  )
>()
external int rayo_generator_next(
  Pointer<RayoGeneratorHandle> generator,
  Pointer<Int32> token,
  Pointer<RayoMessage> msg,
);

@Native<Void Function(Pointer<RayoGeneratorHandle>)>()
external void rayo_generator_reset(Pointer<RayoGeneratorHandle> generator);

@Native<Int32 Function(Pointer<RayoGeneratorHandle>)>(isLeaf: true)
external int rayo_generator_position(Pointer<RayoGeneratorHandle> generator);

@Native<Int32 Function(Pointer<RayoGeneratorHandle>)>(isLeaf: true)
external int rayo_generator_context(Pointer<RayoGeneratorHandle> generator);

@Native<Int Function(Pointer<RayoGeneratorHandle>)>(isLeaf: true)
external int rayo_generator_can_chat(Pointer<RayoGeneratorHandle> generator);

@Native<Int Function(Pointer<RayoGeneratorHandle>)>(isLeaf: true)
external int rayo_generator_threads(Pointer<RayoGeneratorHandle> generator);

@Native<Pointer<Uint8> Function(Pointer<RayoGeneratorHandle>)>(isLeaf: true)
external Pointer<Uint8> rayo_generator_kernels(
  Pointer<RayoGeneratorHandle> generator,
);
