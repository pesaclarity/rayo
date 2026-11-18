import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings/rayo_bindings.dart' as c;
import 'errors.dart';
import 'native_text.dart';
import 'tensor.dart';
import 'verify.dart';

/// A loaded model: weights and graph, shared by its sessions. Close it
/// after them.
class RayoModel {
  RayoModel._(this._handle) : name = cString(c.rayo_model_name(_handle)) {
    // The model is the caller's only once this returns: a failure here
    // (metadata that is not an object) frees it, not leaks it.
    try {
      metadata =
          (jsonDecode(cString(c.rayo_model_metadata_json(_handle))) as Map)
              .cast<String, Object?>();
      using((arena) {
        final view = arena<c.RayoTensorView>();
        inputs = [
          for (var i = 0; i < c.rayo_model_input_count(_handle); i++)
            if (c.rayo_model_input_info(_handle, i, view) == 0)
              RayoTensorInfo.fromView(view.ref),
        ];
        outputs = [
          for (var i = 0; i < c.rayo_model_output_count(_handle); i++)
            if (c.rayo_model_output_info(_handle, i, view) == 0)
              cString(view.ref.name),
        ];
      });
    } on Object {
      c.rayo_model_destroy(_handle);
      rethrow;
    }
  }

  /// Reads a model from [bytes] (a plain-weights file with a graph). With
  /// [sha256] (hex), the bytes must hash to it; with [integrity], they must
  /// match it (its SHA-256 and or Ed25519 signature, docs/SECURITY.md),
  /// checked by the C library on the copy it reads, before it parses
  /// anything: a mismatch throws a [RayoException] of status
  /// [rayoErrIntegrity]. rayo keeps its own copy, so [bytes] may be dropped
  /// afterwards.
  factory RayoModel.load(
    Uint8List bytes, {
    String? sha256,
    RayoIntegrity? integrity,
  }) {
    if (sha256 != null &&
        integrity?.sha256 != null &&
        integrity!.sha256!.toLowerCase() != sha256.toLowerCase()) {
      throw ArgumentError('two different SHA-256s');
    }
    final want = sha256 == null
        ? integrity
        : RayoIntegrity(
            sha256: sha256,
            publicKey: integrity?.publicKey,
            signature: integrity?.signature,
          );
    return using((arena) {
      final data = arena<Uint8>(bytes.isEmpty ? 1 : bytes.length);
      data.asTypedList(bytes.length).setAll(0, bytes);
      final out = arena<Pointer<c.RayoModelHandle>>();
      final msg = arena<c.RayoMessage>();
      check(
        c.rayo_model_create_with(
          data.cast(),
          bytes.length,
          _options(arena, want),
          out,
          msg,
        ),
        msg,
      );
      return RayoModel._(out.value);
    });
  }

  /// Maps the model file at [path] (mmap): its weights are used where they
  /// are, with no copy, and cost memory only as they are read. The way to
  /// load a language model. With [integrity], the mapped bytes must match
  /// it before rayo reads them (a model downloaded over the air: require its
  /// signature).
  factory RayoModel.map(String path, {RayoIntegrity? integrity}) =>
      using((arena) {
        final out = arena<Pointer<c.RayoModelHandle>>();
        final msg = arena<c.RayoMessage>();
        check(
          c.rayo_model_load(
            path.toNativeUtf8(allocator: arena).cast(),
            _options(arena, integrity),
            out,
            msg,
          ),
          msg,
        );
        return RayoModel._(out.value);
      });

  /// Reads the model file at [path].
  factory RayoModel.loadFile(
    String path, {
    String? sha256,
    RayoIntegrity? integrity,
  }) => RayoModel.load(
    File(path).readAsBytesSync(),
    sha256: sha256,
    integrity: integrity,
  );

  static Pointer<c.RayoModelOptions> _options(
    Arena arena,
    RayoIntegrity? integrity,
  ) {
    if (integrity == null) return nullptr;
    final o = arena<c.RayoModelOptions>();
    o.ref.ops = nullptr;
    o.ref.opCount = 0;
    o.ref.readFile = 0;
    o.ref.access = 0;
    o.ref.integrity = integrity.toC(arena);
    return o;
  }

  final Pointer<c.RayoModelHandle> _handle;
  bool _closed = false;

  /// The header's "model", such as `example-classifier`.
  final String name;

  /// The header's "metadata" object.
  late final Map<String, Object?> metadata;

  /// Inputs at their largest sizes.
  late final List<RayoTensorInfo> inputs;
  late final List<String> outputs;

  Pointer<c.RayoModelHandle> get handle {
    if (_closed) throw StateError('model closed');
    return _handle;
  }

  int inputIndex(String name) => _index([for (final i in inputs) i.name], name);

  int outputIndex(String name) => _index(outputs, name);

  RayoMemory get memory => using((arena) {
    final m = arena<c.RayoMemoryInfo>();
    c.rayo_model_memory(handle, m);
    return RayoMemory.fromC(m.ref);
  });

  void close() {
    if (_closed) return;
    _closed = true;
    c.rayo_model_destroy(_handle);
  }

  static int _index(List<String> names, String name) {
    final i = names.indexOf(name);
    if (i < 0) throw RayoException(0, 'no tensor $name');
    return i;
  }
}
