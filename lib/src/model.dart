import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings/rayo_bindings.dart' as c;
import 'errors.dart';
import 'library.dart';
import 'native_text.dart';
import 'tensor.dart';

/// A loaded model: weights and graph, shared by its sessions. Close it
/// after them.
class RayoModel {
  RayoModel._(this._handle) : name = cString(c.rayo_model_name(_handle)) {
    metadata = (jsonDecode(cString(c.rayo_model_metadata_json(_handle))) as Map)
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
  }

  /// Reads a model from [bytes] (a plain-weights file with a graph). With
  /// [sha256] (lower-case hex), the bytes must hash to it. rayo keeps its own
  /// copy, so [bytes] may be dropped afterwards.
  factory RayoModel.load(Uint8List bytes, {String? sha256}) {
    if (sha256 != null && rayoSha256(bytes) != sha256.toLowerCase()) {
      throw RayoException(0, 'the model does not match its SHA-256');
    }
    return using((arena) {
      final data = arena<Uint8>(bytes.isEmpty ? 1 : bytes.length);
      data.asTypedList(bytes.length).setAll(0, bytes);
      final out = arena<Pointer<c.RayoModelHandle>>();
      final msg = arena<c.RayoMessage>();
      check(c.rayo_model_create(data.cast(), bytes.length, out, msg), msg);
      return RayoModel._(out.value);
    });
  }

  /// Reads the model file at [path].
  factory RayoModel.loadFile(String path, {String? sha256}) =>
      RayoModel.load(File(path).readAsBytesSync(), sha256: sha256);

  final Pointer<c.RayoModelHandle> _handle;
  bool _closed = false;

  /// The header's "model", such as `ask-reader`.
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
