/// rayo: on-device inference for Dart and Flutter, on ggml compiled from
/// source.
///
/// ```dart
/// final model = RayoModel.load(bytes, sha256: expected);
/// final session = model.session(threads: 1);
/// session
///   ..bind('T', ids.length)
///   ..setInt32('tokens', ids)
///   ..run();
/// final probabilities = session.output('intent').data;
/// session.close();
/// model.close();
/// ```
///
/// A session runs on the thread that calls [RayoSession.run]. To keep a
/// Flutter UI free, run it in another isolate: pass [RayoSession.address]
/// and open it there with [RayoSession.fromAddress] (see [RayoSession]).
library;

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'src/ffi.dart' as ffi;

/// A rayo call that failed: a bad model, input or graph.
class RayoException implements Exception {
  RayoException(this.message);

  final String message;

  @override
  String toString() => 'RayoException: $message';
}

const _errLen = 512;

/// Whether the native library loads on this device (it does on every target
/// the build hook compiles for). Never throws.
bool rayoAvailable() {
  try {
    ffi.rayo_version();
    return true;
  } on Object {
    return false;
  }
}

/// The runtime's version, such as `0.1.0 (ggml 1.0.0 417a024)`.
String rayoVersion() => ffi.rayo_version().toDartString();

/// Lower-case hex SHA-256 of [bytes], computed natively.
String rayoSha256(Uint8List bytes) => using((arena) {
  final data = arena<Uint8>(bytes.isEmpty ? 1 : bytes.length);
  data.asTypedList(bytes.length).setAll(0, bytes);
  final out = arena<Uint8>(32);
  ffi.rayo_sha256(data, bytes.length, out);
  return [
    for (final b in out.asTypedList(32)) b.toRadixString(16).padLeft(2, '0'),
  ].join();
});

/// An input, state or output the graph declares.
class RayoTensorInfo {
  const RayoTensorInfo(this.name, this.dtype, this.shape);

  final String name;

  /// `f32` or `i32`.
  final String dtype;

  /// Innermost dimension first; sizes or symbols such as `T`.
  final List<String> shape;
}

/// A loaded model: its weights and graph, shared by its sessions.
class RayoModel {
  RayoModel._(this._handle) {
    final info =
        jsonDecode(ffi.rayo_model_info(_handle).toDartString())
            as Map<String, Object?>;
    metadata = (info['metadata']! as Map).cast<String, Object?>();
    List<RayoTensorInfo> tensors(String key) => [
      for (final t in (info[key]! as List).cast<Map<String, Object?>>())
        RayoTensorInfo(
          t['name']! as String,
          t['dtype']! as String,
          (t['shape']! as List).cast<String>(),
        ),
    ];
    inputs = tensors('inputs');
    states = tensors('states');
    outputs = (info['outputs']! as List).cast<String>();
  }

  /// Reads a model from GGUF [bytes]. With [sha256] (lower-case hex), the
  /// bytes must hash to it.
  factory RayoModel.load(Uint8List bytes, {String? sha256}) {
    if (sha256 != null && rayoSha256(bytes) != sha256.toLowerCase()) {
      throw RayoException('the model does not match its SHA-256');
    }
    return using((arena) {
      final data = arena<Uint8>(bytes.isEmpty ? 1 : bytes.length);
      data.asTypedList(bytes.length).setAll(0, bytes);
      final err = arena<Uint8>(_errLen).cast<Utf8>();
      final handle = ffi.rayo_model_load(data, bytes.length, err, _errLen);
      if (handle == nullptr) throw RayoException(err.toDartString());
      return RayoModel._(handle);
    });
  }

  /// Reads a model from the GGUF file at [path].
  factory RayoModel.loadFile(String path, {String? sha256}) =>
      RayoModel.load(File(path).readAsBytesSync(), sha256: sha256);

  final Pointer<ffi.RayoModelHandle> _handle;
  bool _closed = false;

  /// The GGUF metadata, except the graph itself.
  late final Map<String, Object?> metadata;
  late final List<RayoTensorInfo> inputs;
  late final List<RayoTensorInfo> states;
  late final List<String> outputs;

  /// A new session; close it before this model. [threads] above 1 split
  /// each step over that many threads, which pays only for large models.
  RayoSession session({int threads = 1}) {
    _check();
    return using((arena) {
      final err = arena<Uint8>(_errLen).cast<Utf8>();
      final s = ffi.rayo_session_new(_handle, threads, err, _errLen);
      if (s == nullptr) throw RayoException(err.toDartString());
      return RayoSession._(s);
    });
  }

  /// Frees the model. Its sessions must be closed first.
  void close() {
    if (_closed) return;
    _closed = true;
    ffi.rayo_model_free(_handle);
  }

  void _check() {
    if (_closed) throw StateError('closed');
  }
}

/// One output of the last run: float32 values, innermost dimension first in
/// [shape] (so a `[C, T]` output is T rows of C values), without trailing
/// sizes of 1.
class RayoOutput {
  const RayoOutput(this.data, this.shape);

  final Float32List data;
  final List<int> shape;
}

/// Runs a model: sizes bound with [bind], inputs set with [setInt32] or
/// [setFloat32], then [run]. A session must not run on two threads at once.
///
/// In another isolate:
///
/// ```dart
/// final address = session.address;
/// final out = await Isolate.run(() {
///   final s = RayoSession.fromAddress(address)
///     ..setInt32('tokens', ids)
///     ..run();
///   return s.output('logits').data;
/// });
/// ```
class RayoSession {
  RayoSession._(this._handle) : _owner = true;

  /// The session at [address] (from [address]), for use in another isolate
  /// of the same process. Closing this object frees only its own buffers;
  /// the session stays with the object that created it.
  RayoSession.fromAddress(int address)
    : _handle = Pointer.fromAddress(address),
      _owner = false;

  final Pointer<ffi.RayoSessionHandle> _handle;
  final bool _owner;
  final Pointer<Utf8> _err = malloc<Uint8>(_errLen).cast();
  Pointer<Uint8> _buffer = nullptr;
  int _bufferSize = 0;
  bool _closed = false;

  int get address => _handle.address;

  /// Gives size symbol [name] (such as `T`) a value for the next runs.
  void bind(String name, int value) => _call(
    (err) => using(
      (arena) => ffi.rayo_session_bind(
        _handle,
        name.toNativeUtf8(allocator: arena),
        value,
        err,
        _errLen,
      ),
    ),
  );

  void setInt32(String name, Int32List values) => _set(
    name,
    values.buffer.asUint8List(values.offsetInBytes, values.lengthInBytes),
  );

  void setFloat32(String name, Float32List values) => _set(
    name,
    values.buffer.asUint8List(values.offsetInBytes, values.lengthInBytes),
  );

  void _set(String name, Uint8List bytes) {
    _check();
    if (bytes.length > _bufferSize) {
      if (_buffer != nullptr) malloc.free(_buffer);
      _bufferSize = bytes.length;
      _buffer = malloc<Uint8>(_bufferSize);
    }
    _buffer.asTypedList(bytes.length).setAll(0, bytes);
    _call(
      (err) => using(
        (arena) => ffi.rayo_session_set_input(
          _handle,
          name.toNativeUtf8(allocator: arena),
          _buffer.cast(),
          bytes.length,
          err,
          _errLen,
        ),
      ),
    );
  }

  /// Runs the graph on this thread.
  void run() => _call((err) => ffi.rayo_session_run(_handle, err, _errLen));

  /// Output [name] of the last run, copied.
  RayoOutput output(String name) {
    _check();
    return using((arena) {
      final ne = arena<Int64>(4);
      final p = ffi.rayo_session_output(
        _handle,
        name.toNativeUtf8(allocator: arena),
        ne,
        _err,
        _errLen,
      );
      if (p == nullptr) throw RayoException(_err.toDartString());
      final shape = [ne[0], ne[1], ne[2], ne[3]];
      final n = shape.fold(1, (a, b) => a * b);
      while (shape.length > 1 && shape.last == 1) {
        shape.removeLast();
      }
      return RayoOutput(Float32List.fromList(p.asTypedList(n)), shape);
    });
  }

  /// Zeroes the session's state, such as a key/value cache.
  void reset() {
    _check();
    ffi.rayo_session_reset(_handle);
  }

  /// Frees the session (or, for one from [fromAddress], this object's
  /// buffers only).
  void close() {
    if (_closed) return;
    _closed = true;
    malloc.free(_err);
    if (_buffer != nullptr) malloc.free(_buffer);
    if (_owner) ffi.rayo_session_free(_handle);
  }

  void _call(int Function(Pointer<Utf8> err) f) {
    _check();
    if (f(_err) != 0) throw RayoException(_err.toDartString());
  }

  void _check() {
    if (_closed) throw StateError('closed');
  }
}
