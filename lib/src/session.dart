import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'bindings/rayo_bindings.dart' as c;
import 'errors.dart';
import 'model.dart';
import 'native_text.dart';
import 'tensor.dart';

/// Runs a model. The session plans all its memory when it is made; inputs
/// and outputs are views of that memory, written and read in place, so a
/// run copies nothing and allocates nothing. Keep one session for many
/// runs; it must not run on two threads at once.
///
/// ```dart
/// final s = RayoSession(model)..bind('T', n);
/// s.inputInt32(0).setAll(0, ids);
/// s.run();
/// final probabilities = s.outputFloat32(0);
/// ```
class RayoSession {
  /// A new session on [model]; close it before the model.
  factory RayoSession(RayoModel model) => using((arena) {
    final out = arena<Pointer<c.RayoSessionHandle>>();
    final msg = arena<c.RayoMessage>();
    check(c.rayo_session_create(model.handle, out, msg), msg);
    return RayoSession._(out.value, model.handle, owner: true);
  });

  /// The session [ref] made by [reference], for another isolate of this
  /// process. Closing it frees only this object's buffers.
  factory RayoSession.borrow(RayoSessionRef ref) => RayoSession._(
    Pointer.fromAddress(ref.session),
    Pointer.fromAddress(ref.model),
    owner: false,
  );

  RayoSession._(this._handle, this._model, {required bool owner})
    : _owner = owner,
      _inputs = c.rayo_model_input_count(_model),
      _outputs = c.rayo_model_output_count(_model);

  final Pointer<c.RayoSessionHandle> _handle;
  final Pointer<c.RayoModelHandle> _model;
  final bool _owner;
  final int _inputs;
  final int _outputs;
  // Reused by every call.
  final Pointer<c.RayoMessage> _msg = malloc<c.RayoMessage>();
  final Pointer<c.RayoTensorView> _view = malloc<c.RayoTensorView>();
  final Pointer<Uint8> _name = malloc<Uint8>(64);
  bool _closed = false;

  /// What another isolate needs to [RayoSession.borrow] this session.
  RayoSessionRef get reference =>
      RayoSessionRef(_handle.address, _model.address);

  /// The kernels this session runs, such as `neon-dotprod`.
  String get kernels => cString(c.rayo_session_kernels(_live));

  RayoMemory get memory => using((arena) {
    final m = arena<c.RayoMemoryInfo>();
    c.rayo_session_memory(_live, m);
    return RayoMemory.fromC(m.ref);
  });

  /// Sets size symbol [symbol] (such as `T`) for the next runs; inputs and
  /// outputs then have the new sizes.
  void bind(String symbol, int value) {
    final bytes = utf8.encode(symbol);
    if (bytes.length >= 64) throw RayoException(0, 'symbol too long');
    _name.asTypedList(64)
      ..setAll(0, bytes)
      ..[bytes.length] = 0;
    check(c.rayo_session_bind(_live, _name, value, _msg), _msg);
  }

  /// Input [index] at the current sizes, to write into.
  Int32List inputInt32(int index) {
    final v = _input(index, c.rayoDtypeI32);
    return v.data.cast<Int32>().asTypedList(v.bytes ~/ 4);
  }

  Float32List inputFloat32(int index) {
    final v = _input(index, c.rayoDtypeF32);
    return v.data.cast<Float>().asTypedList(v.bytes ~/ 4);
  }

  /// Runs the graph on this thread.
  void run() => check(c.rayo_session_run(_live, _msg), _msg);

  /// Output [index] of the last run: a view, valid until the next run.
  Float32List outputFloat32(int index) {
    final v = _output(index);
    return v.data.cast<Float>().asTypedList(v.bytes ~/ 4);
  }

  /// Output [index]'s current name, dtype and shape.
  RayoTensorInfo outputInfo(int index) =>
      RayoTensorInfo.fromView(_output(index));

  /// Frees the session (a borrowed one: only this object's buffers).
  void close() {
    if (_closed) return;
    _closed = true;
    malloc
      ..free(_msg)
      ..free(_view)
      ..free(_name);
    if (_owner) c.rayo_session_destroy(_handle);
  }

  Pointer<c.RayoSessionHandle> get _live {
    if (_closed) throw StateError('session closed');
    return _handle;
  }

  c.RayoTensorView _input(int index, int dtype) {
    if (index < 0 || index >= _inputs) {
      throw RangeError.range(index, 0, _inputs - 1, 'index');
    }
    check(c.rayo_session_input(_live, index, _view));
    final v = _view.ref;
    if (v.dtype != dtype) {
      throw RayoException(
        0,
        'input $index is ${RayoDtype.fromC(v.dtype).name}',
      );
    }
    return v;
  }

  c.RayoTensorView _output(int index) {
    if (index < 0 || index >= _outputs) {
      throw RangeError.range(index, 0, _outputs - 1, 'index');
    }
    check(c.rayo_session_output(_live, index, _view));
    return _view.ref;
  }
}

/// Addresses of a session and its model, to borrow it in another isolate.
class RayoSessionRef {
  const RayoSessionRef(this.session, this.model);

  final int session;
  final int model;
}
