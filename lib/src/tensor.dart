import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'bindings/rayo_bindings.dart' as c;
import 'native_text.dart';

/// An element type.
enum RayoDtype {
  float32,
  int32,
  int8;

  static RayoDtype fromC(int v) => switch (v) {
    c.rayoDtypeF32 => float32,
    c.rayoDtypeI32 => int32,
    _ => int8,
  };
}

/// A model input's or output's name, dtype and shape (row-major, outermost
/// first).
class RayoTensorInfo {
  const RayoTensorInfo(this.name, this.dtype, this.shape);

  factory RayoTensorInfo.fromView(c.RayoTensorView v) => RayoTensorInfo(
    cString(v.name),
    RayoDtype.fromC(v.dtype),
    [for (var i = 0; i < v.shape.rank; i++) v.shape.dims[i]],
  );

  final String name;
  final RayoDtype dtype;
  final List<int> shape;

  int get count => shape.fold(1, (a, b) => a * b);

  @override
  String toString() => '$name ${dtype.name} $shape';
}

/// Bytes held by a model or a session.
class RayoMemory {
  const RayoMemory(this.weights, this.arena, this.overhead, this.states);

  factory RayoMemory.fromC(c.RayoMemoryInfo m) =>
      RayoMemory(m.weights, m.arena, m.overhead, m.states);

  /// The model's copy of the file and packed weights.
  final int weights;

  /// A session's activations and scratch, planned once.
  final int arena;

  /// Graph, plans and tables.
  final int overhead;

  /// Of [arena]: a session's states (a decoder's key/value cache), in full.
  final int states;

  int get total => weights + arena + overhead;
}

/// What the whole process holds in memory now, as the system counts it
/// (rayo_process_memory): its own (private: the heap, caches) and the
/// resident pages of mapped files (the weights), which the system can drop
/// under pressure. Null where the system does not say.
class RayoProcessMemory {
  const RayoProcessMemory(
    this.privateBytes,
    this.fileBytes,
    this.residentBytes,
    this.peakResidentBytes,
  );

  /// Now, or null where the system does not say (Windows, for now).
  static RayoProcessMemory? now() {
    final m = malloc<c.RayoProcessMemoryInfo>();
    try {
      if (c.rayo_process_memory(m) != 0) return null;
      return RayoProcessMemory(
        m.ref.privateBytes,
        m.ref.fileBytes,
        m.ref.residentBytes,
        m.ref.peakResidentBytes,
      );
    } finally {
      malloc.free(m);
    }
  }

  final int privateBytes;
  final int fileBytes;
  final int residentBytes;
  final int peakResidentBytes;
}
