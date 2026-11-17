import 'dart:ffi';

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
  const RayoMemory(this.weights, this.arena, this.overhead);

  factory RayoMemory.fromC(c.RayoMemoryInfo m) =>
      RayoMemory(m.weights, m.arena, m.overhead);

  /// The model's copy of the file and packed weights.
  final int weights;

  /// A session's activations and scratch, planned once.
  final int arena;

  /// Graph, plans and tables.
  final int overhead;

  int get total => weights + arena + overhead;
}
