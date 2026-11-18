/// rayo: on-device inference in plain C, for Dart and Flutter.
///
/// Load a model once, keep a [RayoSession] for many runs, write inputs and
/// read outputs in place:
///
/// ```dart
/// final model = RayoModel.load(bytes, sha256: expected);
/// final session = RayoSession(model);
/// session.inputInt32(0).setAll(0, tokens);
/// session.run();
/// final scores = session.outputFloat32(0);
/// ```
library;

export 'src/errors.dart' show RayoException;
export 'src/generator.dart' show RayoGenerator, RayoTokenizer;
export 'src/isolate_runner.dart' show runInIsolate;
export 'src/library.dart'
    show rayoAvailable, rayoBuiltinOps, rayoSha256, rayoVersion;
export 'src/model.dart' show RayoModel;
export 'src/session.dart' show RayoSession, RayoSessionRef;
export 'src/verify.dart'
    show RayoIntegrity, rayoErrIntegrity, rayoVerify, rayoVerifyFile;
export 'src/tensor.dart'
    show RayoDtype, RayoMemory, RayoProcessMemory, RayoTensorInfo;
