import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'bindings/rayo_bindings.dart' as c;
import 'errors.dart';
import 'model.dart';
import 'native_text.dart';

/// A model's tokeniser: the table rayo-convert writes into the model file,
/// read in place ([RayoTokenizer.fromModel]), or its tokenizer.json. Close it
/// after the generators that use it, and before its model.
class RayoTokenizer {
  RayoTokenizer._(this._handle);

  /// The table in [model]'s file, read where it is in the mapped file: no
  /// JSON is parsed. Throws when the file has none (convert it again).
  factory RayoTokenizer.fromModel(RayoModel model) => using((arena) {
    final out = arena<Pointer<c.RayoTokenizerHandle>>();
    final msg = arena<c.RayoMessage>();
    check(c.rayo_tokenizer_from_model(model.handle, out, msg), msg);
    return RayoTokenizer._(out.value);
  });

  /// Reads the tokenizer.json at [path].
  factory RayoTokenizer.load(String path) => using((arena) {
    final out = arena<Pointer<c.RayoTokenizerHandle>>();
    final msg = arena<c.RayoMessage>();
    check(
      c.rayo_tokenizer_load(
        path.toNativeUtf8(allocator: arena).cast(),
        out,
        msg,
      ),
      msg,
    );
    return RayoTokenizer._(out.value);
  });

  final Pointer<c.RayoTokenizerHandle> _handle;
  final Pointer<Size> _size = malloc<Size>();
  bool _closed = false;

  Pointer<c.RayoTokenizerHandle> get handle {
    if (_closed) throw StateError('tokenizer closed');
    return _handle;
  }

  /// The bytes token [id] stands for (a character may span tokens).
  List<int> bytesOf(int id) {
    final p = c.rayo_tokenizer_piece(handle, id, _size);
    return p == nullptr ? const [] : List.of(p.asTypedList(_size.value));
  }

  void close() {
    if (_closed) return;
    _closed = true;
    c.rayo_tokenizer_destroy(_handle);
    malloc.free(_size);
  }
}

/// Text from a language model, a token at a time, with the key/value cache
/// kept across turns (include/rayo/generate.h).
///
/// ```dart
/// final model = RayoModel.map('smollm2.weights');
/// final tokenizer = RayoTokenizer.fromModel(model);
/// final g = RayoGenerator(model, tokenizer, temperature: 0.7);
/// g.chat('Hello!');
/// for (final piece in g.reply()) stdout.write(piece);   // stop early to cancel
/// ```
///
/// Each step is a bounded native call (a prompt chunk, or one token), so a
/// reply is cancelled by breaking out of [reply] or calling [cancel]. Run it
/// in a background isolate (`Isolate.run`, or a long-lived isolate fed by a
/// port) so the UI never waits; a generator belongs to the isolate that
/// made it.
class RayoGenerator {
  /// A generator for [model] (a decoder converted by rayo-convert) and its
  /// [tokenizer]. [threads]: null for the default (a phone's fast cores, at
  /// most 2 on Android and 4 elsewhere), else that many. Zero [temperature]
  /// decodes greedily. [context]: the tokens the conversation holds, null
  /// for 1024 (at most the model's); the key/value cache is sized for it.
  /// [chunk]: prompt tokens a run, null for 32. [int8Cache]: the key/value
  /// cache in int8, a quarter of float32's memory.
  factory RayoGenerator(
    RayoModel model,
    RayoTokenizer tokenizer, {
    double temperature = 0.7,
    int topK = 40,
    double topP = 0.9,
    double minP = 0.05,
    double repetitionPenalty = 1.1,
    int seed = 1,
    int? threads,
    int maxTokens = 256,
    int? context,
    int? chunk,
    bool int8Cache = true,
  }) => using((arena) {
    final o = arena<c.RayoGeneratorOptions>();
    o.ref.sampling.temperature = temperature;
    o.ref.sampling.topK = topK;
    o.ref.sampling.topP = topP;
    o.ref.sampling.minP = minP;
    o.ref.sampling.repetitionPenalty = repetitionPenalty;
    o.ref.sampling.seed = seed;
    o.ref.threads = threads ?? c.rayo_cpu_default_threads();
    o.ref.maxTokens = maxTokens;
    o.ref.context = context ?? 0;
    o.ref.chunk = chunk ?? 0;
    o.ref.cache = int8Cache ? c.rayoCacheInt8 : c.rayoCacheFloat32;
    final out = arena<Pointer<c.RayoGeneratorHandle>>();
    final msg = arena<c.RayoMessage>();
    check(
      c.rayo_generator_create(model.handle, tokenizer.handle, o, out, msg),
      msg,
    );
    return RayoGenerator._(out.value, tokenizer);
  });

  RayoGenerator._(this._handle, this._tokenizer);

  final Pointer<c.RayoGeneratorHandle> _handle;
  final RayoTokenizer _tokenizer;
  final Pointer<c.RayoMessage> _msg = malloc<c.RayoMessage>();
  final Pointer<Int32> _token = malloc<Int32>();
  bool _closed = false;
  bool _cancelled = false;

  Pointer<c.RayoGeneratorHandle> get _h {
    if (_closed) throw StateError('generator closed');
    return _handle;
  }

  /// Whether the model has a chat template.
  bool get canChat => c.rayo_generator_can_chat(_h) != 0;

  /// Tokens in the cache, and the most it holds.
  int get position => c.rayo_generator_position(_h);
  int get context => c.rayo_generator_context(_h);
  int get threads => c.rayo_generator_threads(_h);
  String get kernels => cString(c.rayo_generator_kernels(_h));

  /// A user's turn (with an optional [system] text before the first), in the
  /// model's own chat template, run through the model.
  void chat(String user, {String? system}) => using((arena) {
    _cancelled = false;
    final s = system == null
        ? nullptr
        : system.toNativeUtf8(allocator: arena).cast<Uint8>();
    check(
      c.rayo_generator_chat(
        _h,
        s,
        user.toNativeUtf8(allocator: arena).cast(),
        _msg,
      ),
      _msg,
    );
  });

  /// Raw text, run through the model, for the reply to continue.
  void prompt(String text) => using((arena) {
    _cancelled = false;
    final bytes = utf8.encode(text);
    final p = arena<Uint8>(bytes.isEmpty ? 1 : bytes.length);
    p.asTypedList(bytes.length).setAll(0, bytes);
    check(c.rayo_generator_prompt(_h, p, bytes.length, _msg), _msg);
  });

  /// The next token id, or null when the reply has ended (or was cancelled).
  int? next() {
    if (_cancelled) return null;
    check(c.rayo_generator_next(_h, _token, _msg), _msg);
    return _token.value < 0 ? null : _token.value;
  }

  /// The reply as text, piece by piece as it is made (whole characters only).
  Iterable<String> reply() sync* {
    final pending = <int>[];
    for (var id = next(); id != null; id = next()) {
      pending.addAll(_tokenizer.bytesOf(id));
      final cut = _complete(pending);
      if (cut > 0) {
        yield utf8.decode(pending.sublist(0, cut), allowMalformed: true);
        pending.removeRange(0, cut);
      }
    }
    if (pending.isNotEmpty) yield utf8.decode(pending, allowMalformed: true);
  }

  /// Ends the reply being made: the next step returns nothing.
  void cancel() => _cancelled = true;

  /// Forgets the conversation.
  void reset() => c.rayo_generator_reset(_h);

  void close() {
    if (_closed) return;
    _closed = true;
    c.rayo_generator_destroy(_handle);
    malloc
      ..free(_msg)
      ..free(_token);
  }

  /// How many of [b]'s bytes end on a whole UTF-8 character.
  static int _complete(List<int> b) {
    var i = b.length;
    var back = 0;
    while (i > 0 && back < 4 && (b[i - 1] & 0xC0) == 0x80) {
      i--;
      back++;
    }
    if (i == 0) return b.length;
    final lead = b[i - 1];
    final need = lead >= 0xF0
        ? 4
        : lead >= 0xE0
        ? 3
        : lead >= 0xC0
        ? 2
        : 1;
    return back + 1 >= need ? b.length : i - 1;
  }
}
