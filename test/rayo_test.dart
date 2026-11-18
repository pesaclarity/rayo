import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:rayo/rayo.dart';
import 'package:test/test.dart';

// Models and NumPy outputs from tool/make_test_models.py.
final _expected =
    jsonDecode(File('tests/models/expected.json').readAsStringSync())
        as Map<String, Object?>;

Uint8List _file(String name) => File('tests/models/$name').readAsBytesSync();

List<double> _floats(Object? v) => [
  for (final x in v! as List) (x as num).toDouble(),
];

double _worst(List<double> got, List<double> want) {
  expect(got, hasLength(want.length));
  var worst = 0.0;
  for (var i = 0; i < got.length; i++) {
    final d = (got[i] - want[i]).abs();
    if (d > worst) worst = d;
  }
  return worst;
}

void main() {
  final reader = _expected['reader']! as Map<String, Object?>;
  final cases = (reader['cases']! as List).cast<Map<String, Object?>>();

  test('loads the native library', () {
    expect(rayoAvailable(), isTrue);
    expect(rayoVersion(), '0.1.0');
    expect(
      rayoBuiltinOps(),
      containsAll(['dense', 'compare', 'masked_reduce']),
    );
  });

  test('SHA-256', () {
    expect(
      rayoSha256(Uint8List.fromList(utf8.encode('abc'))),
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
  });

  test('describes the model', () {
    final m = RayoModel.load(_file('reader.weights'));
    addTearDown(m.close);
    expect(m.name, 'test-reader');
    expect(m.metadata, {
      'labels': ['a', 'b', 'c'],
      'count': 3,
    });
    expect('${m.inputs}', '[tokens int32 [12], shapes int32 [12]]');
    expect(m.outputs, ['slots', 'intent']);
    expect(m.outputIndex('intent'), 1);
    expect(m.memory.weights, _file('reader.weights').length);
  });

  test('refuses a wrong SHA-256 and a damaged file', () {
    final bytes = _file('reader.weights');
    expect(
      () => RayoModel.load(bytes, sha256: '00' * 32),
      throwsA(isA<RayoException>()),
    );
    expect(
      () => RayoModel.load(Uint8List.sublistView(bytes, 0, 200)),
      throwsA(isA<RayoException>().having((e) => e.status, 'status', 2)),
    );
    RayoModel.load(bytes, sha256: rayoSha256(bytes)).close();
  });

  // RFC 8032's test 1 key; tests/models/reader.weights signed with it
  // (tool/sign_model.py).
  Uint8List hex(String s) => Uint8List.fromList([
    for (var i = 0; i < s.length; i += 2)
      int.parse(s.substring(i, i + 2), radix: 16),
  ]);
  final publicKey = hex(
    'd75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a',
  );
  final signature = hex(
    '932b2f6915239118d8edb3e10047d8f4e6eb84fa233a7fdb68842d246cb2079b'
    'f9385856642a59144177adbe3a0b59a8060b41881718331a01c8e86427123201',
  );

  test('verifies a signed model before reading it', () {
    final bytes = _file('reader.weights');
    final signed = RayoIntegrity(publicKey: publicKey, signature: signature);
    rayoVerify(bytes, signed);
    rayoVerifyFile('tests/models/reader.weights', signed);
    RayoModel.load(bytes, integrity: signed).close();
    final m = RayoModel.map('tests/models/reader.weights', integrity: signed);
    expect(m.name, 'test-reader');
    m.close();
    RayoModel.load(
      bytes,
      integrity: RayoIntegrity(
        sha256: rayoSha256(bytes),
        publicKey: publicKey,
        signature: signature,
      ),
    ).close();
    // A changed byte, another key's signature, a wrong digest.
    final damaged = Uint8List.fromList(bytes)..[bytes.length - 1] ^= 1;
    final integrity = isA<RayoException>().having(
      (e) => e.status,
      'status',
      rayoErrIntegrity,
    );
    expect(() => rayoVerify(damaged, signed), throwsA(integrity));
    expect(
      () => RayoModel.load(damaged, integrity: signed),
      throwsA(integrity),
    );
    final other = Uint8List.fromList(publicKey)..[0] ^= 1;
    expect(
      () => RayoModel.map(
        'tests/models/reader.weights',
        integrity: RayoIntegrity(publicKey: other, signature: signature),
      ),
      throwsA(integrity),
    );
    expect(
      () => RayoModel.load(bytes, integrity: RayoIntegrity(sha256: '00' * 32)),
      throwsA(integrity),
    );
    expect(() => RayoIntegrity(publicKey: publicKey), throwsArgumentError);
    expect(
      () =>
          RayoIntegrity(publicKey: publicKey.sublist(1), signature: signature),
      throwsArgumentError,
    );
    expect(() => RayoIntegrity(sha256: 'xyz'), throwsArgumentError);
  });

  test('matches NumPy, writing inputs and reading outputs in place', () {
    final m = RayoModel.load(_file('reader.weights'));
    final s = RayoSession(m);
    addTearDown(() {
      s.close();
      m.close();
    });
    for (final c in cases) {
      s.bind('T', c['T']! as int);
      s.inputInt32(0).setAll(0, (c['tokens']! as List).cast<int>());
      s.inputInt32(1).setAll(0, (c['shapes']! as List).cast<int>());
      s.run();
      expect(_worst(s.outputFloat32(0), _floats(c['slots'])), lessThan(1e-5));
      expect(_worst(s.outputFloat32(1), _floats(c['intent'])), lessThan(1e-5));
      expect(s.outputInfo(0).shape, [c['T']! as int, 3]);
    }
    expect(s.kernels, isNotEmpty);
    expect(s.memory.arena, greaterThan(0));
  });

  test('runs in another isolate', () async {
    final m = RayoModel.load(_file('reader.weights'));
    final s = RayoSession(m);
    addTearDown(() {
      s.close();
      m.close();
    });
    final c = cases.first;
    final tokens = (c['tokens']! as List).cast<int>();
    final shapes = (c['shapes']! as List).cast<int>();
    final t = c['T']! as int;
    final intent = await runInIsolate(s, (b) {
      b.bind('T', t);
      b.inputInt32(0).setAll(0, tokens);
      b.inputInt32(1).setAll(0, shapes);
      b.run();
      return Float32List.fromList(b.outputFloat32(1));
    });
    expect(_worst(intent, _floats(c['intent'])), lessThan(1e-5));
  });

  test('reports errors instead of crashing', () {
    final m = RayoModel.load(_file('reader.weights'));
    final s = RayoSession(m);
    addTearDown(() {
      s.close();
      m.close();
    });
    expect(() => s.bind('T', 0), throwsA(isA<RayoException>()));
    expect(() => s.bind('T', 13), throwsA(isA<RayoException>()));
    expect(() => s.bind('Q', 1), throwsA(isA<RayoException>()));
    expect(() => s.inputFloat32(0), throwsA(isA<RayoException>()));
    expect(() => s.inputInt32(5), throwsRangeError);
    s.bind('T', 2);
    s.inputInt32(0).setAll(0, [99999, 1]);
    s.inputInt32(1).setAll(0, [1, 1]);
    expect(s.run, throwsA(isA<RayoException>()));
    s.close();
    expect(s.run, throwsStateError);
  });

  test('decodes token by token with a key/value cache', () {
    final d = _expected['decoder']! as Map<String, Object?>;
    final tokens = (d['tokens']! as List).cast<int>();
    final want = [for (final row in d['logits']! as List) _floats(row)];
    final m = RayoModel.load(_file('decoder.weights'));
    final s = RayoSession(m);
    addTearDown(() {
      s.close();
      m.close();
    });
    for (var round = 0; round < 2; round++) {
      s.reset();
      var past = 0;
      for (final n in [2, 1, 1, 1, 1]) {
        s
          ..bind('P', past)
          ..bind('T', n);
        s.inputInt32(0).setAll(0, tokens.sublist(past, past + n));
        s.run();
        past += n;
        expect(_worst(s.outputFloat32(0), want[past - 1]), lessThan(1e-5));
      }
    }
    s.bind('P', tokens.length - 1);
    expect(() => s.bind('T', 2), throwsA(isA<RayoException>()));
  });

  test('a language model chats, streams and cancels', () {
    final model = RayoModel.map('tests/models/llama-q4.weights');
    final tokenizer = RayoTokenizer.load('tests/models/llama.tokenizer.json');
    final g = RayoGenerator(model, tokenizer, temperature: 0, maxTokens: 6);
    expect(g.canChat, isTrue);
    expect(g.context, 256);
    g.chat('hello there', system: 'be brief');
    final first = g.reply().join();
    expect(g.position, greaterThan(6));
    g.reset();
    g.chat('hello there', system: 'be brief');
    expect(g.reply().join(), first); // greedy: the same again
    g.chat('and the cat?');
    var pieces = 0;
    for (final _ in g.reply()) {
      if (++pieces == 2) g.cancel();
    }
    expect(pieces, 2);
    expect(
      () => RayoTokenizer.load('no/such.json'),
      throwsA(isA<RayoException>()),
    );
    g.close();
    tokenizer.close();
    model.close();
  });

  test('a lean language model: its table, an int8 cache, a short context', () {
    final model = RayoModel.map('tests/models/llama-q3.weights');
    final tokenizer = RayoTokenizer.fromModel(model);
    final g = RayoGenerator(
      model,
      tokenizer,
      temperature: 0,
      maxTokens: 4,
      context: 64,
      chunk: 8,
    );
    expect(g.context, 64);
    g.chat('hello there');
    expect(g.reply().join(), isNotEmpty);
    final old = RayoModel.map('tests/models/llama-q4.weights');
    expect(() => RayoTokenizer.fromModel(old), throwsA(isA<RayoException>()));
    final memory = RayoProcessMemory.now();
    if (memory != null) {
      expect(memory.privateBytes, greaterThan(0));
      expect(memory.residentBytes, greaterThanOrEqualTo(memory.privateBytes));
    }
    g.close();
    tokenizer.close();
    old.close();
    model.close();
  });
}
