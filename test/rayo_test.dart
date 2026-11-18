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
}
