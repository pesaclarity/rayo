import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:rayo/rayo.dart';
import 'package:test/test.dart';

// Models and NumPy outputs from tool/make_test_models.py.
final _expected =
    jsonDecode(File('test/models/expected.json').readAsStringSync())
        as Map<String, Object?>;

Uint8List _model(String name) =>
    File('test/models/$name.gguf').readAsBytesSync();

List<double> _floats(Object? v) =>
    (v! as List).cast<num>().map((x) => x.toDouble()).toList();

void _close(List<double> got, List<double> want, {double tol = 1e-4}) {
  expect(got.length, want.length);
  for (var i = 0; i < got.length; i++) {
    final scale = want[i].abs() > 1 ? want[i].abs() : 1.0;
    if ((got[i] - want[i]).abs() > tol * scale) {
      fail('at $i: got ${got[i]}, want ${want[i]}');
    }
  }
}

void main() {
  test('loads the native library', () {
    expect(rayoAvailable(), isTrue);
    expect(rayoVersion(), startsWith('0.1.0 (ggml 1.0.0'));
  });

  test('SHA-256', () {
    expect(
      rayoSha256(Uint8List(0)),
      'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    );
    expect(
      rayoSha256(Uint8List.fromList(utf8.encode('abc'))),
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    );
    final long = Uint8List.fromList(utf8.encode('a' * 1000));
    expect(
      rayoSha256(long),
      '41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3',
    );
  });

  test('reads metadata, inputs and outputs', () {
    final m = RayoModel.load(_model('ops'));
    addTearDown(m.close);
    expect(m.metadata['general.name'], 'ops');
    expect(m.metadata['test.labels'], ['a', 'b']);
    expect(m.metadata['test.count'], 3);
    expect(m.metadata['test.ratio'], 0.5);
    expect(m.metadata['test.flag'], true);
    expect(m.metadata.containsKey('rayo.graph'), isFalse);
    expect(
      [for (final i in m.inputs) '${i.name} ${i.dtype} ${i.shape}'],
      ['ids i32 [T]', 'x f32 [8, T]'],
    );
    expect(m.outputs, contains('softmax'));
  });

  test('rejects a wrong SHA-256 and a damaged file', () {
    final bytes = _model('ops');
    expect(
      () => RayoModel.load(bytes, sha256: '00' * 32),
      throwsA(isA<RayoException>()),
    );
    expect(
      () => RayoModel.load(Uint8List.sublistView(bytes, 0, 100)),
      throwsA(isA<RayoException>()),
    );
    expect(
      () => RayoModel.load(Uint8List.fromList(utf8.encode('not a model'))),
      throwsA(isA<RayoException>()),
    );
    RayoModel.load(bytes, sha256: rayoSha256(bytes)).close();
  });

  test('ops match NumPy', () {
    final m = RayoModel.load(_model('ops'));
    final s = m.session();
    addTearDown(() {
      s.close();
      m.close();
    });
    final cases = ((_expected['ops']! as Map)['cases']! as List)
        .cast<Map<String, Object?>>();
    for (final c in cases) {
      final ids = (c['ids']! as List).cast<int>();
      s
        ..bind('T', ids.length)
        ..setInt32('ids', Int32List.fromList(ids))
        ..setFloat32('x', Float32List.fromList(_floats(c['x'])))
        ..run();
      for (final name in [
        'conv',
        'norm',
        'wide',
        'max',
        'mean',
        'softmax',
        'last',
      ]) {
        _close(s.output(name).data, _floats(c[name]));
      }
      // Trailing sizes of 1 are dropped.
      expect(s.output('norm').shape, ids.length == 1 ? [8] : [8, ids.length]);
    }
  });

  test('Q8_0 weights', () {
    final m = RayoModel.load(_model('q8'));
    final s = m.session();
    addTearDown(() {
      s.close();
      m.close();
    });
    final q = _expected['q8']! as Map<String, Object?>;
    s
      ..bind('T', 5)
      ..setFloat32('x', Float32List.fromList(_floats(q['x'])))
      ..run();
    // ggml quantises the input to Q8_0 too, so not exact.
    _close(s.output('y').data, _floats(q['y']), tol: 3e-2);
  });

  group('decoder with a key/value cache', () {
    final d = _expected['decoder']! as Map<String, Object?>;
    final tokens = (d['tokens']! as List).cast<int>();
    final want = _floats(d['logits']);
    test('all at once', () {
      final m = RayoModel.load(_model('decoder'));
      final s = m.session();
      addTearDown(() {
        s.close();
        m.close();
      });
      s
        ..bind('P', 0)
        ..bind('T', tokens.length)
        ..setInt32('tokens', Int32List.fromList(tokens))
        ..run();
      _close(s.output('logits').data, want);
    });

    test('step by step, on two threads, in another isolate', () async {
      final m = RayoModel.load(_model('decoder'));
      final s = m.session(threads: 2);
      addTearDown(() {
        s.close();
        m.close();
      });
      for (var round = 0; round < 2; round++) {
        s.reset();
        final address = s.address;
        final got = await Isolate.run(() {
          final w = RayoSession.fromAddress(address);
          final out = <double>[];
          // Three tokens, then one at a time.
          var p = 0;
          for (final n in [3, 1, 1, 1]) {
            w
              ..bind('P', p)
              ..bind('T', n)
              ..setInt32('tokens', Int32List.fromList(tokens.sublist(p, p + n)))
              ..run();
            out.addAll(w.output('logits').data);
            p += n;
          }
          w.close();
          return out;
        });
        _close(got, want);
      }
    });

    test('reports a bad binding instead of crashing', () {
      final m = RayoModel.load(_model('decoder'));
      final s = m.session();
      addTearDown(() {
        s.close();
        m.close();
      });
      s
        ..bind('P', 10)
        ..bind('T', 5)
        ..setInt32('tokens', Int32List(5));
      expect(s.run, throwsA(isA<RayoException>()));
      s.bind('T', 2);
      expect(s.run, throwsA(isA<RayoException>()));
      expect(() => s.output('nope'), throwsA(isA<RayoException>()));
      expect(
        () => s.setInt32('nope', Int32List(1)),
        throwsA(isA<RayoException>()),
      );
    });
  });
}
