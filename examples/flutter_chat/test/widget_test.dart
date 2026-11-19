import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:rayo/rayo.dart';
import 'package:rayo_flutter_chat/main.dart';
import 'package:rayo_flutter_chat/runner.dart';

/// Answers at once, and remembers what it was asked.
class FakeRunner implements PromptRunner {
  final asked = <(String, String)>[];

  @override
  Future<String> run(String modelPath, String prompt) async {
    asked.add((modelPath, prompt));
    if (modelPath.isEmpty) throw ArgumentError('no model file');
    return 'A plan for your money.';
  }
}

// A tiny language model of the repository's tests (random weights, with its
// tokeniser's table in the file).
const tinyModel = '../../tests/models/llama-q4h.weights';

void main() {
  testWidgets('runs the prompt and shows the reply', (tester) async {
    final runner = FakeRunner();
    await tester.pumpWidget(ChatApp(runner: runner));
    await tester.enterText(find.byKey(const Key('model')), 'model.weights');
    await tester.enterText(find.byKey(const Key('prompt')), 'Hello');
    await tester.tap(find.byKey(const Key('run')));
    await tester.pumpAndSettle();
    expect(runner.asked, [('model.weights', 'Hello')]);
    expect(find.text('A plan for your money.'), findsOneWidget);
    expect(find.byKey(const Key('error')), findsNothing);
  });

  testWidgets('shows an error instead of a reply', (tester) async {
    await tester.pumpWidget(ChatApp(runner: FakeRunner()));
    await tester.enterText(find.byKey(const Key('model')), '');
    await tester.tap(find.byKey(const Key('run')));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('error')), findsOneWidget);
  });

  test('rayo itself: a tiny model loads and replies', () {
    expect(rayoAvailable(), isTrue);
    expect(File(tinyModel).existsSync(), isTrue);
    final reply = RayoPromptRunner.reply(tinyModel, 'hello', maxTokens: 8);
    expect(reply, isA<String>());
  });
}
