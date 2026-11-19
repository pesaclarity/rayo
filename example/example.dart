// package:rayo from the command line: a model's outputs for some token ids,
// or a language model's reply to a prompt.
//
//   dart run example/example.dart examples/classifier/classifier.weights 4 17 9
//   dart run example/example.dart smollm2.weights --chat 'What is a budget?'
//
// The first runs the small classifier in the repository (three labels'
// probabilities); the second a language model converted by rayo-convert
// (docs/GETTING-STARTED.md), with the reply streamed as it is made.
import 'dart:io';

import 'package:rayo/rayo.dart';

void main(List<String> args) {
  if (args.length < 2) {
    stderr.writeln('usage: example MODEL.weights ID... | --chat TEXT');
    exitCode = 2;
    return;
  }
  if (!rayoAvailable()) {
    stderr.writeln('the rayo native library did not load');
    exitCode = 1;
    return;
  }
  final model = RayoModel.map(args[0]);
  try {
    if (args[1] == '--chat') {
      _chat(model, args.skip(2).join(' '));
    } else {
      _run(model, [for (final a in args.skip(1)) int.parse(a)]);
    }
  } on RayoException catch (e) {
    stderr.writeln(e);
    exitCode = 1;
  } finally {
    model.close();
  }
}

/// One run: bind the sequence length, write the ids in place, read the
/// first output in place.
void _run(RayoModel model, List<int> ids) {
  final session = RayoSession(model);
  try {
    session.bind('T', ids.length);
    session.inputInt32(0).setAll(0, ids);
    session.run();
    print('${model.outputs[0]}: ${session.outputFloat32(0)}');
    print('kernels: ${session.kernels}');
  } finally {
    session.close();
  }
}

/// A reply from a language model, greedy, with the model's chat template.
void _chat(RayoModel model, String prompt) {
  final tokenizer = RayoTokenizer.fromModel(model);
  final generator = RayoGenerator(
    model,
    tokenizer,
    temperature: 0,
    maxTokens: 128,
  );
  try {
    if (generator.canChat) {
      generator.chat(prompt);
    } else {
      generator.prompt(prompt);
    }
    for (final piece in generator.reply()) {
      stdout.write(piece);
    }
    stdout.writeln();
  } finally {
    generator.close();
    tokenizer.close();
  }
}
