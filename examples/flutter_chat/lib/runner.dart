import 'dart:isolate';

import 'package:rayo/rayo.dart';

/// Runs one prompt on a model and gives the whole reply.
abstract interface class PromptRunner {
  Future<String> run(String modelPath, String prompt);
}

/// The real one: package:rayo in a background isolate, so the UI never
/// waits. The model is mapped from [modelPath] (a language model converted
/// by rayo-convert, with its tokeniser's table in the file), used for one
/// reply and closed again. An app that chats keeps the model, tokeniser and
/// generator in a long-lived isolate instead (docs/GETTING-STARTED.md).
class RayoPromptRunner implements PromptRunner {
  const RayoPromptRunner({this.maxTokens = 128});

  final int maxTokens;

  @override
  Future<String> run(String modelPath, String prompt) {
    final max = maxTokens;
    return Isolate.run(() => reply(modelPath, prompt, maxTokens: max));
  }

  /// The reply on this isolate.
  static String reply(String modelPath, String prompt, {int maxTokens = 128}) {
    final model = RayoModel.map(modelPath);
    RayoTokenizer? tokenizer;
    RayoGenerator? generator;
    try {
      tokenizer = RayoTokenizer.fromModel(model);
      generator = RayoGenerator(
        model,
        tokenizer,
        temperature: 0,
        maxTokens: maxTokens,
      );
      if (generator.canChat) {
        generator.chat(prompt);
      } else {
        generator.prompt(prompt);
      }
      return generator.reply().join();
    } finally {
      generator?.close();
      tokenizer?.close();
      model.close();
    }
  }
}
