// A minimal Flutter app on package:rayo: pick a model file, type a prompt,
// get the reply. See README.md.
import 'package:flutter/material.dart';

import 'runner.dart';

/// The model to start with: `flutter run --dart-define=RAYO_MODEL=/path`.
const defaultModel = String.fromEnvironment('RAYO_MODEL');

void main() => runApp(const ChatApp(runner: RayoPromptRunner()));

class ChatApp extends StatelessWidget {
  const ChatApp({super.key, required this.runner});

  final PromptRunner runner;

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'rayo chat',
    theme: ThemeData(colorSchemeSeed: Colors.teal),
    darkTheme: ThemeData(
      colorSchemeSeed: Colors.teal,
      brightness: Brightness.dark,
    ),
    home: PromptPage(runner: runner),
  );
}

class PromptPage extends StatefulWidget {
  const PromptPage({super.key, required this.runner});

  final PromptRunner runner;

  @override
  State<PromptPage> createState() => _PromptPageState();
}

class _PromptPageState extends State<PromptPage> {
  final _model = TextEditingController(text: defaultModel);
  final _prompt = TextEditingController(text: 'What is a budget?');
  String _reply = '';
  String? _error;
  bool _busy = false;

  Future<void> _run() async {
    setState(() {
      _busy = true;
      _error = null;
      _reply = '';
    });
    try {
      final reply = await widget.runner.run(_model.text.trim(), _prompt.text);
      if (mounted) setState(() => _reply = reply);
    } on Object catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  void dispose() {
    _model.dispose();
    _prompt.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('rayo chat')),
    body: ListView(
      padding: const EdgeInsets.all(16),
      children: [
        TextField(
          key: const Key('model'),
          controller: _model,
          decoration: const InputDecoration(
            labelText: 'Model file (.weights)',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 12),
        TextField(
          key: const Key('prompt'),
          controller: _prompt,
          minLines: 2,
          maxLines: 5,
          decoration: const InputDecoration(
            labelText: 'Prompt',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 12),
        FilledButton(
          key: const Key('run'),
          onPressed: _busy ? null : _run,
          child: Text(_busy ? 'Running…' : 'Run'),
        ),
        const SizedBox(height: 16),
        if (_busy) const LinearProgressIndicator(),
        if (_error != null)
          Text(
            _error!,
            key: const Key('error'),
            style: TextStyle(color: Theme.of(context).colorScheme.error),
          ),
        SelectableText(_reply, key: const Key('reply')),
      ],
    ),
  );
}
