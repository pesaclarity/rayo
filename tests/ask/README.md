# The Ask models, for rayo's golden tests

The Pesa Clarity app's Ask reader and writer, as rayo runs them, and what TFLite gave for them:

| File | From |
| --- | --- |
| `ask-reader.weights` | pesaclarity-app `assets/models/ask/ask-reader.weights` (PCWT 1, read out of the released `ask-reader.tflite`, sha256 `f6652fa7…`), with the graph pesaclarity-ml's `akili.ask.reader_weights.rayo_graph` writes |
| `ask-writer.weights` | `akili ask-writer-weights` on the released `ask-writer.tflite` (sha256 `3461c243…`) |
| `ask_reader_decodes.json` | pesaclarity-app `test/fixtures/ask/`: TFLite's outputs for 30 questions |
| `ask_writer_decodes.json` | pesaclarity-app `test/fixtures/ask/`: TFLite's sampled replies to the app's requests |
| `ask_writer_tflite.json` | TFLite (XNNPACK, two threads, as the recordings) fed each recorded reply token by token: the next-token argmax and its logit at every step |

`tests/test_ask.c` checks rayo against them: the reader's every label and probabilities within
1e-4, the writer's next-token choice at every step and its top logit within 0.2. The programs in
`examples/ask/` time them.

The weights are the Pesa Clarity Ask reader and writer, by the Pesa Clarity project, licensed
under CC BY-SA 4.0 (pesaclarity-ml `MODEL-WEIGHTS.md` and `LICENSE-WEIGHTS`), trained on
synthetic data only; they are copied here unchanged apart from the graph field.
