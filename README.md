# LLM-Prototype

A local, educational large-language-model prototype implemented from scratch with PyTorch building blocks. It does **not** call OpenAI or any external AI service and does **not** require an API key.

## Project structure

```text
configs/
  tiny.json                  # Example model-size/settings file
examples/
  generate.py                # Minimal local generation example
src/llm_prototype/
  config.py                  # ModelConfig JSON serialization
  tokenizer.py               # Byte tokenizer and trainable BPE tokenizer
  model.py                   # Transformer, attention, feed-forward, checkpoints
  inference.py               # Text generation helper and CLI
tests/
  test_llm_prototype.py      # Smoke tests for tokenizer/model/checkpoint flow
```

## Included components

- Tokenizer system:
  - `ByteTokenizer` works immediately with UTF-8 bytes.
  - `BPETokenizer` can learn byte-pair merges later from local text.
- Decoder-only transformer architecture.
- Causal multi-head self-attention.
- Transformer feed-forward layers.
- Learned token and positional embeddings.
- Autoregressive inference/text generation.
- Save/load checkpoint support with model config and optional optimizer state.
- JSON configuration file for model size and settings.

## Quick start

Install dependencies locally:

```bash
python -m pip install -e .
```

Run a smoke-test generation with random, untrained weights:

```bash
python -m llm_prototype.inference --config configs/tiny.json --prompt "Hello" --max-new-tokens 20
```

Because no training is included here, an untrained model will produce random text. To generate meaningful text, train weights locally in a separate training workflow and save a checkpoint with `TransformerLanguageModel.save_checkpoint(...)`.
