# LLM-Prototype

A local, educational large-language-model prototype implemented from scratch with PyTorch building blocks. It does **not** call OpenAI or any external AI service and does **not** require an API key.

## Quick start in Codespaces

After opening the Codespace, the only commands you need are:

```bash
git pull
npm start
```

The app starts a local browser interface at `http://localhost:3000`. GitHub Codespaces will forward port `3000` and open the interface automatically when possible.

## Browser interface

The included web app provides:

- Clean main app interface for local LLM experiments.
- Text input box for prompts.
- Generate button.
- Output/response area.
- Model settings panel.
- Temperature setting.
- Max tokens setting.
- Clear output button.
- Console/error display.
- Save/load interface checkpoint controls.
- Optional fields for local model checkpoint and tokenizer paths.

The default **Interface demo** runtime is intentionally dependency-free so the interface can be used immediately with `npm start`. It returns local placeholder text and never contacts an external AI service.

If Python and PyTorch are installed, choose **Local Python transformer** in the Runtime selector to call the repository's bundled untrained transformer through `llm_prototype.inference`. Because training is not included here, untrained model output is random and only useful for smoke testing the code path.

## Project structure

```text
.devcontainer/
  devcontainer.json          # Codespace setup and port forwarding
configs/
  tiny.json                  # Example model-size/settings file
examples/
  generate.py                # Minimal local generation example
public/
  index.html                 # Browser interface markup
  styles.css                 # Browser interface styles
  app.js                     # Browser interface behavior
server.js                    # Dependency-free Node server and local API routes
package.json                 # npm start script
tests/
  test_llm_prototype.py      # Smoke tests for tokenizer/model/checkpoint flow
src/llm_prototype/
  config.py                  # ModelConfig JSON serialization
  tokenizer.py               # Byte tokenizer and trainable BPE tokenizer
  model.py                   # Transformer, attention, feed-forward, checkpoints
  inference.py               # Text generation helper and CLI
```

## Included model components

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

## Python usage

Install Python dependencies locally if you want to run the model code directly:

```bash
python -m pip install -e .
```

Run a smoke-test generation with random, untrained weights:

```bash
python -m llm_prototype.inference --config configs/tiny.json --prompt "Hello" --max-new-tokens 20
```

Because no training is included here, an untrained model will produce random text. To generate meaningful text, train weights locally in a separate training workflow and save a checkpoint with `TransformerLanguageModel.save_checkpoint(...)`.
