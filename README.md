# LLM-Prototype

A local, educational large-language-model prototype implemented from scratch with PyTorch building blocks and a browser-based local conversation interface. It does **not** call OpenAI or any external AI service, does **not** require an API key, and does **not** add paid services.

## Quick start in Codespaces

After opening the Codespace, the normal app still runs with:

```bash
git pull
npm start
```

Local training is a separate feature and runs with:

```bash
npm run train
```

The app starts a local browser interface at `http://localhost:3000`. GitHub Codespaces will forward port `3000` and open the interface automatically when possible.

## Local conversation interface

The included web app provides:

- Chat-style message history.
- User message input box.
- Send button.
- AI response area for the latest local response.
- Basic memory within the current conversation.
- Built-in handling for greetings, conversational check-ins, capability questions, thanks, simple memory prompts, and arithmetic expressions.
- A local knowledge folder at `local_knowledge/`.
- A simple ingestion system that loads `.txt`, `.md`, and `.json` files from `local_knowledge/` recursively.
- A retrieval system that searches local content before responding.
- Fallback responses when the local files do not contain enough matching information.
- A local-only design: no OpenAI calls, no external AI service calls, no API keys, and no paid service dependencies.

The default chat experience is a **local retrieval-and-rules assistant**, not a hosted trained LLM. It uses current browser chat history, local file search, arithmetic parsing, and response templates to feel conversational while staying transparent about its limits.

## Feeding local content into the app

Add generated or local-only content to the `local_knowledge/` folder. Supported file extensions are:

- `.txt`
- `.md`
- `.json`

When you click **Refresh knowledge** or send a new chat message, the Node server reads the folder, chunks the files, tokenizes the chunks, ranks matches against the user message, and includes local source paths in the response.

Do not scrape the internet for this app. If you want more content, manually create or copy local/generated notes into `local_knowledge/`.

## Optional local model generator

The interface keeps the existing local generator tools available in a secondary panel:

- **Interface demo**: dependency-free deterministic local placeholder output.
- **Local Python transformer**: calls the repository's bundled transformer through `llm_prototype.inference` when Python and PyTorch are installed.

Because no trained weights are included, the bundled transformer is not a knowledgeable conversational model by default. Its output is random unless you supply locally trained checkpoints. For useful conversation, use the local chat and retrieval system.


## Local training pipeline

Training is local-only. It does not connect to OpenAI, does not call external AI services, and does not require an API key.

1. Put training files under `local_training_data/`. The trainer loads `.txt`, `.md`, and `.json` files recursively.
2. Run `npm run train` or use the browser **Training** panel.
3. The script tokenizes the combined local text with the existing tokenizer, creates train/validation splits, trains the existing transformer for next-token prediction, prints train/validation loss, clips gradients, and saves PyTorch checkpoints under `checkpoints/`.
4. The latest trained model is saved as `checkpoints/latest.pt`.
5. In the browser, select `latest.pt` from the checkpoint list to set `checkpoints/latest.pt` as the Local Python transformer checkpoint for generation.

You can configure training from the CLI, for example:

```bash
PYTHONPATH=src python3 scripts/train.py --batch-size 4 --learning-rate 0.0003 --max-steps 500 --eval-interval 50 --checkpoint-interval 100
```

Resume from a checkpoint with:

```bash
PYTHONPATH=src python3 scripts/train.py --resume checkpoints/latest.pt
```

This training path can improve the tiny local model relative to random weights, but it does **not** create a ChatGPT-quality model and does **not** make the model intelligent. Quality depends on the amount of local data, model size, training time, and available CPU/GPU. If no `checkpoints/latest.pt` exists, the Python transformer should be considered untrained/random.

## Project structure

```text
configs/
  tiny.json                  # Example model-size/settings file
examples/
  generate.py                # Minimal local generation example
local_knowledge/
  *.md, *.txt, *.json        # Local files ingested by the chat retrieval system
local_training_data/
  *.md, *.txt, *.json        # Local files used by scripts/train.py
public/
  index.html                 # Browser interface markup
  styles.css                 # Browser interface styles
  app.js                     # Browser interface behavior
server.js                    # Dependency-free Node server and local API routes
package.json                 # npm start and npm run train scripts
scripts/
  train.py                    # Local next-token training pipeline
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

Without a trained checkpoint, the model will produce random text. To use trained local weights, run `npm run train` to create `checkpoints/latest.pt`, or save a checkpoint programmatically with `TransformerLanguageModel.save_checkpoint(...)`.
