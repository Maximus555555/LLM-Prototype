# LLM-Prototype

A local, educational large-language-model prototype implemented from scratch with PyTorch building blocks and a browser-based local conversation interface. It does **not** call OpenAI or any external AI service, does **not** require an API key, and does **not** add paid services.

## Quick start in Codespaces

After opening the Codespace, the only commands you need are:

```bash
git pull
npm start
```

The app starts a local browser interface at `http://localhost:3000`. GitHub Codespaces will forward port `3000` and open the interface automatically when possible.

## Local conversation interface

The included web app provides:

- Chat-style message history.
- User message input box.
- Send button.
- AI response area for the latest local response.
- Basic memory within the current conversation.
- A local knowledge folder at `local_knowledge/`.
- A simple ingestion system that loads `.txt`, `.md`, and `.json` files from `local_knowledge/` recursively.
- A retrieval system that searches local content before responding.
- Fallback responses when the local files do not contain enough matching information.
- A local-only design: no OpenAI calls, no external AI service calls, no API keys, and no paid service dependencies.

The default chat experience is a **local retrieval-and-rules assistant**, not a hosted trained LLM. It uses current browser chat history, local file search, and response templates to feel conversational while staying transparent about its limits.

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

## Project structure

```text
configs/
  tiny.json                  # Example model-size/settings file
examples/
  generate.py                # Minimal local generation example
local_knowledge/
  *.md, *.txt, *.json        # Local files ingested by the chat retrieval system
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
