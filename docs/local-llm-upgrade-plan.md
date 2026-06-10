# Local LLM Capability Upgrade Plan

This project is intentionally a local-first educational prototype, not a Claude-class assistant. The browser chat path currently combines deterministic rules, short-term conversation memory, local-file retrieval, optional no-key web retrieval, and template-based answers. That makes it useful for transparent demos and grounded local knowledge lookup, but it does not provide the learned reasoning, instruction following, tool planning, coding depth, or broad world knowledge associated with frontier assistants.

## Why the current project is not Claude-class

1. **The default chat runtime is retrieval plus rules, not neural generation.** The `/api/chat` path returns responses from `createLocalChatResponseAsync`, which delegates to article handling, explicit internet-search handling, user-provided memory storage, arithmetic/basic conversation rules, local knowledge search, and fallback templates.
2. **The visible generator is a placeholder unless a local Python model is selected.** The demo generator creates deterministic text fragments and explicitly labels itself as placeholder output.
3. **The bundled transformer is tiny and starts untrained.** The default config is a small decoder-only model with a 128-token context window, 128-dimensional embeddings, 4 layers, and 4 heads. That is appropriate for tests and education, but not for modern assistant quality.
4. **No pretrained weights are included.** The Python inference path builds an untrained model when no checkpoint is supplied. Randomly initialized weights do not contain language, facts, coding skill, or instruction-following behavior.
5. **The local trainer is basic next-token training.** It trains on local `.txt`, `.md`, and `.json` files and saves checkpoints, but it is not a full pretraining, supervised fine-tuning, preference optimization, safety, evaluation, or serving pipeline.
6. **The current memory is retrieval memory, not model learning.** Persisted user/web knowledge improves future retrieval, but it does not update model weights unless that data is later used in a separate training run.

## What must change without using OpenAI, Anthropic, or other hosted AI APIs

The practical path is not to train a frontier model from scratch. Instead, keep the application local and replace or augment the tiny in-repo generator with an open-weight, instruction-tuned model running on the user's machine.

### Recommended architecture

1. **Add a local model runtime adapter.** Support one or more local inference engines such as llama.cpp, Ollama, vLLM, MLX, or Hugging Face Transformers. The server should call `localhost` only, or spawn a local process, so no hosted AI API is required.
2. **Use open-weight instruction models.** Let the user configure a local model path or local model name. The project should not ship large weights in git; document how to download them separately and require users to verify licenses.
3. **Replace `/api/chat` generation behavior.** Keep the existing retrieval layer, but pass the retrieved snippets, conversation history, system instructions, and the user's message into the local model instead of returning mostly templates.
4. **Add prompt construction and context management.** Build a structured prompt with system policy, local knowledge citations, recent conversation turns, and tool results. Add truncation/summarization for long contexts.
5. **Add streaming output.** For a Claude-like feel, stream tokens from the local runtime to the browser with server-sent events or WebSockets.
6. **Add model management settings.** Expose model path/name, context length, GPU/CPU mode, quantization, temperature, top-p/top-k, max tokens, and timeout in the UI and server config.
7. **Keep retrieval-augmented generation.** Continue using `local_knowledge/` and `data/web_knowledge.json`, but improve ranking with embeddings or a stronger local retriever when possible.
8. **Add local tool use.** If you want agentic behavior, define tools for file search, webpage retrieval, code execution, calculator, and knowledge storage, then implement a local tool-calling loop around the model.
9. **Add evaluation.** Create local benchmark prompts for instruction following, coding, math, refusal/safety boundaries, retrieval faithfulness, and latency. Track regressions before changing models or prompts.

### Minimum viable implementation plan

1. Add a `local-llm` runtime option to the server.
2. Create a runtime interface such as `generateChatCompletion({ messages, settings, signal })`.
3. Implement an Ollama adapter first for ease of setup, or a llama.cpp adapter first for direct GGUF-file control.
4. Change `/api/chat` so it retrieves local context first, then calls the configured local model with that context.
5. Stream the response to the frontend.
6. Add settings UI for local model host/path/name and generation parameters.
7. Document hardware expectations and model-license requirements.
8. Add tests that mock the local runtime so CI does not need a large model.

## Hardware reality

A small quantized local model can run on CPU but will be slow and significantly weaker than frontier hosted systems. Larger models need substantial RAM/VRAM. The application can become much more capable without hosted APIs, but matching Claude-class quality generally requires very large models, expensive hardware, extensive training data, high-quality post-training, and careful evaluation.
