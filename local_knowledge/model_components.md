# Bundled Model Components

The repository includes an educational decoder-only transformer implementation in Python. The model code contains token and positional embeddings, causal self-attention, feed-forward layers, checkpoint helpers, and an autoregressive inference command.

The byte tokenizer works immediately with UTF-8 bytes. A trainable byte-pair tokenizer is also included for future local training workflows. The tiny JSON config is intended for smoke tests and small local experiments, not production inference quality.

Because the repository does not include trained weights, the local Python transformer output is random unless the user supplies locally trained checkpoints. The app should never imply that random untrained output is knowledgeable. For conversational behavior, prefer retrieval over local_knowledge files and rules-based fallback text.
