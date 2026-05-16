"""Decoder-only transformer language model implemented with PyTorch modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from llm_prototype.config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.query_key_value = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        causal_mask = torch.tril(torch.ones(config.max_sequence_length, config.max_sequence_length))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.max_sequence_length, config.max_sequence_length))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, embedding_dim = x.shape
        qkv = self.query_key_value(x)
        query, key, value = qkv.split(embedding_dim, dim=2)
        query = self._shape_for_heads(query, batch_size, sequence_length)
        key = self._shape_for_heads(key, batch_size, sequence_length)
        value = self._shape_for_heads(value, batch_size, sequence_length)

        scores = query @ key.transpose(-2, -1) * (self.head_dim**-0.5)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        return self.residual_dropout(self.output_projection(attended))

    def _shape_for_heads(self, tensor: Tensor, batch_size: int, sequence_length: int) -> Tensor:
        return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)


class FeedForward(nn.Module):
    """Transformer position-wise feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal attention and feed-forward layers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.attention = MultiHeadSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class TransformerLanguageModel(nn.Module):
    """GPT-style language model for local inference and experimentation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embeddings = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.tie_token_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        self.apply(self._init_weights)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> dict[str, Tensor | None]:
        """Run a forward pass and optionally compute cross-entropy loss."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        positions = torch.arange(0, sequence_length, device=input_ids.device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Autoregressively generate token IDs from a prompt."""

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        generated = input_ids
        self.eval()
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_sequence_length :]
            logits = self(context)["logits"][:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], torch.finfo(logits.dtype).min)
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return generated

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save model state, config, and optional optimizer state."""

        checkpoint: dict[str, Any] = {
            "config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
            "step": step,
            "extra": extra or {},
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple["TransformerLanguageModel", dict[str, Any]]:
        """Load a checkpoint and return the model plus checkpoint metadata."""

        checkpoint = torch.load(path, map_location=map_location)
        model = cls(ModelConfig.from_dict(checkpoint["config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
