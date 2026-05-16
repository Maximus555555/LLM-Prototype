"""Configuration objects for the transformer language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    """Hyperparameters that define a decoder-only transformer language model."""

    vocab_size: int = 259
    max_sequence_length: int = 256
    embedding_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    feed_forward_dim: int = 1536
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    tie_token_embeddings: bool = True
    pad_token_id: int = 256
    bos_token_id: int = 257
    eos_token_id: int = 258

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1)")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Build a config from a dictionary."""

        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """Load a model configuration from a JSON file."""

        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write this configuration to a JSON file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)
            file.write("\n")
