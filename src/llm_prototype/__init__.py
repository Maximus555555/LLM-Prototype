"""Educational large language model prototype package."""

from llm_prototype.config import ModelConfig
from llm_prototype.tokenizer import BPETokenizer, ByteTokenizer

__all__ = [
    "BPETokenizer",
    "ByteTokenizer",
    "ModelConfig",
    "TransformerLanguageModel",
]


def __getattr__(name: str):
    if name == "TransformerLanguageModel":
        from llm_prototype.model import TransformerLanguageModel

        return TransformerLanguageModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
