from pathlib import Path

import pytest

from llm_prototype.config import ModelConfig
from llm_prototype.tokenizer import BPETokenizer, ByteTokenizer


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=259,
        max_sequence_length=16,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=64,
        dropout=0.0,
    )


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "hello 🌍"
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokenizer.decode(tokens) == text


def test_bpe_tokenizer_round_trip(tmp_path: Path) -> None:
    tokenizer = BPETokenizer()
    tokenizer.train(["banana bandana", "banana banana"], target_vocab_size=270)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.decode(loaded.encode("banana")) == "banana"


def test_model_forward_generate_and_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from llm_prototype.model import TransformerLanguageModel

    model = TransformerLanguageModel(tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    output = model(input_ids, targets=input_ids)
    assert output["logits"].shape == (2, 8, model.config.vocab_size)
    assert output["loss"] is not None

    generated = model.generate(input_ids[:1, :4], max_new_tokens=3, top_k=10)
    assert generated.shape == (1, 7)

    checkpoint_path = tmp_path / "model.pt"
    model.save_checkpoint(checkpoint_path, step=123)
    loaded, checkpoint = TransformerLanguageModel.load_checkpoint(checkpoint_path)
    assert checkpoint["step"] == 123
    assert loaded.config.to_dict() == model.config.to_dict()
