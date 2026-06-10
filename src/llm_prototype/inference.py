"""Text generation helpers and CLI for local checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_prototype.config import ModelConfig
from llm_prototype.model import TransformerLanguageModel
from llm_prototype.tokenizer import BPETokenizer, ByteTokenizer


def generate_text(
    model: TransformerLanguageModel,
    tokenizer: ByteTokenizer | BPETokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.05,
    device: str = "cpu",
) -> str:
    """Encode a prompt, generate continuation tokens, and decode to text."""

    model.to(device)
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )[0].tolist()
    return tokenizer.decode(output_ids)


def build_untrained_model(config_path: str | Path) -> TransformerLanguageModel:
    """Construct an untrained model from a JSON config for smoke tests or demos."""

    return TransformerLanguageModel(ModelConfig.from_json(config_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a local transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, help="Path to a saved model checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.json"), help="Model config for an untrained demo model.")
    parser.add_argument("--tokenizer", type=Path, help="Optional tokenizer JSON. Defaults to byte tokenizer.")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.checkpoint:
        model, _ = TransformerLanguageModel.load_checkpoint(args.checkpoint, map_location=args.device)
    else:
        model = build_untrained_model(args.config)

    tokenizer = ByteTokenizer()
    if args.tokenizer:
        tokenizer = BPETokenizer.load(args.tokenizer)

    print(generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        args.device,
    ))


if __name__ == "__main__":
    main()
