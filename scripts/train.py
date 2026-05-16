#!/usr/bin/env python3
"""Train the local transformer on files in local_training_data/.

This script uses only local files and the repository's tokenizer/model code. It
trains the decoder-only transformer for next-token prediction and writes PyTorch
checkpoints under checkpoints/.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from llm_prototype.config import ModelConfig
from llm_prototype.model import TransformerLanguageModel
from llm_prototype.tokenizer import ByteTokenizer

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}


@dataclass(slots=True)
class TokenSplits:
    train: torch.Tensor
    validation: torch.Tensor
    files: list[Path]
    character_count: int


def read_json_as_text(path: Path) -> str:
    """Read JSON as stable text while accepting imperfect local data files."""

    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)


def iter_training_files(data_dir: Path) -> Iterable[Path]:
    """Yield supported local training files recursively."""

    if not data_dir.exists():
        return []
    return sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_local_text(data_dir: Path, extra_files: Iterable[Path] = ()) -> tuple[str, list[Path], int]:
    """Load and combine supported local training files plus optional ingested JSON exports."""

    files = list(iter_training_files(data_dir))
    for extra_file in extra_files:
        if extra_file.exists() and extra_file.is_file() and extra_file.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(extra_file)
    files = sorted(dict.fromkeys(files))
    sections: list[str] = []
    character_count = 0
    for path in files:
        if path.suffix.lower() == ".json":
            text = read_json_as_text(path)
        else:
            text = path.read_text(encoding="utf-8")
        text = text.strip()
        if not text:
            continue
        character_count += len(text)
        sections.append(f"\n\n<file:{path.as_posix()}>\n{text}\n")
    return "".join(sections), files, character_count


def build_token_splits(
    data_dir: Path,
    tokenizer: ByteTokenizer,
    validation_fraction: float,
    seed: int,
    extra_files: Iterable[Path] = (),
) -> TokenSplits:
    """Tokenize local text and create deterministic train/validation splits."""

    text, files, character_count = load_local_text(data_dir, extra_files)
    if not files:
        raise ValueError(f"No .txt, .md, or .json files found under {data_dir}/.")
    validation_fraction = min(max(validation_fraction, 0.01), 0.5)
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(tokens) < 4:
        raise ValueError("Training data is too small; add more local text before training.")

    rng = random.Random(seed)
    if len(tokens) > 1:
        # Keep contiguous language-model text but make the split point deterministic.
        min_validation = max(1, int(len(tokens) * validation_fraction))
        min_train = max(2, len(tokens) - min_validation)
        split = min_train
        if len(tokens) >= 20:
            jitter = max(0, int(len(tokens) * 0.02))
            split = max(2, min(len(tokens) - 1, split + rng.randint(-jitter, jitter)))
    else:
        split = 1

    train_ids = torch.tensor(tokens[:split], dtype=torch.long)
    validation_ids = torch.tensor(tokens[split:], dtype=torch.long)
    if validation_ids.numel() < 2:
        validation_ids = train_ids[-min(train_ids.numel(), max(2, validation_ids.numel() + 1)) :].clone()
    return TokenSplits(train=train_ids, validation=validation_ids, files=files, character_count=character_count)


def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a next-token prediction batch from a token tensor."""

    if tokens.numel() < block_size + 1:
        repeats = math.ceil((block_size + 1) / max(1, tokens.numel())) + 1
        tokens = tokens.repeat(repeats)
    max_start = tokens.numel() - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([tokens[start : start + block_size] for start in starts]).to(device)
    y = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts]).to(device)
    return x, y


@torch.no_grad()
def evaluate(
    model: TransformerLanguageModel,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_batches: int,
    device: str,
) -> tuple[float, float]:
    """Estimate train and validation losses."""

    model.eval()
    losses: dict[str, list[float]] = {"train": [], "validation": []}
    for split_name, tokens in (("train", train_tokens), ("validation", validation_tokens)):
        for _ in range(eval_batches):
            x, y = get_batch(tokens, batch_size, block_size, device)
            loss = model(x, targets=y)["loss"]
            assert loss is not None
            losses[split_name].append(float(loss.item()))
    model.train()
    return sum(losses["train"]) / len(losses["train"]), sum(losses["validation"]) / len(losses["validation"])


def save_training_checkpoint(
    model: TransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
    train_loss: float | None,
    validation_loss: float | None,
    extra: dict[str, object],
) -> Path:
    """Save numbered and latest PyTorch checkpoints."""

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"step-{step:06d}.pt"
    checkpoint_extra = {
        **extra,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "checkpoint_kind": "local-training",
    }
    model.save_checkpoint(checkpoint_path, optimizer=optimizer, step=step, extra=checkpoint_extra)
    latest_path = output_dir / "latest.pt"
    shutil.copyfile(checkpoint_path, latest_path)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local transformer on local_training_data/ files.")
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.json"), help="Model config JSON.")
    parser.add_argument("--data-dir", type=Path, default=Path("local_training_data"), help="Local training data directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Checkpoint output directory.")
    parser.add_argument("--web-knowledge-file", type=Path, default=Path("data/web_knowledge.json"), help="Optional local JSON file written by no-key internet ingestion; included in future training when present.")
    parser.add_argument("--resume", type=Path, help="Optional checkpoint to resume from, for example checkpoints/latest.pt.")
    parser.add_argument("--no-auto-resume", action="store_true", help="Do not automatically resume from output-dir/latest.pt when it exists.")
    parser.add_argument("--watch", action="store_true", help="Keep training forever, periodically reloading local training data so new files are included in future batches.")
    parser.add_argument("--reload-data-interval", type=float, default=5.0, help="Seconds to wait between continuous training cycles or while waiting for training files.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=200, help="Number of optimizer steps to run; when resuming, these are additional steps.")
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.batch_size <= 0 or args.max_steps <= 0:
        raise ValueError("batch size and max steps must be positive")
    if args.reload_data_interval < 0:
        raise ValueError("reload data interval must be non-negative")
    if args.eval_interval <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("eval interval and checkpoint interval must be positive")

    torch.manual_seed(args.seed)
    tokenizer = ByteTokenizer()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    latest_checkpoint = args.output_dir / "latest.pt"
    resume_path = args.resume
    if resume_path is None and not args.no_auto_resume and latest_checkpoint.exists():
        resume_path = latest_checkpoint

    if resume_path:
        model, checkpoint = TransformerLanguageModel.load_checkpoint(resume_path, map_location=device)
        start_step = int(checkpoint.get("step", 0))
        print(f"Resuming local transformer training from {resume_path.as_posix()} at step {start_step}.", flush=True)
    else:
        model = TransformerLanguageModel(ModelConfig.from_json(args.config))
        checkpoint = {}
        start_step = 0
        print("Starting local transformer training from fresh random weights.", flush=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(
        "WARNING: local weight training may be slow, especially on CPU. Quality depends on "
        "the amount of data, model size, training time, and CPU/GPU hardware. This will not "
        "produce a ChatGPT-quality model.",
        flush=True,
    )

    current_step = start_step
    last_train_loss: float | None = None
    last_validation_loss: float | None = None
    block_size = model.config.max_sequence_length

    while True:
        try:
            extra_files = [args.web_knowledge_file] if args.web_knowledge_file else []
            splits = build_token_splits(args.data_dir, tokenizer, args.validation_fraction, args.seed + current_step, extra_files)
        except ValueError as error:
            if not args.watch:
                raise
            print(f"Waiting for local training data in {args.data_dir.as_posix()}/: {error}", flush=True)
            time.sleep(args.reload_data_interval)
            continue

        print(
            "Loaded local training data: "
            f"files={len(splits.files)} characters={splits.character_count} "
            f"train_tokens={splits.train.numel()} validation_tokens={splits.validation.numel()} device={device}",
            flush=True,
        )

        extra = {
            "data_dir": args.data_dir.as_posix(),
            "web_knowledge_file": args.web_knowledge_file.as_posix() if args.web_knowledge_file else None,
            "config_path": args.config.as_posix(),
            "files": [path.as_posix() for path in splits.files],
            "character_count": splits.character_count,
            "tokenizer": "byte",
            "no_external_ai_services": True,
            "continuous_training": bool(args.watch),
            "warning": "Local training can be slow; quality depends on data, model size, training time, and hardware.",
        }

        target_step = current_step + args.max_steps

        for step in range(current_step + 1, target_step + 1):
            x, y = get_batch(splits.train, args.batch_size, block_size, device)
            output = model(x, targets=y)
            loss = output["loss"]
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            last_train_loss = float(loss.item())
            current_step = step

            if step == start_step + 1 or step % args.eval_interval == 0 or step == target_step:
                last_train_loss, last_validation_loss = evaluate(
                    model,
                    splits.train,
                    splits.validation,
                    args.batch_size,
                    block_size,
                    args.eval_batches,
                    device,
                )
                print(f"step {step}: train_loss={last_train_loss:.4f} validation_loss={last_validation_loss:.4f}", flush=True)
            else:
                print(f"step {step}: train_loss={last_train_loss:.4f}", flush=True)

            if step % args.checkpoint_interval == 0 or step == target_step:
                checkpoint_path = save_training_checkpoint(
                    model,
                    optimizer,
                    args.output_dir,
                    step,
                    last_train_loss,
                    last_validation_loss,
                    extra,
                )
                print(f"saved checkpoint {checkpoint_path.as_posix()} and {args.output_dir / 'latest.pt'}", flush=True)

        if last_validation_loss is None:
            last_train_loss, last_validation_loss = evaluate(
                model,
                splits.train,
                splits.validation,
                args.batch_size,
                block_size,
                args.eval_batches,
                device,
            )
        save_training_checkpoint(model, optimizer, args.output_dir, current_step, last_train_loss, last_validation_loss, extra)
        print(
            "Training cycle complete. "
            f"latest_checkpoint={(args.output_dir / 'latest.pt').as_posix()} "
            f"train_loss={last_train_loss:.4f} validation_loss={last_validation_loss:.4f}",
            flush=True,
        )

        if not args.watch:
            break
        time.sleep(args.reload_data_interval)


if __name__ == "__main__":
    main()
