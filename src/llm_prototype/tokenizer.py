"""Tokenizer implementations for local language-model experiments.

The byte tokenizer is immediately usable without training. The BPE tokenizer is a
small, transparent implementation that can be trained later on local text only.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable


class ByteTokenizer:
    """UTF-8 byte tokenizer with three special tokens.

    Token IDs 0-255 map directly to byte values. IDs 256-258 are reserved for
    padding, beginning-of-sequence, and end-of-sequence tokens respectively.
    """

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text into token IDs."""

        tokens = list(text.encode("utf-8"))
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into text."""

        byte_values: list[int] = []
        for token_id in token_ids:
            token = int(token_id)
            if 0 <= token <= 255:
                byte_values.append(token)
            elif not skip_special_tokens:
                marker = {
                    self.pad_token_id: self.pad_token,
                    self.bos_token_id: self.bos_token,
                    self.eos_token_id: self.eos_token,
                }.get(token, f"<unk:{token}>")
                byte_values.extend(marker.encode("utf-8"))
        return bytes(byte_values).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Save tokenizer metadata."""

        data = {"type": "byte", "vocab_size": self.vocab_size}
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        """Load a byte tokenizer. The file is validated for clarity."""

        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if data.get("type") != "byte":
            raise ValueError("tokenizer file is not a byte tokenizer")
        return cls()


class BPETokenizer(ByteTokenizer):
    """Minimal byte-pair encoding tokenizer.

    The implementation starts from raw UTF-8 bytes and learns deterministic pair
    merges. It is intentionally compact and dependency-free so the full
    tokenization pipeline lives in this repository.
    """

    def __init__(self, merges: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None) -> None:
        self.merges = merges or []
        self.merge_to_id: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        self.id_to_piece: dict[int, tuple[int, ...]] = {i: (i,) for i in range(256)}
        for index, pair in enumerate(self.merges, start=259):
            merged_piece = pair[0] + pair[1]
            self.merge_to_id[pair] = index
            self.id_to_piece[index] = merged_piece
        self.vocab_size = 259 + len(self.merges)

    def train(self, texts: Iterable[str], target_vocab_size: int) -> None:
        """Learn BPE merges from local text for future tokenization."""

        if target_vocab_size < 259:
            raise ValueError("target_vocab_size must be at least 259")
        sequences = [tuple((byte,) for byte in text.encode("utf-8")) for text in texts]
        merges_needed = target_vocab_size - 259
        self.merges = []
        for _ in range(merges_needed):
            pair_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            self.merges.append(best_pair)
            merged_piece = best_pair[0] + best_pair[1]
            sequences = [self._merge_sequence(sequence, best_pair, merged_piece) for sequence in sequences]
        self.__init__(self.merges)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text using learned BPE merges."""

        pieces = tuple((byte,) for byte in text.encode("utf-8"))
        for pair in self.merges:
            pieces = self._merge_sequence(pieces, pair, pair[0] + pair[1])
        piece_to_id = {piece: token_id for token_id, piece in self.id_to_piece.items()}
        tokens = [piece_to_id[piece] for piece in pieces]
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Decode BPE token IDs back into text."""

        byte_values: list[int] = []
        for token_id in token_ids:
            token = int(token_id)
            if token in self.id_to_piece:
                byte_values.extend(self.id_to_piece[token])
            elif not skip_special_tokens:
                marker = {
                    self.pad_token_id: self.pad_token,
                    self.bos_token_id: self.bos_token,
                    self.eos_token_id: self.eos_token,
                }.get(token, f"<unk:{token}>")
                byte_values.extend(marker.encode("utf-8"))
        return bytes(byte_values).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Save BPE merges and special-token metadata."""

        data = {
            "type": "bpe",
            "merges": [[list(left), list(right)] for left, right in self.merges],
            "special_tokens": {
                "pad_token_id": self.pad_token_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
            },
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a BPE tokenizer from disk."""

        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if data.get("type") != "bpe":
            raise ValueError("tokenizer file is not a BPE tokenizer")
        merges = [(tuple(left), tuple(right)) for left, right in data["merges"]]
        return cls(merges=merges)

    @staticmethod
    def _merge_sequence(
        sequence: tuple[tuple[int, ...], ...],
        pair: tuple[tuple[int, ...], tuple[int, ...]],
        merged_piece: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        merged: list[tuple[int, ...]] = []
        index = 0
        while index < len(sequence):
            if index < len(sequence) - 1 and (sequence[index], sequence[index + 1]) == pair:
                merged.append(merged_piece)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return tuple(merged)
