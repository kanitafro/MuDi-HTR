"""Offline data utilities for CRNN + CTC training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class CTCTextEncoder:
    """Character-level encoder for CTC with reserved blank and unknown tokens."""

    chars: str
    blank_token: str = "<BLANK>"
    unknown_token: str = "<UNK>"

    def __post_init__(self) -> None:
        object.__setattr__(self, "vocab", [self.blank_token, self.unknown_token] + list(self.chars))
        object.__setattr__(self, "char_to_idx", {char: idx for idx, char in enumerate(self.vocab)})
        object.__setattr__(self, "idx_to_char", {idx: char for idx, char in enumerate(self.vocab)})
        object.__setattr__(self, "blank_index", 0)
        object.__setattr__(self, "unknown_index", 1)

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CTCTextEncoder":
        """Build a stable character vocabulary from dataset texts."""
        charset = sorted({char for text in texts for char in text})
        return cls("".join(charset))

    @property
    def num_classes(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx.get(char, self.unknown_index) for char in text]

    def decode(self, token_ids: Iterable[int], collapse_repeats: bool = True) -> str:
        """Decode token ids back to text; optionally apply CTC repeat/blank collapse."""
        output_chars: list[str] = []
        last_id: int | None = None
        for token_id in token_ids:
            if collapse_repeats and token_id == last_id:
                continue
            last_id = token_id
            if token_id == self.blank_index:
                continue
            output_chars.append(self.idx_to_char.get(token_id, self.unknown_token))
        return "".join(output_chars).replace(self.unknown_token, "")


class OfflineHandwritingDataset(Dataset):
    """Dataset for processed offline handwriting samples stored as .pt files."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        dataset_name: str,
        text_encoder: CTCTextEncoder | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.dataset_name = dataset_name
        self.text_encoder = text_encoder

        safe_name = dataset_name.replace("/", "_")
        self.split_dir = self.root_dir / safe_name / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        paths = sorted(self.split_dir.glob("sample_*.pt"))
        if max_samples is not None:
            paths = paths[:max_samples]
        self.sample_paths = paths

        if not self.sample_paths:
            raise FileNotFoundError(f"No processed samples found in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def _load_sample(self, path: Path) -> dict:
        sample = torch.load(path, map_location="cpu")
        image = sample["image"]
        if image.ndim == 2:
            image = image.unsqueeze(0)

        text = str(sample.get("text", ""))
        encoded = self.text_encoder.encode(text) if self.text_encoder else []

        return {
            "image": image.float(),
            "text": text,
            "encoded_text": encoded,
            "source_path": sample.get("source_path", ""),
            "dataset": sample.get("dataset", self.dataset_name),
            "split": sample.get("split", self.split),
        }

    def __getitem__(self, index: int) -> dict:
        return self._load_sample(self.sample_paths[index])

    @classmethod
    def build_text_encoder(
        cls,
        root_dir: str | Path,
        split: str,
        dataset_name: str,
        max_samples: int | None = None,
    ) -> CTCTextEncoder:
        """Scan a processed split and build a text encoder from all labels."""
        safe_name = dataset_name.replace("/", "_")
        split_dir = Path(root_dir) / safe_name / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        sample_paths = sorted(split_dir.glob("sample_*.pt"))
        if max_samples is not None:
            sample_paths = sample_paths[:max_samples]

        texts: list[str] = []
        for sample_path in sample_paths:
            sample = torch.load(sample_path, map_location="cpu")
            texts.append(str(sample.get("text", "")))
        return CTCTextEncoder.from_texts(texts)


def ctc_collate_fn(batch: list[dict]) -> dict:
    """Collate function for CRNN/CTC training."""
    images = torch.stack([item["image"] for item in batch], dim=0)

    encoded_sequences = [item["encoded_text"] for item in batch]
    target_lengths = torch.tensor([len(seq) for seq in encoded_sequences], dtype=torch.long)
    flat_targets = [token for seq in encoded_sequences for token in seq]
    targets = torch.tensor(flat_targets, dtype=torch.long) if flat_targets else torch.empty(0, dtype=torch.long)

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": [item["text"] for item in batch],
        "sources": [item["source_path"] for item in batch],
    }


def create_offline_dataloader(
    root_dir: str | Path,
    split: str,
    dataset_name: str,
    text_encoder: CTCTextEncoder,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: int | None = None,
) -> DataLoader:
    """Create a dataloader for one processed offline split."""
    dataset = OfflineHandwritingDataset(
        root_dir=root_dir,
        split=split,
        dataset_name=dataset_name,
        text_encoder=text_encoder,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
    )
