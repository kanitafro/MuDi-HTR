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
            # ✅ Validate token_id is in valid range
            if not (0 <= token_id < len(self.vocab)):
                raise ValueError(f"Token ID {token_id} out of range [0, {len(self.vocab)-1}]")
            if collapse_repeats and token_id == last_id:
                continue
            last_id = token_id
            if token_id == self.blank_index:
                continue
            char = self.idx_to_char[token_id]
            if char != self.unknown_token:  # ✅ Skip unks, don't replace later
                output_chars.append(char)
        return "".join(output_chars)


class OfflineHandwritingDataset(Dataset):
    """Dataset for processed offline handwriting samples stored as .pt files."""

    def __init__(
        self,
        split: str,
        root_dir: str | Path | None = None,
        dataset_name: str | None = None,
        dataset_path: str | Path | None = None,
        text_encoder: CTCTextEncoder | None = None,
        max_samples: int | None = None,
    ) -> None:
        if dataset_path is None and root_dir is None:
            raise ValueError("Either root_dir or dataset_path must be provided.")

        self.root_dir = Path(dataset_path if dataset_path is not None else root_dir)
        self.split = split
        self.dataset_name = dataset_name or self.root_dir.name
        self.text_encoder = text_encoder

        self.split_dir = self.root_dir / split
        if not self.split_dir.exists() and dataset_name is not None:
            safe_name = dataset_name.replace("/", "_")
            legacy_split_dir = self.root_dir / safe_name / split
            if legacy_split_dir.exists():
                self.split_dir = legacy_split_dir
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
        # ✅ CRITICAL: Add error handling for corrupted .pt files
        try:
            sample = torch.load(path, map_location="cpu")
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint at {path}: {e}") from e
        
        # ✅ Validate required keys
        try:
            image = sample["image"]
        except KeyError as e:
            raise KeyError(f"Sample {path} missing required key 'image': {e}") from e
        
        # ✅ Validate image type and shape
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for image in {path}, got {type(image)}")
        
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim != 3:
            raise ValueError(f"Expected 2D or 3D image tensor, got shape {image.shape} in {path}")
        
        # ✅ Validate image size and dtype
        if image.shape != (1, 128, 512):
            raise ValueError(f"Expected shape (1, 128, 512), got {image.shape} in {path}")
        if image.dtype != torch.float32:
            raise TypeError(f"Expected float32, got {image.dtype} in {path}")
        if image.min() < -0.1 or image.max() > 1.1:  # Allow small numerical errors
            raise ValueError(f"Image not normalized to [0, 1], got range [{image.min():.4f}, {image.max():.4f}] in {path}")
        
        text = str(sample.get("text", ""))
        
        # ✅ Handle empty text samples gracefully
        if not text.strip():
            import warnings
            warnings.warn(f"Empty text in sample {path}", RuntimeWarning)
        
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
        split: str,
        root_dir: str | Path | None = None,
        dataset_name: str | None = None,
        dataset_path: str | Path | None = None,
        max_samples: int | None = None,
    ) -> CTCTextEncoder:
        """Scan a processed split and build a text encoder from all labels."""
        dataset = OfflineHandwritingDataset(
            split=split,
            root_dir=root_dir,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            text_encoder=None,
            max_samples=max_samples,
        )
        split_dir = dataset.split_dir
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


def build_text_encoder_for_dataset_splits(
    root_dir: str | Path,
    dataset_split_pairs: list[tuple[str, str]],
    max_samples_per_split: int | None = None,
) -> CTCTextEncoder:
    """Build one shared CTC encoder from multiple processed dataset splits."""
    texts: list[str] = []
    for dataset_name, split in dataset_split_pairs:
        dataset = OfflineHandwritingDataset(
            split=split,
            root_dir=root_dir,
            dataset_name=dataset_name,
            text_encoder=None,
            max_samples=max_samples_per_split,
        )
        for sample_path in dataset.sample_paths:
            sample = torch.load(sample_path, map_location="cpu")
            texts.append(str(sample.get("text", "")))
    return CTCTextEncoder.from_texts(texts)


def build_text_encoder_for_dataset_paths(
    dataset_paths: list[str | Path],
    split: str = "train",
    max_samples_per_path: int | None = None,
) -> CTCTextEncoder:
    """Build a shared CTC encoder from one or more processed dataset roots."""
    texts: list[str] = []
    for dataset_path in dataset_paths:
        split_dir = Path(dataset_path) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        sample_paths = sorted(split_dir.glob("sample_*.pt"))
        if max_samples_per_path is not None:
            sample_paths = sample_paths[:max_samples_per_path]

        for sample_path in sample_paths:
            # ✅ Add error handling for corrupted files
            try:
                sample = torch.load(sample_path, map_location="cpu")
                texts.append(str(sample.get("text", "")))
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint at {sample_path}: {e}") from e

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
    split: str,
    root_dir: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_path: str | Path | None = None,
    text_encoder: CTCTextEncoder | None = None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: int | None = None,
) -> DataLoader:
    """Create a dataloader for one processed offline split."""
    if text_encoder is None:
        raise ValueError("text_encoder is required to build the offline dataloader.")

    dataset = OfflineHandwritingDataset(
        split=split,
        root_dir=root_dir,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
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
