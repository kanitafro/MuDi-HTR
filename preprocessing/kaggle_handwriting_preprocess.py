"""Preprocess the Kaggle handwriting recognition dataset for the offline CRNN.

The Kaggle dataset ships as image folders plus CSV label files. This script
converts it into the same processed `.pt` sample format used by the offline
training pipeline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import torch

from preprocessing.offline_preprocess import preprocess_image


DEFAULT_IMAGE_SIZE = (128, 512)
DEFAULT_OUTPUT_ROOT = Path("data/processed/offline/kaggle_handwriting_recognition")


@dataclass(frozen=True)
class KaggleSplitSpec:
    split: str
    image_dir: str
    csv_name: str


DEFAULT_SPLITS = (
    KaggleSplitSpec("train", "train_v2", "written_name_train_v2.csv"),
    KaggleSplitSpec("val", "validation_v2", "written_name_validation_v2.csv"),
    KaggleSplitSpec("test", "test_v2", "written_name_test_v2.csv"),
)


IMAGE_COLUMNS = ("image", "filename", "file", "path", "image_id", "img", "image_name")
TEXT_COLUMNS = ("text", "label", "transcription", "word", "name", "identity", "truth", "target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Kaggle handwriting dataset into processed .pt samples.")
    parser.add_argument("--source-root", type=Path, required=True, help="Path to the extracted Kaggle dataset root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for processed samples.")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Processed image height.")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1], help="Processed image width.")
    parser.add_argument("--samples-per-split", type=int, default=0, help="Optional cap per split; 0 processes all samples.")
    parser.add_argument("--preview-count", type=int, default=20, help="Reserved for future preview output; kept for parity.")
    return parser.parse_args()


def _normalize_split_name(split: str) -> str:
    lowered = split.strip().lower()
    return "val" if lowered in {"validation", "valid"} else lowered


def _detect_column(fieldnames: Iterable[str], candidates: tuple[str, ...]) -> str:
    lower_map = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    raise KeyError(f"Could not detect a matching column from {candidates!r} in CSV fields: {list(fieldnames)!r}")


def _resolve_image_path(image_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    if candidate.suffix:
        direct = image_dir / candidate.name
        if direct.exists():
            return direct
    else:
        for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            direct = image_dir / f"{candidate.name}{suffix}"
            if direct.exists():
                return direct

    nested = image_dir / candidate
    if nested.exists():
        return nested

    raise FileNotFoundError(f"Could not resolve image path for {value!r} under {image_dir}")


def _extract_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {csv_path}")
        rows = [dict(row) for row in reader]
    return rows


def _get_row_value(row: dict[str, str], candidates: Iterable[str]) -> str | None:
    lower_row = {str(key).strip().lower(): value for key, value in row.items()}
    for candidate in candidates:
        value = lower_row.get(str(candidate).strip().lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _extract_text(row: dict[str, str]) -> str:
    value = _get_row_value(row, TEXT_COLUMNS)
    if value is not None:
        return value
    raise KeyError(f"Could not find a transcription column in row keys: {list(row.keys())!r}")


def _extract_image_name(row: dict[str, str]) -> str:
    value = _get_row_value(row, IMAGE_COLUMNS)
    if value is not None:
        return value
    raise KeyError(f"Could not find an image column in row keys: {list(row.keys())!r}")


def process_split(source_root: Path, output_root: Path, spec: KaggleSplitSpec, image_size: tuple[int, int], max_samples: int | None) -> int:
    split_root = source_root / spec.image_dir
    csv_path = source_root / spec.csv_name
    if not split_root.exists():
        raise FileNotFoundError(f"Missing image directory: {split_root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    rows = _extract_rows(csv_path)
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    output_dir = output_root / _normalize_split_name(spec.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_sample in output_dir.glob("sample_*.pt"):
        stale_sample.unlink()

    processed = 0
    for index, row in enumerate(rows):
        image_name = _extract_image_name(row)
        text = _extract_text(row)
        image_path = _resolve_image_path(split_root, image_name)
        processed_np = preprocess_image(image_path, image_size=image_size, augment=False, binarize=False)
        tensor_data = torch.tensor(processed_np, dtype=torch.float32).unsqueeze(0)

        torch.save(
            {
                "image": tensor_data,
                "text": text,
                "dataset": "Kaggle/handwriting-recognition",
                "split": _normalize_split_name(spec.split),
                "source_path": str(image_path),
            },
            output_dir / f"sample_{index}.pt",
        )
        processed += 1

    manifest = {
        "dataset": "Kaggle/handwriting-recognition",
        "split": _normalize_split_name(spec.split),
        "source_root": str(source_root),
        "image_dir": spec.image_dir,
        "csv_name": spec.csv_name,
        "num_samples": processed,
        "image_size": {"height": image_size[0], "width": image_size[1]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "kaggle_local",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Saved {processed} samples to {output_dir}")
    return processed


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    output_root = args.output_root
    image_size = (args.image_height, args.image_width)
    max_samples = args.samples_per_split if args.samples_per_split > 0 else None

    total = 0
    for split_spec in DEFAULT_SPLITS:
        total += process_split(source_root, output_root, split_spec, image_size=image_size, max_samples=max_samples)

    print(f"Completed Kaggle handwriting preprocessing: {total} samples total.")


if __name__ == "__main__":
    main()