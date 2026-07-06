"""Preprocess the GNHK handwriting dataset for the offline CRNN.

The dataset provides one JSON file per image, with token-level transcription
items that include text and line indices. This script converts each
image/JSON pair into the processed `.pt` sample format used by the offline
training pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import torch

from preprocessing.offline_preprocess import preprocess_image


DEFAULT_IMAGE_SIZE = (128, 512)
DEFAULT_OUTPUT_ROOT = Path("data/processed/offline/gnhk")
DEFAULT_SPLITS = ("train", "test")


@dataclass(frozen=True)
class GNHKSplitSpec:
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the GNHK handwriting dataset into processed .pt samples.")
    parser.add_argument("--source-root", type=Path, required=True, help="Path to the extracted GNHK dataset root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for processed samples.")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Source splits to preprocess, e.g. train test.")
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Processed image height.")
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1], help="Processed image width.")
    parser.add_argument("--samples-per-split", type=int, default=0, help="Optional cap per split; 0 processes all samples.")
    return parser.parse_args()


def _normalize_split_name(split: str) -> str:
    lowered = split.strip().lower()
    return "val" if lowered in {"validation", "valid"} else lowered


def _load_json_payload(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        root = payload.get("root", payload)
        if isinstance(root, list):
            return [item for item in root if isinstance(item, dict)]
        raise ValueError(f"Unexpected JSON structure in {json_path}: expected a list under 'root'.")

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    raise ValueError(f"Unexpected JSON structure in {json_path}: expected a list or a dict.")


def _extract_transcription(entries: list[dict]) -> str:
    lines: defaultdict[int, list[str]] = defaultdict(list)

    for index, entry in enumerate(entries):
        text = str(entry.get("text", "")).strip()
        if not text:
            continue

        line_index = entry.get("line_idx")
        if line_index is None:
            line_index = index

        try:
            normalized_index = int(line_index)
        except (TypeError, ValueError):
            normalized_index = index

        lines[normalized_index].append(text)

    line_texts = [" ".join(tokens) for _, tokens in sorted(lines.items(), key=lambda item: item[0]) if tokens]
    return re.sub(r"\s+", " ", " ".join(line_texts)).strip()


def _resolve_image_path(split_root: Path, json_path: Path) -> Path:
    stem = json_path.stem
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidate = split_root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    sibling = json_path.with_suffix(".jpg")
    if sibling.exists():
        return sibling

    raise FileNotFoundError(f"Could not find an image for JSON file {json_path}")


def process_split(source_root: Path, output_root: Path, split: str, image_size: tuple[int, int], max_samples: int | None) -> int:
    split_name = _normalize_split_name(split)
    split_root = source_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    json_paths = sorted(split_root.glob("**/*.json"))
    if max_samples is not None and max_samples > 0:
        json_paths = json_paths[:max_samples]

    if not json_paths:
        raise FileNotFoundError(f"No JSON samples found in {split_root}")

    output_dir = output_root / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_sample in output_dir.glob("sample_*.pt"):
        stale_sample.unlink()

    processed = 0
    skipped = 0
    for index, json_path in enumerate(json_paths):
        entries = _load_json_payload(json_path)
        transcription = _extract_transcription(entries)
        if not transcription:
            skipped += 1
            continue

        image_path = _resolve_image_path(split_root, json_path)
        processed_np = preprocess_image(image_path, image_size=image_size, augment=False, binarize=True)
        tensor_data = torch.tensor(processed_np, dtype=torch.float32).unsqueeze(0)

        torch.save(
            {
                "image": tensor_data,
                "text": transcription,
                "dataset": "GNHK",
                "split": split_name,
                "source_path": str(image_path),
                "json_path": str(json_path),
            },
            output_dir / f"sample_{index}.pt",
        )
        processed += 1

    manifest = {
        "dataset": "GNHK",
        "split": split_name,
        "source_root": str(source_root),
        "num_samples": processed,
        "num_skipped": skipped,
        "image_size": {"height": image_size[0], "width": image_size[1]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "gnhk_local",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Saved {processed} samples to {output_dir} (skipped {skipped})")
    return processed


def main() -> None:
    args = parse_args()
    image_size = (args.image_height, args.image_width)
    max_samples = args.samples_per_split if args.samples_per_split > 0 else None

    total = 0
    for split in args.splits:
        total += process_split(args.source_root, args.output_root, split, image_size=image_size, max_samples=max_samples)

    print(f"Completed GNHK preprocessing: {total} samples total.")


if __name__ == "__main__":
    main()