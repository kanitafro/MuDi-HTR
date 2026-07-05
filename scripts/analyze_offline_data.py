"""Standalone offline dataset analysis for report tables and discussion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze processed offline handwriting splits.")
    parser.add_argument("--dataset-path", required=True, help="Processed dataset root.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Split names to analyze.")
    parser.add_argument("--output", default="experiments/results/offline/data_analysis.json", help="Output JSON path.")
    return parser.parse_args()


def analyze_split(dataset_path: Path, split_name: str) -> dict:
    split_dir = dataset_path / split_name
    sample_paths = sorted(split_dir.glob("sample_*.pt"))
    lengths: list[int] = []
    characters: Counter[str] = Counter()
    image_means: list[float] = []

    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu")
        text = str(sample.get("text", ""))
        image = sample["image"]
        lengths.append(len(text))
        characters.update(text)
        image_means.append(float(image.mean()))

    return {
        "split": split_name,
        "num_samples": len(sample_paths),
        "avg_text_length": mean(lengths) if lengths else 0.0,
        "max_text_length": max(lengths, default=0),
        "unique_characters": len(characters),
        "top_characters": characters.most_common(20),
        "image_mean": mean(image_means) if image_means else 0.0,
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    summary = {
        "dataset_path": str(dataset_path),
        "splits": {split_name: analyze_split(dataset_path, split_name) for split_name in args.splits},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()