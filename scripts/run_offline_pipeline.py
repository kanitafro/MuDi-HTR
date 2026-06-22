"""Offline handwriting preprocessing entrypoint using Hugging Face streaming."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from preprocessing.offline_preprocess import preprocess_image


OUTPUT_ROOT = Path("data/processed/offline")
FIGURES_ROOT = Path("experiments/figures/offline")
DEFAULT_DATASET = "to-be/OpenHand-Synth"
DEFAULT_SPLITS = ("train", "test")


def _save_preview_grid(
    images: list[np.ndarray],
    output_path: Path,
) -> None:
    """Save a small preview grid for fast sanity checks."""
    if not images:
        return

    cols = min(5, len(images))
    rows = (len(images) + cols - 1) // cols
    cell_h, cell_w = images[0].shape
    canvas = np.full((rows * cell_h, cols * cell_w), 1.0, dtype=np.float32)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = row * cell_h
        y1 = y0 + cell_h
        x0 = col * cell_w
        x1 = x0 + cell_w
        canvas[y0:y1, x0:x1] = image

    out_img = Image.fromarray((canvas * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_path)


def process_data_split(
    dataset_name: str,
    split_name: str,
    num_samples: int,
    image_size: tuple[int, int],
    augment_train: bool,
    preview_count: int,
    seed: int,
) -> None:
    """Stream one split from Hugging Face and save processed PyTorch samples."""
    print(f"Streaming dataset='{dataset_name}' split='{split_name}' from Hugging Face...")
    dataset = load_dataset(dataset_name, split=split_name, streaming=True)

    safe_dataset_name = dataset_name.replace("/", "_")
    output_dir = OUTPUT_ROOT / safe_dataset_name / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    use_augmentation = augment_train and split_name.lower() == "train"
    rng = np.random.default_rng(seed)
    preview_images: list[np.ndarray] = []
    processed_count = 0

    for index, item in enumerate(tqdm(dataset, total=num_samples)):
        if index >= num_samples:
            break

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            item["image"].save(temp_path)
            processed_np = preprocess_image(
                temp_path,
                image_size=image_size,
                augment=use_augmentation,
                rng=rng,
            )
            tensor_data = torch.tensor(processed_np, dtype=torch.float32).unsqueeze(0)
            text = item.get("text", "")

            torch.save(
                {
                    "image": tensor_data,
                    "text": text,
                    "dataset": dataset_name,
                    "split": split_name,
                    # Keep a stable stream identifier instead of the deleted temp file path.
                    "source_path": f"hf://{dataset_name}/{split_name}/{index}",
                },
                output_dir / f"sample_{index}.pt",
            )

            if len(preview_images) < preview_count:
                preview_images.append(processed_np)

            processed_count += 1
        finally:
            if temp_path.exists():
                temp_path.unlink()

    preview_path = FIGURES_ROOT / safe_dataset_name / f"{split_name}_preview.png"
    _save_preview_grid(preview_images, preview_path)

    manifest = {
        "dataset": dataset_name,
        "split": split_name,
        "num_samples": processed_count,
        "image_size": {"height": image_size[0], "width": image_size[1]},
        "augmentation": {
            "enabled": use_augmentation,
            "rotation_degrees": 5.0,
            "scale_delta": 0.10,
        },
        "preview_figure": str(preview_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "huggingface_streaming",
        "seed": seed,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Saved {processed_count} samples to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline dataset streaming/preprocessing."""
    parser = argparse.ArgumentParser(description="Run offline preprocessing from Hugging Face streams.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to process, e.g. train validation test.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=1000,
        help="Maximum number of streamed samples per split.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Target image height after preprocessing.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Target image width after preprocessing.",
    )
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Apply light augmentation only on the train split.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=20,
        help="How many samples to include in split preview grids.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_size = (args.height, args.width)

    for split in args.splits:
        process_data_split(
            args.dataset,
            split,
            num_samples=args.samples_per_split,
            image_size=target_size,
            augment_train=args.augment_train,
            preview_count=args.preview_count,
            seed=args.seed,
        )
    print("Offline data processing complete.")
