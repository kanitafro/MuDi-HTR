"""Stage-1 offline handwriting preprocessing entrypoint using Hugging Face streaming.

This helper keeps the existing OpenHand-Synth pretraining path intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import urllib.request

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.offline_preprocess import preprocess_image



OUTPUT_ROOT = Path("data/processed/offline")
FIGURES_ROOT = Path("experiments/figures/offline")


DEFAULT_DATASET = "to-be/OpenHand-Synth"
DEFAULT_SPLITS = ("train", "val")
DEFAULT_DATASET_SPECS = (
    "to-be/OpenHand-Synth:train,val",
    "Voxel51/iam_handwriting_finevision:train",
)


SOURCE_SPLIT_ALIASES = {
    "val": ("validation", "valid", "test"),
    "validation": ("validation", "valid", "test"),
    "valid": ("valid", "validation", "test"),
    "test": ("test", "validation", "valid"),
    "train": ("train",),
}


TEXT_KEYS = (
    "assistant",
    "messages",
    "conversation",
    "conversations",
    "dialogue",
    "text",
    "label",
    "transcription",
    "caption",
    "answer",
    "ground_truth",
    "gt",
)

FINEVISION_SAMPLE_URL = "https://huggingface.co/datasets/Voxel51/iam_handwriting_finevision/raw/main/samples.json"


def dataset_slug(dataset_name: str) -> str:
    """Map a source dataset id to a stable output folder name."""
    lowered = dataset_name.lower()
    if "openhand" in lowered:
        return "openhand_synth"
    if "voxel51" in lowered and "finevision" in lowered:
        return "iam_handwriting_finevision"
    if "gnhk" in lowered:
        return "gnhk"
    return dataset_name.replace("/", "_").replace("-", "_").lower()


def normalize_split_name(split_name: str) -> str:
    """Map user-facing split names to folder names used by training."""
    lowered = split_name.strip().lower()
    if lowered in {"validation", "valid"}:
        return "val"
    return lowered


def build_output_dir(dataset_name: str, split_name: str) -> Path:
    """Return the exact on-disk directory for a processed dataset split."""
    return OUTPUT_ROOT / dataset_slug(dataset_name) / normalize_split_name(split_name)


def resolve_source_split_name(dataset_name: str, split_name: str) -> str:
    """Resolve the split name to use when streaming from Hugging Face."""
    split_key = normalize_split_name(split_name)
    candidates = SOURCE_SPLIT_ALIASES.get(split_key, (split_key,))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            load_dataset(dataset_name, split=candidate, streaming=True)
            return candidate
        except Exception as error:  # pragma: no cover - fallback path
            last_error = error

    if last_error is not None:
        raise last_error
    return split_key


def extract_transcription(item: dict) -> str:
    """Extract the ground-truth transcription from a dataset row."""
    def _extract_value(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            role = str(value.get("role", "")).strip().lower()
            if role == "assistant":
                for content_key in ("content", "text", "value", "answer", "caption", "transcription"):
                    extracted = _extract_value(value.get(content_key))
                    if extracted:
                        return extracted
            for content_key in ("content", "text", "value", "answer", "caption", "transcription", "label"):
                if content_key in value:
                    extracted = _extract_value(value[content_key])
                    if extracted:
                        return extracted
            for key in TEXT_KEYS:
                if key in value:
                    extracted = _extract_value(value[key])
                    if extracted:
                        return extracted
            for nested_value in value.values():
                extracted = _extract_value(nested_value)
                if extracted:
                    return extracted
            return ""
        if isinstance(value, (list, tuple)):
            if any(isinstance(entry, dict) and "role" in entry for entry in value):
                for entry in value:
                    if isinstance(entry, dict) and str(entry.get("role", "")).strip().lower() == "assistant":
                        extracted = _extract_value(
                            entry.get("content")
                            or entry.get("text")
                            or entry.get("value")
                            or entry.get("answer")
                            or entry.get("caption")
                            or entry.get("transcription")
                        )
                        if extracted:
                            return extracted
                return ""
            for entry in value:
                extracted = _extract_value(entry)
                if extracted:
                    return extracted
            return ""
        return str(value).strip()

    for key in TEXT_KEYS:
        if key in item:
            extracted = _extract_value(item[key])
            if extracted:
                return extracted
    return ""


@lru_cache(maxsize=1)
def load_finevision_samples() -> list[dict]:
    """Load the FineVision FiftyOne sample manifest from the dataset repository."""
    with urllib.request.urlopen(FINEVISION_SAMPLE_URL, timeout=120) as response:
        manifest = json.load(response)

    samples = manifest.get("samples", []) if isinstance(manifest, dict) else []
    if not isinstance(samples, list):
        raise ValueError("Unexpected FineVision samples.json structure.")
    return [sample for sample in samples if isinstance(sample, dict)]


def is_finevision_dataset(dataset_name: str) -> bool:
    lowered = dataset_name.lower()
    return "finevision" in lowered and "voxel51" in lowered


def extract_finevision_transcription(sample: dict) -> str:
    """Extract the handwritten transcription from a FineVision manifest sample."""
    return str(sample.get("assistant", "")).strip()


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
    num_samples: int | None,
    image_size: tuple[int, int],
    augment_train: bool,
    preview_count: int,
    seed: int,
) -> None:
    """Stream one split from Hugging Face and save processed PyTorch samples."""
    source_split_name = resolve_source_split_name(dataset_name, split_name)
    print(
        f"Streaming dataset='{dataset_name}' source split='{source_split_name}' "
        f"-> output split='{normalize_split_name(split_name)}'..."
    )
    dataset = load_dataset(dataset_name, split=source_split_name, streaming=True)

    safe_dataset_name = dataset_slug(dataset_name)
    normalized_split_name = normalize_split_name(split_name)
    output_dir = build_output_dir(dataset_name, split_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stale_sample in output_dir.glob("sample_*.pt"):
        stale_sample.unlink()

    use_augmentation = augment_train and normalized_split_name == "train" and not is_finevision_dataset(dataset_name)
    use_binarization = not is_finevision_dataset(dataset_name)
    rng = np.random.default_rng(seed)
    preview_images: list[np.ndarray] = []
    processed_count = 0

    finevision_samples: list[dict] | None = None
    if is_finevision_dataset(dataset_name):
        finevision_samples = load_finevision_samples()
        if num_samples and num_samples > 0:
            finevision_samples = finevision_samples[:num_samples]

    scratch_dir = output_dir / "_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(tqdm(dataset, total=num_samples if num_samples and num_samples > 0 else None)):
        if num_samples and num_samples > 0 and index >= num_samples:
            break

        temp_path = scratch_dir / f"sample_{index}.png"

        try:
            item["image"].save(temp_path)
            processed_np = preprocess_image(
                temp_path,
                image_size=image_size,
                augment=use_augmentation,
                binarize=use_binarization,
                rng=rng,
            )
            tensor_data = torch.tensor(processed_np, dtype=torch.float32).unsqueeze(0)
            if finevision_samples is not None:
                if index >= len(finevision_samples):
                    break
                sample = finevision_samples[index]
                text = extract_finevision_transcription(sample)
                source_path = sample.get("filepath", f"data/{index:05d}.png")
            else:
                text = extract_transcription(item)
                source_path = f"hf://{dataset_name}/{source_split_name}/{index}"

            if not text:
                continue

            torch.save(
                {
                    "image": tensor_data,
                    "text": text,
                    "dataset": dataset_name,
                    "split": normalized_split_name,
                    "source_path": source_path,
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
        "split": normalized_split_name,
        "num_samples": processed_count,
        "image_size": {"height": image_size[0], "width": image_size[1]},
        "augmentation": {
            "enabled": use_augmentation,
            "rotation_degrees": 5.0,
            "scale_delta": 0.10,
        },
        "binarization": "otsu" if use_binarization else "disabled",
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
        "--dataset-name",
        default=None,
        help="Stable output folder name under data/processed/offline (defaults to an inferred slug).",
    )
    parser.add_argument(
        "--dataset-spec",
        action="append",
        default=[],
        help="Dataset spec in the form dataset_id:split1,split2. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to process, e.g. train val test.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=0,
        help="Maximum number of streamed samples per split. Use 0 to process the full split.",
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


def parse_dataset_specs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Build dataset/split pairs from CLI arguments."""
    if args.dataset_spec:
        return parse_dataset_spec_strings(args.dataset_spec)

    return [(args.dataset, list(args.splits))]


def parse_dataset_spec_strings(dataset_specs: list[str]) -> list[tuple[str, list[str]]]:
    """Parse raw dataset spec strings into (dataset_name, splits) tuples."""
    parsed_specs: list[tuple[str, list[str]]] = []
    for spec in dataset_specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid dataset spec '{spec}'. Expected format dataset_id:split1,split2"
            )
        dataset_name, split_text = spec.split(":", 1)
        splits = [split.strip() for split in split_text.split(",") if split.strip()]
        if not dataset_name or not splits:
            raise ValueError(
                f"Invalid dataset spec '{spec}'. Dataset name and at least one split are required."
            )
        parsed_specs.append((dataset_name, splits))
    return parsed_specs


if __name__ == "__main__":
    args = parse_args()
    if args.dataset_name is not None:
        print(
            f"Using output folder slug '{args.dataset_name}' is not required anymore; "
            "folder names are inferred from the dataset id."
        )
    target_size = (args.height, args.width)
    dataset_specs = parse_dataset_specs(args) if args.dataset_spec else parse_dataset_spec_strings(list(DEFAULT_DATASET_SPECS))

    if not args.dataset_spec and args.dataset != DEFAULT_DATASET:
        dataset_specs = [(args.dataset, list(args.splits))]

    for dataset_name, splits in dataset_specs:
        for split in splits:
            process_data_split(
                dataset_name,
                split,
                num_samples=args.samples_per_split,
                image_size=target_size,
                augment_train=args.augment_train,
                preview_count=args.preview_count,
                seed=args.seed,
            )
    print("Offline data processing complete.")
