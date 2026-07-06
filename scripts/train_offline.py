"""Offline CRNN training for MuDi-HTR.

Two-stage domain adaptation:
- Stage 1: pretrain on synthetic OpenHand-Synth
- Stage 2: fine-tune on the GNHK handwriting dataset when explicitly enabled

The script stores structured metrics, best checkpoints per stage, and
report-ready evaluation artifacts for the final fine-tuned model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from statistics import mean
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.offline import CRNN, build_text_encoder_for_dataset_paths, create_offline_dataloader
from models.offline import resolve_processed_dataset_root


RESULTS_ROOT = Path("experiments/results/offline")
CHECKPOINT_ROOT = Path("checkpoints/offline")
METRICS_PATH = RESULTS_ROOT / "training_metrics.json"
SUMMARY_PATH = RESULTS_ROOT / "training_summary.txt"
ERROR_LOG_PATH = RESULTS_ROOT / "error_analysis.txt"
LATEX_TABLE_PATH = RESULTS_ROOT / "report_summary.tex"

DEFAULT_STAGE1_DATASET_PATH = Path("data/processed/offline/openhand_synth")
DEFAULT_STAGE2_DATASET_PATH = Path("data/processed/offline/gnhk")
DEFAULT_STAGE2_VALIDATION_FRACTION = 0.1


def validate_dataset_split(dataset_path: Path, split_name: str, label: str) -> None:
    split_dir = dataset_path / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"{label} split directory missing: {split_dir}")
    if not list(split_dir.glob("sample_*.pt")):
        raise FileNotFoundError(f"{label} split has no processed samples: {split_dir}")


def validate_dataset_paths(stage1_path: Path, stage1_val_split: str, stage2_path: Path, stage2_val_split: str) -> None:
    if stage1_path.resolve() == stage2_path.resolve():
        raise ValueError("Stage 1 and stage 2 dataset paths must be different.")

    missing_paths = [path for path in (stage1_path, stage2_path) if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Offline dataset path(s) missing: {missing_text}")

    validate_dataset_split(stage1_path, "train", "Stage 1 train")
    validate_dataset_split(stage1_path, stage1_val_split, "Stage 1 validation")
    validate_dataset_split(stage2_path, "train", "Stage 2 train")
    if stage2_val_split != "auto":
        validate_dataset_split(stage2_path, stage2_val_split, "Stage 2 validation")


def summarize_processed_dataset(dataset_path: Path, split_name: str, max_samples: int | None = None) -> dict:
    split_dir = dataset_path / split_name
    sample_paths = sorted(split_dir.glob("sample_*.pt"))
    if max_samples is not None:
        sample_paths = sample_paths[:max_samples]

    text_lengths: list[int] = []
    char_counter: Counter[str] = Counter()
    image_means: list[float] = []
    image_ranges: list[tuple[float, float]] = []

    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu")
        text = str(sample.get("text", ""))
        image = sample["image"]

        text_lengths.append(len(text))
        char_counter.update(text)
        image_means.append(float(image.mean()))
        image_ranges.append((float(image.min()), float(image.max())))

    min_value = min((item[0] for item in image_ranges), default=0.0)
    max_value = max((item[1] for item in image_ranges), default=0.0)

    return {
        "dataset_path": str(dataset_path),
        "split": split_name,
        "num_samples": len(sample_paths),
        "avg_text_length": mean(text_lengths) if text_lengths else 0.0,
        "max_text_length": max(text_lengths, default=0),
        "min_text_length": min(text_lengths, default=0),
        "unique_characters": len(char_counter),
        "character_frequency_top20": char_counter.most_common(20),
        "image_mean": mean(image_means) if image_means else 0.0,
        "image_min": min_value,
        "image_max": max_value,
    }


def summarize_sample_paths(sample_paths: list[Path], dataset_path: Path, split_name: str) -> dict:
    """Summarize a split that lives only as an in-memory list of processed sample paths."""
    text_lengths: list[int] = []
    char_counter: Counter[str] = Counter()
    image_means: list[float] = []
    image_ranges: list[tuple[float, float]] = []

    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu")
        text = str(sample.get("text", ""))
        image = sample["image"]

        text_lengths.append(len(text))
        char_counter.update(text)
        image_means.append(float(image.mean()))
        image_ranges.append((float(image.min()), float(image.max())))

    min_value = min((item[0] for item in image_ranges), default=0.0)
    max_value = max((item[1] for item in image_ranges), default=0.0)

    return {
        "dataset_path": str(dataset_path),
        "split": split_name,
        "num_samples": len(sample_paths),
        "avg_text_length": mean(text_lengths) if text_lengths else 0.0,
        "max_text_length": max(text_lengths, default=0),
        "min_text_length": min(text_lengths, default=0),
        "unique_characters": len(char_counter),
        "character_frequency_top20": char_counter.most_common(20),
        "image_mean": mean(image_means) if image_means else 0.0,
        "image_min": min_value,
        "image_max": max_value,
    }


def split_sample_paths(
    sample_paths: list[Path],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split processed sample paths into deterministic train and validation subsets."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    if len(sample_paths) < 2:
        raise ValueError("At least two samples are required to create a held-out validation subset.")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(sample_paths), generator=generator).tolist()
    shuffled_paths = [sample_paths[index] for index in permutation]

    validation_count = max(1, int(round(len(shuffled_paths) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled_paths) - 1)

    validation_paths = shuffled_paths[:validation_count]
    train_paths = shuffled_paths[validation_count:]

    if not train_paths:
        raise ValueError("Not enough samples to keep a non-empty training subset.")

    return train_paths, validation_paths


def build_data_analysis(
    stage1_path: Path,
    stage1_val_split: str,
    stage2_path: Path,
    stage2_val_split: str,
    validation_fraction: float,
    seed: int,
    stage1_train_max_samples: int | None = None,
    stage1_val_max_samples: int | None = None,
    stage2_max_samples: int | None = None,
    stage2_val_max_samples: int | None = None,
) -> dict:
    stage2_train_dir = stage2_path / "train"
    stage2_sample_paths = sorted(stage2_train_dir.glob("sample_*.pt"))
    if stage2_max_samples is not None:
        stage2_sample_paths = stage2_sample_paths[:stage2_max_samples]
    stage2_train_paths, stage2_val_paths = split_sample_paths(
        stage2_sample_paths,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    analysis = {
        "stage1": {
            "train": summarize_processed_dataset(stage1_path, "train", max_samples=stage1_train_max_samples),
            "val": summarize_processed_dataset(stage1_path, stage1_val_split, max_samples=stage1_val_max_samples),
        },
        "stage2": {
            "train": summarize_processed_dataset(stage2_path, "train", max_samples=stage2_max_samples) if stage2_val_split != "auto" else summarize_sample_paths(stage2_train_paths, stage2_path, "train_holdout"),
            "val": summarize_processed_dataset(stage2_path, stage2_val_split, max_samples=stage2_val_max_samples) if stage2_val_split != "auto" else summarize_sample_paths(stage2_val_paths, stage2_path, "val_holdout"),
        },
    }
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the offline CRNN in two stages.")
    parser.add_argument(
        "--stage1-dataset-path",
        default=str(DEFAULT_STAGE1_DATASET_PATH),
        help="Processed OpenHand-Synth root containing train/val splits.",
    )
    parser.add_argument("--stage1-train-split", default="train", help="Training split for stage 1.")
    parser.add_argument("--stage1-val-split", default="val", help="Validation split for stage 1.")
    parser.add_argument(
        "--stage2-dataset-path",
        default=str(DEFAULT_STAGE2_DATASET_PATH),
        help="Processed GNHK root containing train/test splits (validation can be carved from train).",
    )
    parser.add_argument("--stage2-train-split", default="train", help="Training split for stage 2.")
    parser.add_argument("--stage2-val-split", default="auto", help="Validation split for stage 2. Use 'auto' to carve out a held-out subset from train.")
    parser.add_argument("--stage2-validation-fraction", type=float, default=DEFAULT_STAGE2_VALIDATION_FRACTION, help="Fraction of stage 2 train samples reserved for validation when --stage2-val-split is auto.")
    parser.add_argument("--max-vocab-samples", type=int, default=50000, help="Cap samples used to build the shared vocab.")
    parser.add_argument("--stage1-max-samples", type=int, default=None, help="Optional cap for stage 1 train samples.")
    parser.add_argument("--stage1-val-max-samples", type=int, default=None, help="Optional cap for stage 1 val samples.")
    parser.add_argument("--stage2-max-samples", type=int, default=None, help="Optional cap for stage 2 train samples.")
    parser.add_argument("--stage2-val-max-samples", type=int, default=None, help="Optional cap for stage 2 val samples.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs-stage1", type=int, default=25, help="Pretraining epochs.")
    parser.add_argument("--epochs-stage2", type=int, default=60, help="Fine-tuning epochs.")
    parser.add_argument("--run-finetune", action="store_true", help="Run the stage 2 fine-tuning phase after pretraining.")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip stage 1 training and load an existing pretrained checkpoint.")
    parser.add_argument("--pretrained-checkpoint", default=str(CHECKPOINT_ROOT / "pretrained.pth"), help="Path to an existing pretrained checkpoint used when --skip-stage1 is enabled.")
    parser.add_argument("--dataset-name", default=None, help="Compatibility alias for the stage 2 dataset name or processed root.")
    parser.add_argument("--lr-stage1", type=float, default=1e-3, help="Learning rate for stage 1.")
    parser.add_argument("--lr-stage2", type=float, default=1e-4, help="Learning rate for stage 2.")
    parser.add_argument("--stage2-head-only-epochs", type=int, default=3, help="Train only the classifier head for this many initial fine-tuning epochs.")
    parser.add_argument("--stage2-freeze-cnn-epochs", type=int, default=6, help="Freeze the CNN backbone for this many fine-tuning epochs.")
    parser.add_argument("--stage2-backbone-lr-scale", type=float, default=0.05, help="LR multiplier for the CNN backbone during stage 2.")
    parser.add_argument("--stage2-head-lr-scale", type=float, default=5.0, help="LR multiplier for the classifier head during stage 2.")
    parser.add_argument("--hidden-size", type=int, default=256, help="CRNN hidden size.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def edit_distance(source: Iterable[str], target: Iterable[str]) -> int:
    source_seq = list(source)
    target_seq = list(target)
    previous_row = list(range(len(target_seq) + 1))
    for i, source_item in enumerate(source_seq, start=1):
        current_row = [i]
        for j, target_item in enumerate(target_seq, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (source_item != target_item)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def compute_cer_wer(predictions: list[str], references: list[str]) -> tuple[float, float]:
    char_errors = 0
    char_total = 0
    word_errors = 0
    word_total = 0

    for prediction, reference in zip(predictions, references):
        char_errors += edit_distance(prediction, reference)
        char_total += max(1, len(reference))

        reference_words = reference.split()
        prediction_words = prediction.split()
        word_errors += edit_distance(prediction_words, reference_words)
        word_total += max(1, len(reference_words))

    return char_errors / max(1, char_total), word_errors / max(1, word_total)


def backtrace_alignment(reference: str, prediction: str) -> list[tuple[str, str, str]]:
    rows = len(reference)
    cols = len(prediction)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    ops = [["match"] * (cols + 1) for _ in range(rows + 1)]

    for row in range(1, rows + 1):
        dp[row][0] = row
        ops[row][0] = "del"
    for col in range(1, cols + 1):
        dp[0][col] = col
        ops[0][col] = "ins"

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            cost_sub = dp[row - 1][col - 1] + (reference[row - 1] != prediction[col - 1])
            cost_del = dp[row - 1][col] + 1
            cost_ins = dp[row][col - 1] + 1
            best_cost = min(cost_sub, cost_del, cost_ins)
            dp[row][col] = best_cost
            if best_cost == cost_sub:
                ops[row][col] = "match" if reference[row - 1] == prediction[col - 1] else "sub"
            elif best_cost == cost_del:
                ops[row][col] = "del"
            else:
                ops[row][col] = "ins"

    row = rows
    col = cols
    alignment: list[tuple[str, str, str]] = []
    while row > 0 or col > 0:
        op = ops[row][col]
        if op in {"match", "sub"}:
            alignment.append((op, reference[row - 1], prediction[col - 1]))
            row -= 1
            col -= 1
        elif op == "del":
            alignment.append((op, reference[row - 1], ""))
            row -= 1
        else:
            alignment.append((op, "", prediction[col - 1]))
            col -= 1
    alignment.reverse()
    return alignment


def greedy_decode(logits: torch.Tensor, encoder) -> list[str]:
    token_ids = logits.argmax(dim=-1).permute(1, 0).tolist()
    return [encoder.decode(sequence) for sequence in token_ids]


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def configure_finetune_trainability(model: nn.Module, epoch: int, head_only_epochs: int, freeze_cnn_epochs: int) -> None:
    if epoch <= head_only_epochs:
        set_module_trainable(model.cnn, False)
        set_module_trainable(model.sequence_projection, False)
        set_module_trainable(model.rnn, False)
        set_module_trainable(model.classifier, True)
        return

    if epoch <= freeze_cnn_epochs:
        set_module_trainable(model.cnn, False)
        set_module_trainable(model.sequence_projection, True)
        set_module_trainable(model.rnn, True)
        set_module_trainable(model.classifier, True)
        return

    set_module_trainable(model.cnn, True)
    set_module_trainable(model.sequence_projection, True)
    set_module_trainable(model.rnn, True)
    set_module_trainable(model.classifier, True)


def checkpoint_is_compatible(model: nn.Module, checkpoint_state_dict: dict[str, torch.Tensor]) -> bool:
    model_state_dict = model.state_dict()
    if set(model_state_dict.keys()) != set(checkpoint_state_dict.keys()):
        return False

    for key, value in checkpoint_state_dict.items():
        if model_state_dict[key].shape != value.shape:
            return False
    return True


def build_optimizer(
    model: nn.Module,
    lr: float,
    stage_name: str,
    backbone_lr_scale: float,
    head_lr_scale: float,
) -> AdamW:
    if stage_name != "finetune":
        return AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    return AdamW(
        [
            {"params": model.cnn.parameters(), "lr": lr * backbone_lr_scale},
            {"params": model.sequence_projection.parameters(), "lr": lr},
            {"params": model.rnn.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr * head_lr_scale},
        ],
        weight_decay=1e-4,
    )


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer | None,
    encoder,
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_predictions: list[str] = []
    all_references: list[str] = []
    total_samples = 0

    for batch in dataloader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        references = batch["texts"]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            log_probs = logits.log_softmax(dim=-1)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=logits.size(0),
                dtype=torch.long,
                device=device,
            )
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_predictions = greedy_decode(logits.detach().cpu(), encoder)
        all_predictions.extend(batch_predictions)
        all_references.extend(references)
        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)

        del images, targets, target_lengths, logits, log_probs, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, total_samples)
    cer, wer = compute_cer_wer(all_predictions, all_references)
    return {
        "loss": avg_loss,
        "cer": cer,
        "wer": wer,
        "predictions": all_predictions,
        "references": all_references,
    }


@torch.no_grad()
def preview_validation_examples(
    model: nn.Module,
    dataloader,
    encoder,
    device: torch.device,
    max_examples: int = 2,
    beam_width: int = 5,
) -> list[dict[str, str | float]]:
    model.eval()
    for batch in dataloader:
        images = batch["images"].to(device)
        references = batch["texts"]

        logits = model(images)
        greedy_predictions = greedy_decode(logits.detach().cpu(), encoder)
        beam_predictions = model.decode_beam_search(
            logits.detach(),
            alphabet=encoder.vocab,
            beam_width=beam_width,
            blank_index=encoder.blank_index,
        )

        previews: list[dict[str, str | float]] = []
        for index, reference in enumerate(references[:max_examples]):
            greedy_prediction = greedy_predictions[index] if index < len(greedy_predictions) else ""
            beam_prediction, beam_confidence = beam_predictions[index] if index < len(beam_predictions) else ("", 0.0)
            previews.append(
                {
                    "reference": reference,
                    "greedy": greedy_prediction,
                    "beam": beam_prediction,
                    "beam_confidence": float(beam_confidence),
                }
            )
        return previews

    return []


def print_validation_preview(stage_name: str, previews: list[dict[str, str | float]]) -> None:
    if not previews:
        print(f"[{stage_name}] preview: no validation samples available.", flush=True)
        return

    for index, preview in enumerate(previews, start=1):
        reference = preview["reference"]
        greedy = preview["greedy"]
        beam = preview["beam"]
        beam_confidence = preview["beam_confidence"]
        print(
            f"[{stage_name}] preview {index}: ref={reference!r} | greedy={greedy!r} | beam={beam!r} (conf={beam_confidence:.3f})",
            flush=True,
        )


def build_loaders(args: argparse.Namespace, encoder):
    stage1_root = Path(args.stage1_dataset_path)
    stage2_root = resolve_processed_dataset_root(args.data_root, args.dataset_name) if args.dataset_name else Path(args.stage2_dataset_path)

    stage1_train = create_offline_dataloader(
        dataset_path=stage1_root,
        split=args.stage1_train_split,
        text_encoder=encoder,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_samples=args.stage1_max_samples,
        augment=True,
    )
    stage1_val = create_offline_dataloader(
        dataset_path=stage1_root,
        split=args.stage1_val_split,
        text_encoder=encoder,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.stage1_val_max_samples,
    )
    if args.stage2_val_split == "auto":
        stage2_sample_paths = sorted((stage2_root / "train").glob("sample_*.pt"))
        if args.stage2_max_samples is not None:
            stage2_sample_paths = stage2_sample_paths[:args.stage2_max_samples]
        stage2_train_paths, stage2_val_paths = split_sample_paths(
            stage2_sample_paths,
            validation_fraction=args.stage2_validation_fraction,
            seed=args.seed,
        )
        stage2_train = create_offline_dataloader(
            dataset_path=stage2_root,
            split=args.stage2_train_split,
            text_encoder=encoder,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            sample_paths=stage2_train_paths,
            augment=True,
        )
        stage2_val = create_offline_dataloader(
            dataset_path=stage2_root,
            split="val",
            text_encoder=encoder,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            sample_paths=stage2_val_paths,
        )
    else:
        stage2_train = create_offline_dataloader(
            dataset_path=stage2_root,
            split=args.stage2_train_split,
            text_encoder=encoder,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_samples=args.stage2_max_samples,
            augment=True,
        )
        stage2_val = create_offline_dataloader(
            dataset_path=stage2_root,
            split=args.stage2_val_split,
            text_encoder=encoder,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=args.stage2_val_max_samples,
        )
    return stage1_train, stage1_val, stage2_train, stage2_val


def fit_stage(
    stage_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    encoder,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_path: Path,
    head_only_epochs: int = 0,
    freeze_cnn_epochs: int = 0,
    backbone_lr_scale: float = 0.1,
    head_lr_scale: float = 1.5,
    resume_from_checkpoint: bool = True,
) -> dict:
    criterion = nn.CTCLoss(blank=encoder.blank_index, zero_infinity=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if stage_name == "finetune" and freeze_cnn_epochs > 0:
        configure_finetune_trainability(model, 1, head_only_epochs, freeze_cnn_epochs)
    else:
        set_module_trainable(model.cnn, True)
        set_module_trainable(model.sequence_projection, True)
        set_module_trainable(model.rnn, True)
        set_module_trainable(model.classifier, True)

    optimizer = build_optimizer(model, lr, stage_name, backbone_lr_scale, head_lr_scale)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_val_cer = float("inf")
    best_epoch = 0
    start_epoch = 1

    if resume_from_checkpoint and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        saved_model_state = checkpoint.get("model_state_dict", {})
        if saved_model_state and checkpoint_is_compatible(model, saved_model_state):
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_val_cer = float(checkpoint.get("best_val_cer", best_val_cer))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
        else:
            print(
                f"[{stage_name}] skipping checkpoint resume because classifier classes do not match the current vocab.",
                flush=True,
            )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_cer": [],
        "train_wer": [],
        "val_loss": [],
        "val_cer": [],
        "val_wer": [],
        "learning_rate": [],
    }

    for epoch in range(start_epoch, epochs + 1):
        if stage_name == "finetune":
            configure_finetune_trainability(model, epoch, head_only_epochs, freeze_cnn_epochs)
            if epoch == head_only_epochs + 1 and head_only_epochs > 0:
                print(f"[{stage_name}] unfreezing encoder blocks after head warmup.", flush=True)
            elif epoch == freeze_cnn_epochs + 1 and freeze_cnn_epochs >= head_only_epochs:
                print(f"[{stage_name}] unfreezing CNN backbone for adaptive fine-tuning.", flush=True)

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, encoder, device, use_amp, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, None, encoder, device, False, None)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_cer"].append(train_metrics["cer"])
        history["train_wer"].append(train_metrics["wer"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_cer"].append(val_metrics["cer"])
        history["val_wer"].append(val_metrics["wer"])
        history["learning_rate"].append(float(max(group["lr"] for group in optimizer.param_groups)))

        print(
            f"[{stage_name}] epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} cer {train_metrics['cer']:.4f} wer {train_metrics['wer']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} cer {val_metrics['cer']:.4f} wer {val_metrics['wer']:.4f}"
        )

        if stage_name == "finetune" and (epoch == start_epoch or epoch % 5 == 0 or epoch == epochs):
            previews = preview_validation_examples(model, val_loader, encoder, device, max_examples=2, beam_width=5)
            print_validation_preview(stage_name, previews)

        if val_metrics["cer"] <= best_val_cer:
            best_val_cer = float(val_metrics["cer"])
            best_epoch = epoch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "encoder_vocab": encoder.vocab,
                    "hidden_size": model.classifier.in_features // 2,
                    "best_val_cer": best_val_cer,
                    "best_epoch": best_epoch,
                    "history": history,
                },
                checkpoint_path,
            )

        scheduler.step(val_metrics["cer"])

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "best_val_cer": best_val_cer,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }


def stage_config_summary(args: argparse.Namespace) -> dict:
    return {
        "stage1": {
            "dataset_path": str(args.stage1_dataset_path),
            "train_split": args.stage1_train_split,
            "val_split": args.stage1_val_split,
            "epochs": args.epochs_stage1,
            "learning_rate": args.lr_stage1,
        },
        "run_finetune": args.run_finetune,
        "stage2": {
            "dataset_path": str(args.stage2_dataset_path),
            "train_split": args.stage2_train_split,
            "val_split": args.stage2_val_split,
            "epochs": args.epochs_stage2,
            "learning_rate": args.lr_stage2,
            "head_only_epochs": args.stage2_head_only_epochs,
            "freeze_cnn_epochs": args.stage2_freeze_cnn_epochs,
            "backbone_lr_scale": args.stage2_backbone_lr_scale,
            "head_lr_scale": args.stage2_head_lr_scale,
        },
    }


def format_stage_summary_table(stage_results: dict) -> str:
    rows = [
        "Stage         | Best Epoch | Best CER | Checkpoint",
        "--------------|------------|----------|----------------------------",
    ]
    for stage_name, result in stage_results.items():
        rows.append(
            f"{stage_name:<13} | {result['best_epoch']:>10} | {result['best_val_cer']:.4f}  | {result['checkpoint']}"
        )
    return "\n".join(rows)


def build_final_report(
    model: nn.Module,
    dataloader,
    encoder,
    device: torch.device,
    checkpoint_path: Path,
    stage_label: str,
    max_error_examples: int = 20,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CTCLoss(blank=encoder.blank_index, zero_infinity=True)
    eval_metrics = run_epoch(model, dataloader, criterion, None, encoder, device, False, None)
    predictions = eval_metrics.pop("predictions")
    references = eval_metrics.pop("references")

    substitution_counter: Counter[tuple[str, str]] = Counter()
    error_examples: list[str] = []
    for prediction, reference in zip(predictions, references):
        if prediction == reference:
            continue
        alignment = backtrace_alignment(reference, prediction)
        substitution_counter.update(
            (ref_char, pred_char)
            for op, ref_char, pred_char in alignment
            if op == "sub" and ref_char and pred_char
        )
        if len(error_examples) < max_error_examples:
            error_examples.append(
                f"REF: {reference}\nPRED: {prediction}\nALIGN: {alignment}\n"
            )

    top_substitutions = [
        f"{ref_char!r} -> {pred_char!r}: {count}"
        for (ref_char, pred_char), count in substitution_counter.most_common(20)
    ]
    if not top_substitutions:
        top_substitutions = ["No substitutions recorded."]

    latex_table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\begin{tabular}{lccc}",
                "Stage & Best epoch & CER & WER " + r"\\",
            r"\hline",
                f"{stage_label} & {checkpoint.get('best_epoch', '?')} & {eval_metrics['cer']:.4f} & {eval_metrics['wer']:.4f} " + r"\\",
            r"\end{tabular}",
            r"\caption{Offline CRNN evaluation summary.}",
            r"\end{table}",
        ]
    )

    summary_lines = [
        "Offline CRNN final evaluation",
        f"Checkpoint: {checkpoint_path}",
        f"Samples: {len(references)}",
        f"Loss: {eval_metrics['loss']:.4f}",
        f"CER: {eval_metrics['cer']:.4f}",
        f"WER: {eval_metrics['wer']:.4f}",
        "",
        "Top substitution pairs:",
        *[f"  {line}" for line in top_substitutions],
        "",
        "Representative errors:",
        *(error_examples or ["  None"]),
    ]

    return {
        "metrics": eval_metrics,
        "summary_lines": summary_lines,
        "latex_table": latex_table,
        "top_substitutions": top_substitutions,
        "error_examples": error_examples,
    }


def load_checkpoint_into_model(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state_dict = checkpoint["model_state_dict"]
    model_state_dict = model.state_dict()
    compatible_state_dict = {
        key: value
        for key, value in checkpoint_state_dict.items()
        if key in model_state_dict and model_state_dict[key].shape == value.shape
    }
    skipped_keys = sorted(set(checkpoint_state_dict) - set(compatible_state_dict))
    model_state_dict.update(compatible_state_dict)
    model.load_state_dict(model_state_dict)
    if skipped_keys:
        print(f"Loaded checkpoint with {len(compatible_state_dict)} compatible tensors; skipped {len(skipped_keys)} incompatible tensors.")
    return checkpoint


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    print("Starting offline training setup...", flush=True)

    stage1_path = Path(args.stage1_dataset_path)
    stage2_path = Path(args.stage2_dataset_path)
    print("Validating offline dataset paths...", flush=True)
    validate_dataset_paths(stage1_path, args.stage1_val_split, stage2_path, args.stage2_val_split)
    print("Building dataset analysis and shared vocabulary...", flush=True)

    vocab_sample_cap = args.max_vocab_samples
    explicit_caps = [
        cap for cap in (
            args.stage1_max_samples,
            args.stage1_val_max_samples,
            args.stage2_max_samples,
            args.stage2_val_max_samples,
        )
        if cap is not None
    ]
    if explicit_caps:
        vocab_sample_cap = min(vocab_sample_cap, *explicit_caps)

    data_analysis = build_data_analysis(
        stage1_path,
        args.stage1_val_split,
        stage2_path,
        args.stage2_val_split,
        args.stage2_validation_fraction,
        args.seed,
        stage1_train_max_samples=args.stage1_max_samples,
        stage1_val_max_samples=args.stage1_val_max_samples,
        stage2_max_samples=args.stage2_max_samples,
        stage2_val_max_samples=args.stage2_val_max_samples,
    )

    vocab_splits = list(
        dict.fromkeys(
            [args.stage1_train_split, args.stage1_val_split, args.stage2_train_split]
            + ([] if args.stage2_val_split == "auto" else [args.stage2_val_split])
        )
    )
    encoder = build_text_encoder_for_dataset_paths(
        dataset_paths=[stage1_path, stage2_path],
        splits=vocab_splits,
        max_samples_per_path=vocab_sample_cap,
    )

    print("Creating dataloaders and model...", flush=True)
    stage1_train, stage1_val, stage2_train, stage2_val = build_loaders(args, encoder)
    model = CRNN(num_classes=encoder.num_classes, hidden_size=args.hidden_size).to(device)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_ROOT / "data_analysis.json", "w", encoding="utf-8") as file:
        json.dump(data_analysis, file, indent=2)

    print(f"Using device: {device}")
    print(f"CTC classes: {encoder.num_classes}")
    print(f"Stage 1 dataset path: {stage1_path}")
    print(f"Stage 2 dataset path: {stage2_path}")
    if args.stage2_val_split == "auto":
        print(f"Stage 2 validation: {args.stage2_validation_fraction:.0%} holdout from train")

    pretrained_checkpoint_path = Path(args.pretrained_checkpoint)
    if args.skip_stage1:
        if not pretrained_checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint_path}")
        load_checkpoint_into_model(model, pretrained_checkpoint_path, device)
        stage1_result = {
            "best_val_cer": float("nan"),
            "best_epoch": 0,
            "checkpoint": str(pretrained_checkpoint_path),
            "history": [],
            "skipped": True,
        }
        print(f"Loaded pretrained checkpoint from {pretrained_checkpoint_path}")
    else:
        stage1_result = fit_stage(
            stage_name="pretrain",
            model=model,
            train_loader=stage1_train,
            val_loader=stage1_val,
            encoder=encoder,
            device=device,
            epochs=args.epochs_stage1,
            lr=args.lr_stage1,
            checkpoint_path=pretrained_checkpoint_path,
        )

    if args.run_finetune:
        stage2_result = fit_stage(
            stage_name="finetune",
            model=model,
            train_loader=stage2_train,
            val_loader=stage2_val,
            encoder=encoder,
            device=device,
            epochs=args.epochs_stage2,
            lr=args.lr_stage2,
            checkpoint_path=CHECKPOINT_ROOT / "finetuned.pth",
            head_only_epochs=args.stage2_head_only_epochs,
            freeze_cnn_epochs=args.stage2_freeze_cnn_epochs,
            backbone_lr_scale=args.stage2_backbone_lr_scale,
            head_lr_scale=args.stage2_head_lr_scale,
            resume_from_checkpoint=False,
        )
        final_stage_label = "Final fine-tuned"
        final_checkpoint_path = CHECKPOINT_ROOT / "finetuned.pth"
        final_dataloader = stage2_val
    else:
        stage2_result = None
        final_stage_label = "Stage 1 pretrain"
        final_checkpoint_path = pretrained_checkpoint_path
        final_dataloader = stage1_val

    report = build_final_report(
        model=model,
        dataloader=final_dataloader,
        encoder=encoder,
        device=device,
        checkpoint_path=final_checkpoint_path,
        stage_label=final_stage_label,
    )

    stage_results = {"pretrain": stage1_result}
    if stage2_result is not None:
        stage_results["finetune"] = stage2_result

    metrics = {
        "config": stage_config_summary(args),
        "device": str(device),
        "encoder_classes": encoder.num_classes,
        "data_analysis": data_analysis,
        "stages": {
            "pretrain": stage1_result,
            **({"finetune": stage2_result} if stage2_result is not None else {}),
        },
        "final_report": {
            "metrics": report["metrics"],
            "top_substitutions": report["top_substitutions"],
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join([
            "TRAINING SUMMARY",
            "================",
            format_stage_summary_table(stage_results),
            "",
            f"Final CER: {report['metrics']['cer']:.4f}",
            f"Final WER: {report['metrics']['wer']:.4f}",
            f"Metrics JSON: {METRICS_PATH}",
            f"Error log: {ERROR_LOG_PATH}",
            f"LaTeX table: {LATEX_TABLE_PATH}",
        ]) + "\n")

    with open(ERROR_LOG_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(report["summary_lines"]) + "\n")

    with open(LATEX_TABLE_PATH, "w", encoding="utf-8") as file:
        file.write(report["latex_table"] + "\n")

    print(f"Saved checkpoints to {CHECKPOINT_ROOT}")
    print(f"Saved training metrics to {METRICS_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved error log to {ERROR_LOG_PATH}")
    print(f"Saved LaTeX table to {LATEX_TABLE_PATH}")


if __name__ == "__main__":
    main()