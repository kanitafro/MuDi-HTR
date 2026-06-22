"""Offline CRNN training for MuDi-HTR.

This script follows the project plan in a compact way:
- stage 1: pretrain on OpenHand-Synth
- stage 2: fine-tune on GNHK
- track CER/WER
- save checkpoints and a short JSON summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from models.offline import (
    CRNN,
    build_text_encoder_for_dataset_splits,
    create_offline_dataloader,
)


RESULTS_ROOT = Path("experiments/results/offline")
CHECKPOINT_ROOT = Path("checkpoints/offline")
DEFAULT_STAGE1_DATASET = "to-be/OpenHand-Synth"
DEFAULT_STAGE2_DATASET = "your-org/GNHK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the offline CRNN in two stages.")
    parser.add_argument("--data-root", default="data/processed/offline", help="Processed offline data root.")
    parser.add_argument("--stage1-dataset", default=DEFAULT_STAGE1_DATASET, help="Dataset id used for pretraining.")
    parser.add_argument("--stage1-train-split", default="train", help="Training split for stage 1.")
    parser.add_argument("--stage1-val-split", default="test", help="Validation split for stage 1.")
    parser.add_argument("--stage2-dataset", default=DEFAULT_STAGE2_DATASET, help="Dataset id used for fine-tuning.")
    parser.add_argument("--stage2-train-split", default="train", help="Training split for stage 2.")
    parser.add_argument("--stage2-val-split", default="test", help="Validation split for stage 2.")
    parser.add_argument("--max-vocab-samples", type=int, default=50000, help="Cap samples used to build the shared vocab.")
    parser.add_argument("--stage1-max-samples", type=int, default=None, help="Optional cap for stage 1 train samples.")
    parser.add_argument("--stage1-val-max-samples", type=int, default=None, help="Optional cap for stage 1 val samples.")
    parser.add_argument("--stage2-max-samples", type=int, default=None, help="Optional cap for stage 2 train samples.")
    parser.add_argument("--stage2-val-max-samples", type=int, default=None, help="Optional cap for stage 2 val samples.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs-stage1", type=int, default=15, help="Pretraining epochs.")
    parser.add_argument("--epochs-stage2", type=int, default=10, help="Fine-tuning epochs.")
    parser.add_argument("--lr-stage1", type=float, default=1e-3, help="Learning rate for stage 1.")
    parser.add_argument("--lr-stage2", type=float, default=1e-4, help="Learning rate for stage 2.")
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

    cer = char_errors / max(1, char_total)
    wer = word_errors / max(1, word_total)
    return cer, wer


@torch.no_grad()
def greedy_decode(logits: torch.Tensor, encoder) -> list[str]:
    token_ids = logits.argmax(dim=-1).permute(1, 0).tolist()
    return [encoder.decode(sequence) for sequence in token_ids]


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer | None,
    encoder,
    device: torch.device,
    use_amp: bool,
    scaler: GradScaler | None,
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

        with autocast(enabled=use_amp):
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

    avg_loss = total_loss / max(1, total_samples)
    cer, wer = compute_cer_wer(all_predictions, all_references)
    return {"loss": avg_loss, "cer": cer, "wer": wer}


def build_loaders(args: argparse.Namespace, encoder) -> tuple[object, object, object, object]:
    stage1_train = create_offline_dataloader(
        root_dir=args.data_root,
        split=args.stage1_train_split,
        dataset_name=args.stage1_dataset,
        text_encoder=encoder,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_samples=args.stage1_max_samples,
    )
    stage1_val = create_offline_dataloader(
        root_dir=args.data_root,
        split=args.stage1_val_split,
        dataset_name=args.stage1_dataset,
        text_encoder=encoder,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.stage1_val_max_samples,
    )
    stage2_train = create_offline_dataloader(
        root_dir=args.data_root,
        split=args.stage2_train_split,
        dataset_name=args.stage2_dataset,
        text_encoder=encoder,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_samples=args.stage2_max_samples,
    )
    stage2_val = create_offline_dataloader(
        root_dir=args.data_root,
        split=args.stage2_val_split,
        dataset_name=args.stage2_dataset,
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
) -> dict:
    criterion = nn.CTCLoss(blank=encoder.blank_index, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    best_val_cer = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, encoder, device, use_amp, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, None, encoder, device, False, None)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"[{stage_name}] epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} cer {train_metrics['cer']:.4f} wer {train_metrics['wer']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} cer {val_metrics['cer']:.4f} wer {val_metrics['wer']:.4f}"
        )

        if val_metrics["cer"] < best_val_cer:
            best_val_cer = val_metrics["cer"]
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
                    "history": history,
                },
                checkpoint_path,
            )

    return {
        "best_val_cer": best_val_cer,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    stage_vocab_pairs = [
        (args.stage1_dataset, args.stage1_train_split),
        (args.stage2_dataset, args.stage2_train_split),
    ]
    encoder = build_text_encoder_for_dataset_splits(
        root_dir=args.data_root,
        dataset_split_pairs=stage_vocab_pairs,
        max_samples_per_split=args.max_vocab_samples,
    )

    stage1_train, stage1_val, stage2_train, stage2_val = build_loaders(args, encoder)
    model = CRNN(num_classes=encoder.num_classes, hidden_size=args.hidden_size).to(device)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"CTC classes: {encoder.num_classes}")
    print(f"Stage 1 dataset: {args.stage1_dataset}")
    print(f"Stage 2 dataset: {args.stage2_dataset}")

    stage1_result = fit_stage(
        stage_name="pretrained",
        model=model,
        train_loader=stage1_train,
        val_loader=stage1_val,
        encoder=encoder,
        device=device,
        epochs=args.epochs_stage1,
        lr=args.lr_stage1,
        checkpoint_path=CHECKPOINT_ROOT / "pretrained.pth",
    )

    stage2_result = fit_stage(
        stage_name="finetuned",
        model=model,
        train_loader=stage2_train,
        val_loader=stage2_val,
        encoder=encoder,
        device=device,
        epochs=args.epochs_stage2,
        lr=args.lr_stage2,
        checkpoint_path=CHECKPOINT_ROOT / "finetuned.pth",
    )

    summary = {
        "device": str(device),
        "encoder_classes": encoder.num_classes,
        "stage1": stage1_result,
        "stage2": stage2_result,
    }
    summary_path = RESULTS_ROOT / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved checkpoints to {CHECKPOINT_ROOT}")
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
