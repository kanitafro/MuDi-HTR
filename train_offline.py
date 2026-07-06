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
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW

from models.offline import (
    CRNN,
    OfflineHandwritingDataset,
    build_text_encoder_for_dataset_splits,
    create_offline_dataloader,
    resolve_processed_dataset_root,
)


RESULTS_ROOT = Path("experiments/results/offline")
CHECKPOINT_ROOT = Path("checkpoints/offline")
DEFAULT_STAGE1_DATASET = "to-be/OpenHand-Synth"
DEFAULT_STAGE2_DATASET = "GNHK"
DEFAULT_STAGE2_VALIDATION_FRACTION = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the offline CRNN in two stages.")
    parser.add_argument("--data-root", default="data/processed/offline", help="Processed offline data root.")
    parser.add_argument("--stage1-dataset", default=DEFAULT_STAGE1_DATASET, help="Dataset id used for pretraining.")
    parser.add_argument("--stage1-train-split", default="train", help="Training split for stage 1.")
    parser.add_argument("--stage1-val-split", default="val", help="Validation split for stage 1.")
    parser.add_argument("--stage2-dataset", default=DEFAULT_STAGE2_DATASET, help="Dataset id used for fine-tuning.")
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
    parser.add_argument("--run-finetune", action="store_true", help="Run stage 2 fine-tuning after pretraining.")
    parser.add_argument("--lr-stage1", type=float, default=1e-3, help="Learning rate for stage 1.")
    parser.add_argument("--lr-stage2", type=float, default=2e-5, help="Learning rate for stage 2.")
    parser.add_argument("--stage2-freeze-cnn-epochs", type=int, default=2, help="Freeze the CNN backbone for this many fine-tuning epochs.")
    parser.add_argument("--stage2-backbone-lr-scale", type=float, default=0.1, help="LR multiplier for the CNN backbone during stage 2.")
    parser.add_argument("--stage2-head-lr-scale", type=float, default=1.5, help="LR multiplier for the classifier head during stage 2.")
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


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def build_optimizer(
    model: nn.Module,
    lr: float,
    stage_name: str,
    backbone_lr_scale: float,
    head_lr_scale: float,
) -> AdamW:
    if stage_name != "finetuned":
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


def split_sample_paths(sample_paths: list[Path], validation_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
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

    avg_loss = total_loss / max(1, total_samples)
    cer, wer = compute_cer_wer(all_predictions, all_references)
    return {"loss": avg_loss, "cer": cer, "wer": wer}


def build_loaders(args: argparse.Namespace, encoder) -> tuple[object, object, object, object]:
    stage1_root = resolve_processed_dataset_root(args.data_root, args.stage1_dataset)
    stage2_root = resolve_processed_dataset_root(args.data_root, args.stage2_dataset)

    stage1_train = create_offline_dataloader(
        dataset_path=stage1_root,
        split=args.stage1_train_split,
        dataset_name=args.stage1_dataset,
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
        dataset_name=args.stage1_dataset,
        text_encoder=encoder,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.stage1_val_max_samples,
    )
    if args.stage2_val_split == "auto":
        stage2_dataset = OfflineHandwritingDataset(
            split=args.stage2_train_split,
            dataset_path=stage2_root,
            dataset_name=args.stage2_dataset,
            text_encoder=encoder,
            max_samples=args.stage2_max_samples,
        )
        stage2_sample_paths = stage2_dataset.sample_paths
        stage2_train_paths, stage2_val_paths = split_sample_paths(
            stage2_sample_paths,
            validation_fraction=args.stage2_validation_fraction,
            seed=args.seed,
        )
        stage2_train = create_offline_dataloader(
            dataset_path=stage2_root,
            split=args.stage2_train_split,
            dataset_name=args.stage2_dataset,
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
            dataset_name=args.stage2_dataset,
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
            dataset_name=args.stage2_dataset,
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
    freeze_cnn_epochs: int = 0,
    backbone_lr_scale: float = 0.1,
    head_lr_scale: float = 1.5,
) -> dict:
    criterion = nn.CTCLoss(blank=encoder.blank_index, zero_infinity=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    if stage_name == "finetuned" and freeze_cnn_epochs > 0:
        set_module_trainable(model.cnn, False)
    else:
        set_module_trainable(model.cnn, True)

    optimizer = build_optimizer(model, lr, stage_name, backbone_lr_scale, head_lr_scale)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_val_cer = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        if stage_name == "finetuned" and freeze_cnn_epochs > 0 and epoch == freeze_cnn_epochs + 1:
            set_module_trainable(model.cnn, True)
            print(f"[{stage_name}] unfreezing CNN backbone for adaptive fine-tuning.")

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

            scheduler.step(val_metrics["cer"])

    return {
        "best_val_cer": best_val_cer,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def build_final_report(
    model: nn.Module,
    dataloader,
    encoder,
    device: torch.device,
    checkpoint_path: Path,
    stage_label: str | None = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CTCLoss(blank=encoder.blank_index, zero_infinity=True)
    eval_metrics = run_epoch(model, dataloader, criterion, None, encoder, device, False, None)

    summary_lines = [
        "Offline CRNN final evaluation",
        f"Checkpoint: {checkpoint_path}",
        f"Samples: {len(eval_metrics['references'])}",
        f"Loss: {eval_metrics['loss']:.4f}",
        f"CER: {eval_metrics['cer']:.4f}",
        f"WER: {eval_metrics['wer']:.4f}",
    ]

    if stage_label:
        summary_lines.insert(1, f"Stage: {stage_label}")

    return {
        "metrics": eval_metrics,
        "summary_lines": summary_lines,
        "top_substitutions": [],
        "error_examples": [],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    stage_vocab_pairs = [
        (args.stage1_dataset, (args.stage1_train_split, args.stage1_val_split)),
        (args.stage2_dataset, (args.stage2_train_split, args.stage2_val_split)),
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
        stage_name="pretrain",
        model=model,
        train_loader=stage1_train,
        val_loader=stage1_val,
        encoder=encoder,
        device=device,
        epochs=args.epochs_stage1,
        lr=args.lr_stage1,
        checkpoint_path=CHECKPOINT_ROOT / "pretrained.pth",
    )

    if args.run_finetune:
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
            freeze_cnn_epochs=args.stage2_freeze_cnn_epochs,
            backbone_lr_scale=args.stage2_backbone_lr_scale,
            head_lr_scale=args.stage2_head_lr_scale,
        )
        report = build_final_report(
            model=model,
            dataloader=stage2_val,
            encoder=encoder,
            device=device,
            checkpoint_path=CHECKPOINT_ROOT / "finetuned.pth",
        )
    else:
        stage2_result = None
        report = build_final_report(
            model=model,
            dataloader=stage1_val,
            encoder=encoder,
            device=device,
            checkpoint_path=CHECKPOINT_ROOT / "pretrained.pth",
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
