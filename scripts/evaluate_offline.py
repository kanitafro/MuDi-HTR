"""Offline CRNN beam-search evaluation for MuDi-HTR.

This script loads the fine-tuned offline checkpoint, runs inference on the
offline test split, decodes the logits with the CRNN's pure Python/NumPy CTC
beam search decoder, and reports overall CER/WER.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from models.offline import (
    CRNN,
    CTCTextEncoder,
    create_offline_dataloader,
    resolve_processed_dataset_root,
)


DEFAULT_DATA_ROOT = Path("data/processed/offline")
DEFAULT_DATASET_NAME = "Kaggle/handwriting-recognition"
DEFAULT_CHECKPOINT = Path("checkpoints/offline/pretrained.pth")
DEFAULT_TEST_SPLIT = "test"


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, pool: tuple[int, int]) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )


class LegacyCRNN(nn.Module):
    def __init__(self, num_classes: int = 80, hidden_size: int = 256) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(1, 64, (2, 2)),
            ConvBlock(64, 128, (2, 2)),
            ConvBlock(128, 256, (2, 1)),
            ConvBlock(256, 256, (2, 1)),
            ConvBlock(256, 512, (2, 1)),
        )
        self.final_pool = nn.MaxPool2d((4, 1))
        self.sequence_projection = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.25,
        )
        self.sequence_dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {x.shape}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 channel (grayscale), got {x.shape[1]}")
        if x.shape[2] != 128 or x.shape[3] != 512:
            raise ValueError(f"Expected (128, 512) input size, got {x.shape[2:]}")
        if x.dtype != torch.float32:
            raise TypeError(f"Expected float32 input, got {x.dtype}")

        features = self.cnn(x)
        features = self.final_pool(features)

        _, _, height, _ = features.shape
        if height != 1:
            raise ValueError(f"Expected pooled feature height of 1, got {height}.")

        sequence = features.squeeze(2).permute(0, 2, 1)
        sequence = self.sequence_projection(sequence)
        sequence, _ = self.rnn(sequence)
        sequence = self.sequence_dropout(sequence)
        logits = self.classifier(sequence)
        return logits.permute(1, 0, 2)


def _logsumexp(values: list[float]) -> float:
    finite_values = [value for value in values if value > -np.inf]
    if not finite_values:
        return -np.inf
    max_value = max(finite_values)
    return float(max_value + np.log(sum(np.exp(value - max_value) for value in finite_values)))


def _ctc_beam_search_single(
    log_probs: np.ndarray,
    alphabet: list[str],
    beam_width: int = 10,
    blank_index: int = 0,
) -> tuple[str, float]:
    if log_probs.ndim != 2:
        raise ValueError(f"Expected 2D log-probabilities (T, C), got shape {log_probs.shape}")
    if log_probs.shape[1] != len(alphabet):
        raise ValueError(f"Alphabet size {len(alphabet)} does not match logits classes {log_probs.shape[1]}")

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -np.inf)}
    candidate_width = min(max(beam_width * 2, 10), max(1, log_probs.shape[1] - 1))

    for timestep in log_probs:
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        top_indices = np.argsort(timestep)[::-1][:candidate_width]

        def update(prefix: tuple[int, ...], blank_score: float | None = None, nonblank_score: float | None = None) -> None:
            existing_blank, existing_nonblank = next_beams.get(prefix, (-np.inf, -np.inf))
            if blank_score is not None:
                existing_blank = _logsumexp([existing_blank, blank_score])
            if nonblank_score is not None:
                existing_nonblank = _logsumexp([existing_nonblank, nonblank_score])
            next_beams[prefix] = (existing_blank, existing_nonblank)

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = _logsumexp([p_blank, p_nonblank])
            update(prefix, blank_score=total + float(timestep[blank_index]))

            for class_index in top_indices:
                if class_index == blank_index:
                    continue
                score = float(timestep[class_index])
                if prefix and prefix[-1] == class_index:
                    update(prefix, nonblank_score=p_nonblank + score)
                    update(prefix + (class_index,), nonblank_score=p_blank + score)
                else:
                    update(prefix + (class_index,), nonblank_score=total + score)

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _logsumexp([item[1][0], item[1][1]]),
                reverse=True,
            )[:beam_width]
        )

    best_prefix, (best_blank, best_nonblank) = max(
        beams.items(),
        key=lambda item: _logsumexp([item[1][0], item[1][1]]),
    )
    best_score = _logsumexp([best_blank, best_nonblank])
    total_score = _logsumexp([_logsumexp([blank, nonblank]) for blank, nonblank in beams.values()])

    characters: list[str] = []
    last_token: int | None = None
    for token in best_prefix:
        if token == blank_index or token == last_token:
            last_token = token
            continue
        last_token = token
        if 0 <= token < len(alphabet):
            token_text = alphabet[token]
            if token_text not in {"<BLANK>", "<UNK>"}:
                characters.append(token_text)

    confidence = float(np.exp(best_score - total_score)) if total_score > -np.inf else 0.0
    return "".join(characters), confidence


class BeamSearchDecoderMixin:
    def decode_beam_search(
        self,
        logits: torch.Tensor,
        alphabet: list[str],
        beam_width: int = 10,
        blank_index: int = 0,
    ) -> list[tuple[str, float]]:
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        if logits.ndim != 3:
            raise ValueError(f"Expected logits with shape (T, B, C) or (T, C), got {tuple(logits.shape)}")

        log_probs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
        decoded: list[tuple[str, float]] = []
        for batch_index in range(log_probs.shape[1]):
            text, probability = _ctc_beam_search_single(
                log_probs[:, batch_index, :],
                alphabet=alphabet,
                beam_width=beam_width,
                blank_index=blank_index,
            )
            decoded.append((text, probability))
        return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the offline CRNN with beam search decoding.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Processed offline data root.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Dataset name used to resolve the processed root.")
    parser.add_argument("--split", default=DEFAULT_TEST_SPLIT, help="Dataset split to evaluate.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Fine-tuned checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--beam-width", type=int, default=10, help="CTC beam width.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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
        word_errors += edit_distance(prediction.split(), reference.split())
        word_total += max(1, len(reference.split()))

    cer = char_errors / max(1, char_total)
    wer = word_errors / max(1, word_total)
    return cer, wer


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, CTCTextEncoder]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    encoder_vocab = list(checkpoint.get("encoder_vocab") or [])
    if len(encoder_vocab) < 3:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a usable encoder_vocab."
        )

    hidden_size_key = "sequence_projection.0.weight" if "sequence_projection.0.weight" in state_dict else "sequence_projection.1.weight"
    hidden_size = int(
        checkpoint.get("hidden_size")
        or checkpoint.get("model_hidden_size")
        or state_dict[hidden_size_key].shape[0]
    )
    num_classes = int(state_dict["classifier.weight"].shape[0])

    if any(key.startswith("cnn.0.0.") for key in state_dict):
        model: nn.Module = LegacyCRNN(num_classes=num_classes, hidden_size=hidden_size).to(device)
    else:
        model = CRNN(num_classes=num_classes, hidden_size=hidden_size).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # The checkpoint stores [blank, unk, chars...]. CTCTextEncoder expects only chars.
    encoder = CTCTextEncoder("".join(encoder_vocab[2:]))

    if not hasattr(model, "decode_beam_search"):
        model.decode_beam_search = BeamSearchDecoderMixin.decode_beam_search.__get__(model, model.__class__)  # type: ignore[attr-defined]

    return model, encoder


def resolve_existing_split(dataset_root: Path, requested_split: str) -> str:
    candidate_splits = [requested_split, "test", "val", "validation", "valid", "train"]
    seen: set[str] = set()
    for split_name in candidate_splits:
        normalized = "val" if split_name in {"validation", "valid"} else split_name
        if normalized in seen:
            continue
        seen.add(normalized)
        if (dataset_root / normalized).exists():
            return normalized
    raise FileNotFoundError(f"No usable split directory found under {dataset_root}. Checked: {candidate_splits}")


@torch.inference_mode()
def evaluate_model(
    model: CRNN,
    dataloader,
    alphabet: list[str],
    beam_width: int,
    device: torch.device,
) -> tuple[float, float, list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []

    total_batches = len(dataloader)
    print(f"Evaluating {total_batches} batches with beam width {beam_width}...", flush=True)

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["images"].to(device)
        texts = batch["texts"]

        logits = model(images)
        decoded = model.decode_beam_search(logits, alphabet=alphabet, beam_width=beam_width)
        batch_predictions = [text for text, _confidence in decoded]

        predictions.extend(batch_predictions)
        references.extend(texts)

        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            print(f"  processed batch {batch_index}/{total_batches}", flush=True)

    cer, wer = compute_cer_wer(predictions, references)
    return cer, wer, predictions, references


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print("Starting offline beam-search evaluation...", flush=True)

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_root = resolve_processed_dataset_root(args.data_root, args.dataset_name)
    evaluation_split = resolve_existing_split(data_root, args.split)
    print(f"Using dataset root: {data_root}", flush=True)
    print(f"Using split: {evaluation_split} (requested: {args.split})", flush=True)
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    model, encoder = load_checkpoint(checkpoint_path, device)

    print("Building evaluation dataloader...", flush=True)
    dataloader = create_offline_dataloader(
        dataset_path=data_root,
        split=evaluation_split,
        dataset_name=args.dataset_name,
        text_encoder=encoder,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augment=False,
    )
    print(f"Loaded {len(dataloader.dataset)} samples.", flush=True)

    cer, wer, predictions, references = evaluate_model(
        model=model,
        dataloader=dataloader,
        alphabet=encoder.vocab,
        beam_width=args.beam_width,
        device=device,
    )

    print("Offline Beam Search Evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset root: {data_root}")
    print(f"Split: {evaluation_split} (requested: {args.split})")
    print(f"Samples: {len(predictions)}")
    print(f"Beam width: {args.beam_width}")
    print(f"Beam Search CER: {cer:.4f}")
    print(f"Beam Search WER: {wer:.4f}")


if __name__ == "__main__":
    main()