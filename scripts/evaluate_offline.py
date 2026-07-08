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
DEFAULT_DATASET_NAME = "GNHK"
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
    parser.add_argument(
        "--fail-on-vocab-mismatch",
        action="store_true",
        help="Exit with error if checkpoint encoder vocab and model classifier size disagree.",
    )
    parser.add_argument(
        "--dump-vocab-mappings",
        action="store_true",
        help="If set, attempt to build a dataset-derived encoder and print side-by-side mappings for inspection.",
    )
    parser.add_argument(
        "--line-segment",
        action="store_true",
        help="Segment paragraph images into horizontal line bands and run inference per-line, then join predictions for scoring.",
    )
    parser.add_argument(
        "--min-line-height",
        type=int,
        default=6,
        help="Minimum pixel height for a detected line band (rows).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for quick tests).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert image polarity (255 - image) before inference; useful when foreground/background is swapped.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none", "clahe", "otsu", "gamma", "histeq"],
        default="none",
        help="Optional simple preprocessing to apply to images before inference.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _apply_clahe_numpy(arr: np.ndarray) -> np.ndarray:
    # arr expected in [0,1]
    try:
        import cv2

        img = (arr * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(img)
        return out.astype(np.float32) / 255.0
    except Exception:
        # fallback to histogram equalization
        hist, bins = np.histogram(arr.flatten(), 256, [0, 1])
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) / max(1, (cdf.max() - cdf.min()))
        out = np.interp(arr.flatten(), bins[:-1], cdf).reshape(arr.shape)
        return out.astype(np.float32)


def _apply_histeq_numpy(arr: np.ndarray) -> np.ndarray:
    # simple histogram equalization on [0,1] floats
    hist, bins = np.histogram(arr.flatten(), 256, [0, 1])
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / max(1, (cdf.max() - cdf.min()))
    out = np.interp(arr.flatten(), bins[:-1], cdf).reshape(arr.shape)
    return out.astype(np.float32)


def _apply_otsu_numpy(arr: np.ndarray) -> np.ndarray:
    try:
        import cv2

        img = (arr * 255).astype(np.uint8)
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (th.astype(np.float32) / 255.0)
    except Exception:
        # simple mean-based threshold fallback
        thr = arr.mean()
        return (arr < thr).astype(np.float32)


def preprocess_tensor(img: torch.Tensor, method: str = "none", invert: bool = False) -> torch.Tensor:
    """Apply simple preprocessing to a single image tensor of shape (1,H,W).
    Tensor values expected to be float, roughly in [0,1]. Returns same shape and dtype float32.
    """
    if img.ndim != 3 or img.shape[0] != 1:
        return img

    arr = img.squeeze(0).detach().cpu().numpy().astype(np.float32)
    # clamp/assume 0..1
    arr = np.clip(arr, 0.0, 1.0)

    if invert:
        arr = 1.0 - arr

    if method == "none":
        out = arr
    elif method == "clahe":
        out = _apply_clahe_numpy(arr)
    elif method == "otsu":
        out = _apply_otsu_numpy(arr)
    elif method == "gamma":
        gamma = 0.8
        out = np.power(arr, gamma)
    elif method == "histeq":
        out = _apply_histeq_numpy(arr)
    else:
        out = arr

    out_t = torch.from_numpy(out.astype(np.float32)).unsqueeze(0)
    return out_t


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
    line_segment: bool = False,
    min_line_height: int = 6,
    preprocess: str = "none",
    invert: bool = False,
) -> tuple[float, float, list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []

    total_batches = len(dataloader)
    print(f"Evaluating {total_batches} batches with beam width {beam_width}...", flush=True)

    import torch.nn.functional as F
    import numpy as np

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["images"]
        # apply preprocessing on CPU per-sample
        B = images.shape[0]
        processed_images: list[torch.Tensor] = []
        for b in range(B):
            img = images[b]
            img_proc = preprocess_tensor(img, method=preprocess, invert=invert)
            processed_images.append(img_proc)
        images_cpu = torch.stack(processed_images, dim=0)
        images_dev = images_cpu.to(device)
        texts = batch["texts"]

        batch_predictions: list[str] = []

        # Process samples individually to auto-detect large paragraph images
        for b in range(B):
            img_cpu = images_cpu[b]
            H, W = img_cpu.shape[1], img_cpu.shape[2]
            do_segment = line_segment or (H > 256 or W > 1024)

            if not do_segment:
                inp = images_dev[b : b + 1]
                with torch.inference_mode():
                    logits = model(inp)
                decoded = model.decode_beam_search(logits, alphabet=alphabet, beam_width=beam_width)
                batch_predictions.append(decoded[0][0])
            else:
                # Per-sample line segmentation + per-line decoding
                img = img_cpu.detach()
                if img.ndim != 3 or img.shape[0] != 1:
                    raise ValueError(f"Expected image tensor (1,H,W), got {tuple(img.shape)}")
                arr = img.squeeze(0).numpy()
                # horizontal projection
                row_mean = arr.mean(axis=1)
                thr = float(np.clip(row_mean.mean() - 0.25 * row_mean.std(), 0.0, 1.0))
                mask = row_mean < thr
                # group contiguous True runs
                bands: list[tuple[int, int]] = []
                i = 0
                Hm = mask.shape[0]
                while i < Hm:
                    if not mask[i]:
                        i += 1
                        continue
                    j = i
                    while j < Hm and mask[j]:
                        j += 1
                    a = max(0, i - 2)
                    b_end = min(Hm, j + 2)
                    if (b_end - a) >= min_line_height:
                        bands.append((a, b_end))
                    i = j + 1
                if not bands:
                    bands = [(0, Hm)]

                per_line_preds: list[str] = []
                for (a, b_end) in bands:
                    crop = img[:, a:b_end, :].unsqueeze(0)
                    # Preserve aspect: scale line height -> 128, width scaled proportionally
                    _, h, w = crop.shape
                    target_h = 128
                    scaled_w = max(1, int(round(float(w) * (target_h / float(h)))))
                    try:
                        if scaled_w <= 512:
                            crop_resized = F.interpolate(crop, size=(target_h, scaled_w), mode="bilinear", align_corners=False)
                            pad_total = 512 - scaled_w
                            pad_left = pad_total // 2
                            pad_right = pad_total - pad_left
                            crop_resized = F.pad(crop_resized, (pad_left, pad_right, 0, 0), value=1.0)
                        else:
                            crop_resized = F.interpolate(crop, size=(target_h, 512), mode="bilinear", align_corners=False)
                    except Exception:
                        crop_resized = F.interpolate(crop, size=(128, 512), mode="bilinear", align_corners=False)
                    crop_resized = crop_resized.to(device)
                    with torch.inference_mode():
                        logits = model(crop_resized)
                    decoded = model.decode_beam_search(logits, alphabet=alphabet, beam_width=beam_width)
                    per_line_preds.append(decoded[0][0])

                joined = " ".join([p for p in per_line_preds if p])
                batch_predictions.append(joined)

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

    # Diagnostic: verify classifier size matches encoder vocab length
    try:
        classifier_size = int(model.classifier.weight.shape[0])
    except Exception:
        classifier_size = None

    print(f"Loaded checkpoint encoder vocab size: {len(encoder.vocab)}", flush=True)
    print(f"Loaded model classifier size: {classifier_size}", flush=True)

    if classifier_size is not None and classifier_size != len(encoder.vocab):
        print("WARNING: classifier size does not match encoder vocab size.", flush=True)
        print(f"  encoder_vocab_len={len(encoder.vocab)} classifier_size={classifier_size}", flush=True)
        if args.dump_vocab_mappings:
            # Attempt to build a dataset-derived encoder for comparison
            try:
                from models.offline import build_text_encoder_for_dataset_splits

                print("Attempting to build dataset-derived encoder for mapping...")
                dataset_encoder = build_text_encoder_for_dataset_splits(
                    args.data_root, [(args.dataset_name, args.split)], max_samples_per_split=200
                )
                print(f"Dataset-derived encoder vocab size: {len(dataset_encoder.vocab)}")
                print("Index | checkpoint_token | dataset_token")
                for i in range(min(60, max(len(encoder.vocab), len(dataset_encoder.vocab)))):
                    a = encoder.vocab[i] if i < len(encoder.vocab) else "<MISSING>"
                    b = dataset_encoder.vocab[i] if i < len(dataset_encoder.vocab) else "<MISSING>"
                    print(f"{i:3d} | {a!r:20} | {b!r}")
            except Exception as e:
                print(f"Failed to build dataset encoder for mapping: {e}", flush=True)

        if args.fail_on_vocab_mismatch:
            raise SystemExit("Failing due to vocab/classifier size mismatch as requested by --fail-on-vocab-mismatch")

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
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(dataloader.dataset)} samples.", flush=True)

    cer, wer, predictions, references = evaluate_model(
        model=model,
        dataloader=dataloader,
        alphabet=encoder.vocab,
        beam_width=args.beam_width,
        device=device,
        line_segment=args.line_segment,
        min_line_height=args.min_line_height,
        preprocess=args.preprocess,
        invert=args.invert,
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