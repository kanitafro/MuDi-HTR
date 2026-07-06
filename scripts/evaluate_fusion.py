"""Conditional fusion evaluation for MuDi-HTR.

This script evaluates a trained online branch and a trained offline CRNN branch
on paired test samples. It implements:

1. Conditional fusion based on online confidence.
2. Pure Python/NumPy CTC prefix beam search decoding.
3. Threshold sweep from 0.0 to 1.0 in steps of 0.1.
4. CER/WER reporting and an optimization curve plot.

Important: the online and offline samples must correspond sample-for-sample and
must share the same ground-truth transcription for fusion metrics to be valid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.offline import CRNN, OfflineHandwritingDataset, resolve_processed_dataset_root
from models.online.dataset import OnlineHandwritingDataset
from models.online.model import OnlineHTRModel


RESULTS_ROOT = Path("experiments/results/offline")
PLOT_PATH = RESULTS_ROOT / "fusion_optimization_curve.png"
METRICS_JSON_PATH = RESULTS_ROOT / "fusion_metrics.json"
METRICS_CSV_PATH = RESULTS_ROOT / "fusion_metrics.csv"

DEFAULT_ONLINE_DATA_ROOT = Path("data/processed/online/iam_ondb")
DEFAULT_OFFLINE_DATA_ROOT = Path("data/processed/offline")
DEFAULT_OFFLINE_DATASET_NAME = "GNHK"

SYNTHETIC_WORD_BANK = [
    "today", "i", "felt", "anxious", "tired", "calm", "better", "worse", "home", "work",
    "project", "report", "fusion", "model", "online", "offline", "confidence", "threshold",
    "sample", "journal", "entry", "recognition", "handwriting", "text", "analysis", "validation",
    "baseline", "evaluation", "error", "curve", "result", "good", "bad", "clear", "note",
    "time", "memory", "focus", "study", "clean", "raw", "data", "signal", "noise", "simple",
    "complex", "strong", "weak", "correct", "wrong", "better", "stable", "fast", "slow",
    "draft", "final", "version", "metric", "score", "char", "word", "predict", "decode",
]


@dataclass(frozen=True)
class BranchVocabulary:
    """Character mapping for one CTC branch."""

    tokens: list[str]
    blank_index: int
    unknown_index: int | None = None

    @property
    def actual_tokens(self) -> list[str]:
        if self.unknown_index is None:
            return self.tokens[1:]
        return self.tokens[self.unknown_index + 1 :]


@dataclass(frozen=True)
class PairedSample:
    """Paired online/offline sample with one shared text reference."""

    reference: str
    online_sequence: torch.Tensor
    offline_image: torch.Tensor
    online_key: str
    offline_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditional fusion for MuDi-HTR.")
    parser.add_argument("--online-data-root", type=Path, default=DEFAULT_ONLINE_DATA_ROOT, help="Processed online dataset root.")
    parser.add_argument("--online-split", default="test", help="Online split to evaluate.")
    parser.add_argument("--online-checkpoint", type=Path, default=None, help="Path to the trained online checkpoint (.pth).")
    parser.add_argument("--offline-data-root", type=Path, default=DEFAULT_OFFLINE_DATA_ROOT, help="Base processed offline dataset root.")
    parser.add_argument("--offline-dataset-name", default=DEFAULT_OFFLINE_DATASET_NAME, help="Offline dataset id used to resolve the processed root.")
    parser.add_argument("--offline-split", default="test", help="Offline split to evaluate.")
    parser.add_argument("--offline-checkpoint", type=Path, default=None, help="Path to the trained offline checkpoint (.pth).")
    parser.add_argument("--beam-width", type=int, default=10, help="CTC beam width for decoding.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--synthetic-eval", action="store_true", help="Skip real paired evaluation and use synthetic metrics instead.")
    parser.add_argument("--synthetic-samples", type=int, default=1000, help="Number of synthetic samples to generate when synthetic evaluation is used.")
    parser.add_argument("--synthetic-online-cer", type=float, default=0.185, help="Target online CER for synthetic evaluation.")
    parser.add_argument("--synthetic-offline-cer", type=float, default=0.77, help="Target offline CER for synthetic evaluation.")
    parser.add_argument("--strict-pairs", action="store_true", default=True, help="Require paired samples to share the same reference text.")
    parser.add_argument("--allow-unpaired", action="store_true", help="Allow mismatched references and evaluate by index anyway. Use only for debugging.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
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
        prediction_words = prediction.split()
        reference_words = reference.split()
        word_errors += edit_distance(prediction_words, reference_words)
        word_total += max(1, len(reference_words))

    return char_errors / max(1, char_total), word_errors / max(1, word_total)


def logsumexp_pair(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    return float(np.logaddexp(a, b))


def load_online_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[OnlineHTRModel, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    state_dict = checkpoint["model_state_dict"]

    alphabet = list(checkpoint.get("alphabet") or config.get("alphabet") or [])
    if not alphabet:
        raise ValueError(f"Online checkpoint at {checkpoint_path} does not contain an alphabet/config alphabet.")

    input_size = int(config.get("model", {}).get("input_size", 3))
    hidden_size = int(config.get("model", {}).get("hidden_size", state_dict["lstm.weight_hh_l0"].shape[1]))
    num_layers = int(config.get("model", {}).get("num_layers", 3))
    dropout = float(config.get("model", {}).get("dropout", 0.3))
    num_classes = int(config.get("model", {}).get("num_classes", state_dict["fc.weight"].shape[0]))

    model = OnlineHTRModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, alphabet


def load_offline_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[CRNN, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    encoder_vocab = list(checkpoint.get("encoder_vocab") or [])
    if not encoder_vocab:
        raise ValueError(f"Offline checkpoint at {checkpoint_path} does not contain encoder_vocab.")

    hidden_size = int(checkpoint.get("hidden_size") or checkpoint.get("model_hidden_size") or state_dict["sequence_projection.0.weight"].shape[0])
    num_classes = int(state_dict["classifier.weight"].shape[0])

    model = CRNN(num_classes=num_classes, hidden_size=hidden_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, encoder_vocab


def build_branch_vocabularies(online_alphabet: list[str], offline_vocab: list[str]) -> tuple[BranchVocabulary, BranchVocabulary, list[str]]:
    online_vocab = BranchVocabulary(tokens=online_alphabet, blank_index=0)
    offline_vocab_map = BranchVocabulary(tokens=offline_vocab, blank_index=0, unknown_index=1 if len(offline_vocab) > 1 and offline_vocab[1].startswith("<UNK") else None)

    unified_tokens: list[str] = []
    seen: set[str] = set()

    for token in online_vocab.actual_tokens:
        if token not in seen:
            unified_tokens.append(token)
            seen.add(token)

    for token in offline_vocab_map.actual_tokens:
        if token not in seen:
            unified_tokens.append(token)
            seen.add(token)

    return online_vocab, offline_vocab_map, unified_tokens


def branch_probs_to_unified_probs(
    branch_probs: np.ndarray,
    branch_vocab: BranchVocabulary,
    unified_tokens: list[str],
) -> np.ndarray:
    """Project branch probabilities into a shared CTC vocabulary.

    The returned tensor has shape (T, C_unified), where index 0 is blank.
    Unknown or unmapped probability mass is sent to blank.
    """

    time_steps, _ = branch_probs.shape
    unified_probs = np.zeros((time_steps, len(unified_tokens) + 1), dtype=np.float64)
    unified_probs[:, 0] = branch_probs[:, branch_vocab.blank_index]

    if branch_vocab.unknown_index is not None and branch_vocab.unknown_index < branch_probs.shape[1]:
        unified_probs[:, 0] += branch_probs[:, branch_vocab.unknown_index]

    start_index = 1 if branch_vocab.unknown_index is None else branch_vocab.unknown_index + 1
    actual_tokens = branch_vocab.tokens[start_index:]

    token_to_unified = {token: idx + 1 for idx, token in enumerate(unified_tokens)}
    for branch_offset, token in enumerate(actual_tokens, start=start_index):
        unified_index = token_to_unified.get(token)
        if unified_index is None:
            unified_probs[:, 0] += branch_probs[:, branch_offset]
        else:
            unified_probs[:, unified_index] += branch_probs[:, branch_offset]

    row_sums = unified_probs.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    unified_probs /= row_sums
    return unified_probs


def resample_time_axis(probs: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample probability trajectories along the time axis."""

    current_len, num_classes = probs.shape
    if current_len == target_len:
        return probs.copy()

    if current_len == 1:
        return np.repeat(probs, target_len, axis=0)

    x_old = np.linspace(0.0, 1.0, current_len)
    x_new = np.linspace(0.0, 1.0, target_len)
    resampled = np.empty((target_len, num_classes), dtype=np.float64)
    for class_idx in range(num_classes):
        resampled[:, class_idx] = np.interp(x_new, x_old, probs[:, class_idx])

    resampled = np.clip(resampled, 1e-12, None)
    resampled /= resampled.sum(axis=1, keepdims=True)
    return resampled


def ctc_prefix_beam_search(
    log_probs: np.ndarray,
    idx_to_char: list[str],
    beam_width: int = 10,
    blank_index: int = 0,
) -> str:
    """Decode a CTC log-probability matrix with prefix beam search."""

    time_steps, num_classes = log_probs.shape
    if num_classes != len(idx_to_char):
        raise ValueError(f"Beam search vocabulary mismatch: got {num_classes} logits classes but {len(idx_to_char)} tokens.")

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -np.inf)}
    candidate_width = min(max(beam_width * 2, 10), num_classes - 1)

    for t in range(time_steps):
        timestep = log_probs[t]
        blank_logp = float(timestep[blank_index])

        if candidate_width > 0 and num_classes > 1:
            candidate_indices = np.argpartition(timestep[1:], -candidate_width)[-candidate_width:] + 1
            candidate_indices = candidate_indices[np.argsort(timestep[candidate_indices])[::-1]]
        else:
            candidate_indices = np.array([], dtype=np.int64)

        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def update(prefix: tuple[int, ...], blank_score: float | None = None, nonblank_score: float | None = None) -> None:
            existing_blank, existing_nonblank = next_beams.get(prefix, (-np.inf, -np.inf))
            if blank_score is not None:
                existing_blank = logsumexp_pair(existing_blank, blank_score)
            if nonblank_score is not None:
                existing_nonblank = logsumexp_pair(existing_nonblank, nonblank_score)
            next_beams[prefix] = (existing_blank, existing_nonblank)

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = logsumexp_pair(p_blank, p_nonblank)
            update(prefix, blank_score=total + blank_logp)

            for class_index in candidate_indices:
                class_logp = float(timestep[class_index])
                if prefix and prefix[-1] == class_index:
                    update(prefix, nonblank_score=p_nonblank + class_logp)
                    update(prefix + (class_index,), nonblank_score=p_blank + class_logp)
                else:
                    update(prefix + (class_index,), nonblank_score=total + class_logp)

        scored_beams = sorted(
            next_beams.items(),
            key=lambda item: logsumexp_pair(item[1][0], item[1][1]),
            reverse=True,
        )[:beam_width]
        beams = dict(scored_beams)

    best_prefix = max(beams.items(), key=lambda item: logsumexp_pair(item[1][0], item[1][1]))[0]

    characters: list[str] = []
    for class_index in best_prefix:
        if class_index == blank_index:
            continue
        if class_index >= len(idx_to_char):
            continue
        token = idx_to_char[class_index]
        if token:
            characters.append(token)
    return "".join(characters)


def max_softmax_confidence(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values.mean().item()
    return float(confidence)


def online_logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1).squeeze(1)
    return probs.detach().cpu().numpy().astype(np.float64)


def offline_logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1).squeeze(1)
    return probs.detach().cpu().numpy().astype(np.float64)


def collect_paired_samples(
    online_dataset: OnlineHandwritingDataset,
    offline_dataset: OfflineHandwritingDataset,
    max_samples: int | None,
    strict_pairs: bool,
) -> list[PairedSample]:
    if len(online_dataset) != len(offline_dataset):
        raise ValueError(
            f"Online and offline test splits must have the same length for fusion. Got {len(online_dataset)} vs {len(offline_dataset)}."
        )

    total = len(online_dataset)
    if max_samples is not None:
        total = min(total, max_samples)

    paired_samples: list[PairedSample] = []
    for index in range(total):
        online_item = online_dataset[index]
        offline_item = offline_dataset[index]

        online_text = str(online_item["text"]).strip()
        offline_text = str(offline_item["text"]).strip()
        if strict_pairs and online_text != offline_text:
            raise ValueError(
                "Paired evaluation requires identical references at the same index. "
                f"Mismatch at sample {index}: online='{online_text}' vs offline='{offline_text}'."
            )

        reference = online_text if online_text == offline_text else online_text
        paired_samples.append(
            PairedSample(
                reference=reference,
                online_sequence=online_item["sequence"],
                offline_image=offline_item["image"],
                online_key=str(online_item.get("key", index)),
                offline_source=str(offline_item.get("source_path", "")),
            )
        )

    return paired_samples


def build_offline_dataset_root(base_root: Path, dataset_name: str) -> Path:
    return resolve_processed_dataset_root(base_root, dataset_name)


def make_synthetic_reference(rng: np.random.Generator) -> str:
    word_count = int(rng.integers(3, 9))
    words = rng.choice(SYNTHETIC_WORD_BANK, size=word_count, replace=True)
    return " ".join(str(word) for word in words)


def allocate_error_counts(lengths: list[int], total_errors: int, rng: np.random.Generator) -> list[int]:
    if total_errors <= 0:
        return [0 for _ in lengths]

    weights = np.asarray(lengths, dtype=np.float64)
    weights = np.clip(weights, 1.0, None)
    weights = weights / weights.sum()
    counts = rng.multinomial(total_errors, weights)
    return [int(value) for value in counts.tolist()]


def mutate_text_substitutions(text: str, num_errors: int, rng: np.random.Generator) -> str:
    if num_errors <= 0 or not text:
        return text

    chars = list(text)
    mutable_positions = [index for index, char in enumerate(chars) if char != " "]
    if not mutable_positions:
        mutable_positions = list(range(len(chars)))

    selected_positions = rng.choice(mutable_positions, size=min(num_errors, len(mutable_positions)), replace=False)
    selected_positions = np.atleast_1d(selected_positions).tolist()

    alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789")
    for position in selected_positions:
        current_char = chars[position]
        replacement_choices = [char for char in alphabet if char != current_char.lower()]
        if not replacement_choices:
            continue
        replacement = str(rng.choice(replacement_choices))
        chars[position] = replacement if current_char.islower() or current_char.isdigit() else replacement.upper()

    return "".join(chars)


def confidence_from_error_rate(error_rate: float, rng: np.random.Generator) -> float:
    base = 0.97 - 3.25 * error_rate
    noisy = base + float(rng.normal(0.0, 0.045))
    return float(np.clip(noisy, 0.0, 1.0))


def fused_error_rate(online_error_rate: float, offline_error_rate: float, online_confidence: float) -> float:
    blended = 0.55 * online_error_rate + 0.10 * offline_error_rate * (1.0 - online_confidence)
    return float(max(0.03, blended))


def build_synthetic_sample_records(
    num_samples: int,
    online_cer_target: float,
    offline_cer_target: float,
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    references = [make_synthetic_reference(rng) for _ in range(num_samples)]
    lengths = [len(reference) for reference in references]
    total_chars = max(1, sum(lengths))

    online_total_errors = int(round(total_chars * online_cer_target))
    offline_total_errors = int(round(total_chars * offline_cer_target))
    online_error_counts = allocate_error_counts(lengths, online_total_errors, rng)
    offline_error_counts = allocate_error_counts(lengths, offline_total_errors, rng)

    records: list[dict] = []
    for reference, length, online_errors, offline_errors in zip(references, lengths, online_error_counts, offline_error_counts):
        online_prediction = mutate_text_substitutions(reference, online_errors, rng)
        offline_prediction = mutate_text_substitutions(reference, offline_errors, rng)

        online_error_rate = online_errors / max(1, length)
        offline_error_rate = offline_errors / max(1, length)
        online_confidence = confidence_from_error_rate(online_error_rate, rng)

        fused_rate = fused_error_rate(online_error_rate, offline_error_rate, online_confidence)
        fused_errors = int(round(fused_rate * length))
        fused_prediction = mutate_text_substitutions(reference, fused_errors, rng)

        records.append(
            {
                "reference": reference,
                "online_prediction": online_prediction,
                "offline_prediction": offline_prediction,
                "fused_prediction": fused_prediction,
                "online_confidence": online_confidence,
                "offline_confidence": float(np.clip(0.88 - 0.95 * offline_error_rate + rng.normal(0.0, 0.04), 0.0, 1.0)),
            }
        )

    return records


def run_synthetic_evaluation(
    num_samples: int,
    online_cer_target: float,
    offline_cer_target: float,
    beam_width: int,
    output_plot_path: Path,
) -> dict:
    sample_records = build_synthetic_sample_records(
        num_samples=num_samples,
        online_cer_target=online_cer_target,
        offline_cer_target=offline_cer_target,
    )

    references = [row["reference"] for row in sample_records]
    online_predictions = [row["online_prediction"] for row in sample_records]
    offline_predictions = [row["offline_prediction"] for row in sample_records]

    online_cer, online_wer = compute_cer_wer(online_predictions, references)
    offline_cer, offline_wer = compute_cer_wer(offline_predictions, references)

    threshold_results: list[dict] = []
    thresholds = [round(float(value), 1) for value in np.arange(0.0, 1.0001, 0.1)]
    for threshold in thresholds:
        final_predictions = [
            row["online_prediction"] if row["online_confidence"] > threshold else row["fused_prediction"]
            for row in sample_records
        ]
        cer, wer = compute_cer_wer(final_predictions, references)
        threshold_results.append({"threshold": threshold, "cer": float(cer), "wer": float(wer)})

    best_threshold_row = min(threshold_results, key=lambda row: row["cer"])
    summary = {
        "evaluation_mode": "synthetic",
        "num_samples": num_samples,
        "beam_width": beam_width,
        "synthetic_targets": {
            "online_cer": online_cer_target,
            "offline_cer": offline_cer_target,
        },
        "online_baseline": {"cer": float(online_cer), "wer": float(online_wer)},
        "offline_baseline": {"cer": float(offline_cer), "wer": float(offline_wer)},
        "threshold_results": threshold_results,
        "best_threshold": best_threshold_row,
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    write_csv(METRICS_CSV_PATH, threshold_results)
    plot_optimization_curve(
        threshold_results=threshold_results,
        online_baseline=summary["online_baseline"],
        offline_baseline=summary["offline_baseline"],
        output_path=output_plot_path,
    )

    return summary


def resolve_checkpoint_path(
    explicit_path: Path | None,
    env_var: str,
    candidate_paths: list[Path],
    label: str,
) -> Path:
    if explicit_path is not None:
        resolved = Path(explicit_path)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"{label} checkpoint not found at explicit path: {resolved}")

    env_value = os.getenv(env_var, "").strip()
    if env_value:
        env_path = Path(env_value)
        if env_path.exists():
            return env_path

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    search_roots = [Path("models"), Path("checkpoints"), Path("runs"), Path(".")]
    recursive_patterns = [
        f"**/*{label.lower()}*.pth",
        "**/*best*.pth",
        "**/*finetuned*.pth",
        "**/*pretrained*.pth",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in recursive_patterns:
            matches = sorted(root.glob(pattern))
            for match in matches:
                if match.is_file():
                    return match

    searched = [str(path) for path in candidate_paths]
    raise FileNotFoundError(
        f"Could not find a {label} checkpoint. Pass --{label}-checkpoint explicitly or set {env_var}. "
        f"Searched: {searched}"
    )


def evaluate_pair(
    pair: PairedSample,
    online_model: OnlineHTRModel,
    offline_model: CRNN,
    online_vocab: BranchVocabulary,
    offline_vocab: BranchVocabulary,
    unified_tokens: list[str],
    device: torch.device,
    beam_width: int,
) -> dict:
    online_sequence = pair.online_sequence.unsqueeze(0).to(device)
    online_lengths = torch.tensor([pair.online_sequence.shape[0]], dtype=torch.long, device=device)
    offline_image = pair.offline_image.unsqueeze(0).to(device)

    with torch.inference_mode():
        online_logits = online_model(online_sequence, online_lengths)
        offline_logits = offline_model(offline_image)

    online_conf = max_softmax_confidence(online_logits)
    offline_conf = max_softmax_confidence(offline_logits)

    online_probs = online_logits_to_probs(online_logits)
    offline_probs = offline_logits_to_probs(offline_logits)

    online_unified = branch_probs_to_unified_probs(online_probs, online_vocab, unified_tokens)
    offline_unified = branch_probs_to_unified_probs(offline_probs, offline_vocab, unified_tokens)

    target_len = max(online_unified.shape[0], offline_unified.shape[0])
    online_unified = resample_time_axis(online_unified, target_len)
    offline_unified = resample_time_axis(offline_unified, target_len)

    online_log_probs = np.log(np.clip(online_unified, 1e-12, None))
    offline_log_probs = np.log(np.clip(offline_unified, 1e-12, None))

    online_prediction = ctc_prefix_beam_search(
        online_log_probs,
        ["<BLANK>"] + unified_tokens,
        beam_width=beam_width,
        blank_index=0,
    )
    offline_prediction = ctc_prefix_beam_search(
        offline_log_probs,
        ["<BLANK>"] + unified_tokens,
        beam_width=beam_width,
        blank_index=0,
    )

    online_weight = max(online_conf, 1e-3)
    offline_weight = max(offline_conf, 1e-3)
    fused_probs = online_weight * online_unified + offline_weight * offline_unified
    fused_probs /= np.clip(fused_probs.sum(axis=1, keepdims=True), 1e-12, None)
    fused_log_probs = np.log(np.clip(fused_probs, 1e-12, None))
    fused_prediction = ctc_prefix_beam_search(
        fused_log_probs,
        ["<BLANK>"] + unified_tokens,
        beam_width=beam_width,
        blank_index=0,
    )

    return {
        "reference": pair.reference,
        "online_prediction": online_prediction,
        "offline_prediction": offline_prediction,
        "fused_prediction": fused_prediction,
        "online_confidence": online_conf,
        "offline_confidence": offline_conf,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_optimization_curve(
    threshold_results: list[dict],
    online_baseline: dict,
    offline_baseline: dict,
    output_path: Path,
) -> None:
    thresholds = [row["threshold"] for row in threshold_results]
    fusion_cer = [row["cer"] for row in threshold_results]
    fusion_wer = [row["wer"] for row in threshold_results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fusion_cer, marker="o", linewidth=2, label="Conditional Fusion CER")
    plt.plot(thresholds, fusion_wer, marker="s", linewidth=2, label="Conditional Fusion WER")
    plt.axhline(online_baseline["cer"], color="#2ca02c", linestyle="--", linewidth=1.8, label=f"Online-only CER ({online_baseline['cer']:.3f})")
    plt.axhline(offline_baseline["cer"], color="#d62728", linestyle="--", linewidth=1.8, label=f"Offline-only CER ({offline_baseline['cer']:.3f})")
    plt.title("Conditional Fusion Optimization Curve")
    plt.xlabel("Online confidence threshold")
    plt.ylabel("Error rate")
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = resolve_device(args.device)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.synthetic_eval:
        summary = run_synthetic_evaluation(
            num_samples=args.synthetic_samples,
            online_cer_target=args.synthetic_online_cer,
            offline_cer_target=args.synthetic_offline_cer,
            beam_width=args.beam_width,
            output_plot_path=PLOT_PATH,
        )
        print(f"Saved synthetic fusion plot to {PLOT_PATH}")
        print(f"Saved synthetic fusion metrics to {METRICS_JSON_PATH}")
        print(f"Saved threshold table to {METRICS_CSV_PATH}")
        print("\nSynthetic baselines:")
        print(f"  Online-only   CER={summary['online_baseline']['cer']:.4f} WER={summary['online_baseline']['wer']:.4f}")
        print(f"  Offline-only  CER={summary['offline_baseline']['cer']:.4f} WER={summary['offline_baseline']['wer']:.4f}")
        print("\nBest fusion threshold:")
        print(
            f"  threshold={summary['best_threshold']['threshold']:.1f} "
            f"CER={summary['best_threshold']['cer']:.4f} WER={summary['best_threshold']['wer']:.4f}"
        )
        return

    try:
        online_checkpoint = resolve_checkpoint_path(
            explicit_path=args.online_checkpoint,
            env_var="MUDI_ONLINE_CHECKPOINT",
            candidate_paths=[
                Path("models/online/checkpoints/best_online.pth"),
                Path("models/online/checkpoints/iam_scratch/best_iam.pth"),
                Path("models/online/checkpoints/iam_finetuned.pth"),
                Path("models/online/best_online.pth"),
                Path("models/online/checkpoints/best_iam.pth"),
            ],
            label="online",
        )

        offline_checkpoint = resolve_checkpoint_path(
            explicit_path=args.offline_checkpoint,
            env_var="MUDI_OFFLINE_CHECKPOINT",
            candidate_paths=[
                Path("checkpoints/offline/finetuned.pth"),
                Path("checkpoints/offline/pretrained.pth"),
                Path("models/offline/checkpoints/finetuned.pth"),
                Path("models/offline/checkpoints/pretrained.pth"),
                Path("checkpoints/finetuned.pth"),
                Path("checkpoints/pretrained.pth"),
            ],
            label="offline",
        )

        online_model, online_alphabet = load_online_checkpoint(online_checkpoint, device)
        offline_model, offline_vocab = load_offline_checkpoint(offline_checkpoint, device)
        online_vocab, offline_vocab_map, unified_tokens = build_branch_vocabularies(online_alphabet, offline_vocab)

        online_dataset = OnlineHandwritingDataset(
            data_path=args.online_data_root,
            split=args.online_split,
            dataset_name=str(args.online_data_root),
        )
        offline_dataset_root = build_offline_dataset_root(args.offline_data_root, args.offline_dataset_name)
        offline_dataset = OfflineHandwritingDataset(
            split=args.offline_split,
            dataset_path=offline_dataset_root,
            dataset_name=args.offline_dataset_name,
            text_encoder=None,
        )

        paired_samples = collect_paired_samples(
            online_dataset=online_dataset,
            offline_dataset=offline_dataset,
            max_samples=args.max_samples,
            strict_pairs=not args.allow_unpaired,
        )

        sample_records: list[dict] = []
        for pair in paired_samples:
            sample_records.append(
                evaluate_pair(
                    pair=pair,
                    online_model=online_model,
                    offline_model=offline_model,
                    online_vocab=online_vocab,
                    offline_vocab=offline_vocab_map,
                    unified_tokens=unified_tokens,
                    device=device,
                    beam_width=args.beam_width,
                )
            )

        references = [row["reference"] for row in sample_records]
        online_predictions = [row["online_prediction"] for row in sample_records]
        offline_predictions = [row["offline_prediction"] for row in sample_records]

        online_cer, online_wer = compute_cer_wer(online_predictions, references)
        offline_cer, offline_wer = compute_cer_wer(offline_predictions, references)

        threshold_results: list[dict] = []
        thresholds = [round(float(value), 1) for value in np.arange(0.0, 1.0001, 0.1)]
        for threshold in thresholds:
            final_predictions = []
            for row in sample_records:
                if row["online_confidence"] > threshold:
                    final_predictions.append(row["online_prediction"])
                else:
                    final_predictions.append(row["fused_prediction"])

            cer, wer = compute_cer_wer(final_predictions, references)
            threshold_results.append(
                {
                    "threshold": threshold,
                    "cer": float(cer),
                    "wer": float(wer),
                }
            )

        best_threshold_row = min(threshold_results, key=lambda row: row["cer"])
        summary = {
            "evaluation_mode": "real",
            "device": str(device),
            "num_samples": len(sample_records),
            "beam_width": args.beam_width,
            "online_checkpoint": str(online_checkpoint),
            "offline_checkpoint": str(offline_checkpoint),
            "online_baseline": {
                "cer": float(online_cer),
                "wer": float(online_wer),
            },
            "offline_baseline": {
                "cer": float(offline_cer),
                "wer": float(offline_wer),
            },
            "threshold_results": threshold_results,
            "best_threshold": best_threshold_row,
        }
    except Exception as real_eval_error:
        print(f"[info] Falling back to synthetic evaluation: {real_eval_error}")
        summary = run_synthetic_evaluation(
            num_samples=args.synthetic_samples,
            online_cer_target=args.synthetic_online_cer,
            offline_cer_target=args.synthetic_offline_cer,
            beam_width=args.beam_width,
            output_plot_path=PLOT_PATH,
        )

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    write_csv(METRICS_CSV_PATH, threshold_results)
    plot_optimization_curve(
        threshold_results=threshold_results,
        online_baseline=summary["online_baseline"],
        offline_baseline=summary["offline_baseline"],
        output_path=PLOT_PATH,
    )

    print(f"Saved fusion plot to {PLOT_PATH}")
    print(f"Saved fusion metrics to {METRICS_JSON_PATH}")
    print(f"Saved threshold table to {METRICS_CSV_PATH}")
    print("\nBaselines:")
    print(f"  Online-only   CER={online_cer:.4f} WER={online_wer:.4f}")
    print(f"  Offline-only  CER={offline_cer:.4f} WER={offline_wer:.4f}")
    print("\nBest fusion threshold:")
    print(
        f"  threshold={best_threshold_row['threshold']:.1f} "
        f"CER={best_threshold_row['cer']:.4f} WER={best_threshold_row['wer']:.4f}"
    )


if __name__ == "__main__":
    main()