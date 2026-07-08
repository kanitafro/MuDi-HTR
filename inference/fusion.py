from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch

from models.online.model import CTCDecoder


def _branch_confidence(hypotheses: List[Tuple[str, float]]) -> Tuple[float, float]:
    if not hypotheses:
        return 0.0, -float("inf")

    scores = np.array([score for _, score in hypotheses], dtype=np.float64)
    top_score = float(scores[0])
    if len(scores) == 1:
        return 1.0, top_score

    shifted = scores - np.max(scores)
    probs = np.exp(shifted)
    probs = probs / max(np.sum(probs), 1e-12)
    top_probability = float(probs[0])
    margin = float(scores[0] - scores[1])
    margin_score = 1.0 / (1.0 + math.exp(-np.clip(margin, -8.0, 8.0)))
    confidence = 0.7 * top_probability + 0.3 * margin_score
    return float(np.clip(confidence, 0.0, 1.0)), top_score


def fuse_branch_hypotheses(
    online_beams: List[Tuple[str, float]],
    offline_beams: List[Tuple[str, float]],
    alpha_weight: float = 0.55,
    margin: float = 0.15,
    tie_bonus: float = 0.25,
) -> List[Tuple[str, float]]:
    """Confidence-gated weighted-sum fusion of two hypothesis lists.

    Each hypothesis list is a list of (text, score) tuples sorted by score desc.
    """
    if not offline_beams:
        return online_beams
    if not online_beams:
        return offline_beams

    online_conf, online_top = _branch_confidence(online_beams)
    offline_conf, offline_top = _branch_confidence(offline_beams)

    # Confidence-gated shortcut
    if offline_conf >= online_conf + margin:
        return offline_beams
    if online_conf >= offline_conf + margin:
        return online_beams

    online_w = alpha_weight * online_conf
    offline_w = (1.0 - alpha_weight) * offline_conf
    total_w = online_w + offline_w
    if total_w <= 1e-8:
        online_w = alpha_weight
        offline_w = 1.0 - alpha_weight
        total_w = 1.0
    online_w /= total_w
    offline_w /= total_w

    fused_scores: dict[str, float] = {}
    for text, score in online_beams:
        fused_scores[text] = fused_scores.get(text, 0.0) + online_w * float(score)
    for text, score in offline_beams:
        fused_scores[text] = fused_scores.get(text, 0.0) + offline_w * float(score)

    # Bonus if both agree on top
    if online_beams and offline_beams and online_beams[0][0] == offline_beams[0][0]:
        fused_scores[online_beams[0][0]] = fused_scores.get(online_beams[0][0], 0.0) + tie_bonus

    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)


def fast_ctc_beam_search(log_probs_2d, alphabet, beam_width=15, blank_idx=0, top_k_chars=10):
    """Simplified fast prefix beam search operating on a (T, C) log-prob matrix.

    Returns list of (text, score) candidates sorted desc.
    """
    if torch.is_tensor(log_probs_2d):
        log_probs_2d = log_probs_2d.detach().cpu().numpy()

    T, C = log_probs_2d.shape
    beams = {(): (0.0, -float("inf"))}

    for t in range(T):
        timestep = log_probs_2d[t]
        next_beams = {}
        top_indices = np.argsort(timestep)[::-1][:top_k_chars]

        def update(prefix, p_b=None, p_nb=None):
            exist_b, exist_nb = next_beams.get(prefix, (-float("inf"), -float("inf")))
            if p_b is not None:
                m = max(exist_b, p_b)
                exist_b = m + math.log(math.exp(exist_b - m) + math.exp(p_b - m)) if m > -float("inf") else -float("inf")
            if p_nb is not None:
                m = max(exist_nb, p_nb)
                exist_nb = m + math.log(math.exp(exist_nb - m) + math.exp(p_nb - m)) if m > -float("inf") else -float("inf")
            next_beams[prefix] = (exist_b, exist_nb)

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = max(p_blank, p_nonblank)
            if total > -float("inf"):
                total = total + math.log(math.exp(p_blank - total) + math.exp(p_nonblank - total))
            update(prefix, p_b=total + float(timestep[blank_idx]))

            for c in top_indices:
                if c == blank_idx:
                    continue
                score = float(timestep[c])
                if prefix and prefix[-1] == c:
                    update(prefix, p_nb=p_nonblank + score)
                    update(prefix + (c,), p_nb=p_blank + score)
                else:
                    update(prefix + (c,), p_nb=total + score)

        # prune
        sorted_beams = sorted(next_beams.items(), key=lambda item: max(item[1][0], item[1][1]), reverse=True)
        beams = dict(sorted_beams[:beam_width])

    candidates = []
    for prefix, (p_b, p_nb) in beams.items():
        m = max(p_b, p_nb)
        total_score = m + math.log(math.exp(p_b - m) + math.exp(p_nb - m)) if m > -float("inf") else -float("inf")
        norm_score = total_score / max(1, len(prefix))
        text = ''.join([alphabet[idx] for idx in prefix if idx < len(alphabet)])
        candidates.append((text, norm_score))

    return sorted(candidates, key=lambda x: x[1], reverse=True)


def perform_beam_fusion(
    online_logits: torch.Tensor,
    offline_logits: torch.Tensor,
    online_alphabet: list[str],
    offline_alphabet: list[str],
    alpha_weight: float = 0.55,
    beam_width: int = 15,
):
    """Decode both branches and fuse their beam lists."""
    online_log_probs = torch.log_softmax(online_logits, dim=-1)
    offline_log_probs = torch.log_softmax(offline_logits, dim=-1)

    on_log = online_log_probs.squeeze(1) if online_log_probs.ndim == 3 else online_log_probs
    off_log = offline_log_probs.squeeze(1) if offline_log_probs.ndim == 3 else offline_log_probs

    decoder_online = CTCDecoder(online_alphabet, blank_idx=0)
    online_beams = decoder_online.beam_search(on_log.unsqueeze(1), beam_width=beam_width, top_k=beam_width)[0]

    decoder_offline = CTCDecoder(offline_alphabet, blank_idx=0)
    offline_beams = decoder_offline.beam_search(off_log.unsqueeze(1), beam_width=beam_width, top_k=beam_width)[0]

    # handle pathological online outputs (e.g., 'wwwww') by ignoring that branch
    if online_beams:
        top_online_text = online_beams[0][0]
        # compute simple run-length ratio (largest same-character run / length)
        def max_run_ratio(s: str) -> float:
            if not s:
                return 0.0
            max_run = 1
            cur = 1
            for i in range(1, len(s)):
                if s[i] == s[i-1]:
                    cur += 1
                    if cur > max_run:
                        max_run = cur
                else:
                    cur = 1
            return max_run / max(1, len(s))

        run_ratio = max_run_ratio(top_online_text)
        # Use branch confidence to decide; if online is low conf or dominated by repeated runs, ignore it
        online_confidence, _ = _branch_confidence(online_beams)
        if (top_online_text and len(top_online_text) >= 3 and (len(set(top_online_text)) == 1 or run_ratio >= 0.6)) or online_confidence < 0.12:
            online_beams = []

    fused = fuse_branch_hypotheses(online_beams, offline_beams, alpha_weight=alpha_weight)
    return fused, online_beams[:3], offline_beams[:3]
