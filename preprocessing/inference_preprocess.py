from __future__ import annotations

from io import BytesIO
from typing import List, Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from preprocessing.offline_preprocess import preprocess_image_from_pil, preprocess_image_from_array


def _apply_clahe_pil(pil_img: Image.Image) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return np.array(pil_img)
    arr = np.array(pil_img).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr)


def try_preprocess_variants(pil_img: Image.Image, image_size: Tuple[int, int] = (128, 512)) -> List[Dict]:
    """Return a list of preprocessing candidates with metadata and processed arrays.

    Each dict: {method, invert, proc_arr}
    """
    candidates = []
    methods = ["none", "otsu", "clahe"]
    for method in methods:
        for invert in (False, True):
            img = pil_img.copy()
            if invert:
                img = ImageOps.invert(img)

            if method == "clahe":
                try:
                    arr = _apply_clahe_pil(img)
                    proc_arr = preprocess_image_from_array(arr, image_size=image_size, augment=False, binarize=False)
                except Exception:
                    proc_arr = preprocess_image_from_pil(img, image_size=image_size, augment=False, binarize=True)
            elif method == "otsu":
                proc_arr = preprocess_image_from_pil(img, image_size=image_size, augment=False, binarize=True)
            else:
                proc_arr = preprocess_image_from_pil(img, image_size=image_size, augment=False, binarize=False)

            candidates.append({"method": method, "invert": invert, "proc_arr": proc_arr})

    return candidates


def select_best_candidate(
    offline_model: torch.nn.Module,
    offline_alphabet: list[str],
    candidates: List[Dict],
    device: torch.device,
    beam_width: int = 15,
) -> Dict:
    """Run offline model on each candidate and return the best with its decoded text and confidence."""
    best = None
    best_conf = -float("inf")
    results = []
    EMPTY_TEXT_PENALTY = 0.6  # penalize empty decoded strings to prefer non-empty readable outputs
    for c in candidates:
        proc_arr = c["proc_arr"]
        tensor = torch.tensor(proc_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = offline_model(tensor)
            decoded = offline_model.decode_beam_search(logits, alphabet=offline_alphabet, beam_width=beam_width)
            text, conf = decoded[0] if decoded else ("", -float("inf"))
        # Penalize empty texts so that an inverted blank image with high confidence
        # doesn't override a readable non-empty candidate.
        score = conf
        if not text:
            score = conf - EMPTY_TEXT_PENALTY
        # Prefer candidates that include spaces (likely separate words)
        SPACE_BONUS = 0.3
        if isinstance(text, str) and ' ' in text and len(text.strip()) > 0:
            score = score + SPACE_BONUS
        results.append({"method": c["method"], "invert": c["invert"], "text": text, "conf": conf, "score": score})
        if score > best_conf or (score == best_conf and len(text) > len(best.get('text','')) if best is not None else False):
            best_conf = score
            best = results[-1]

    if best is None:
        return {"method": None, "invert": False, "text": "", "conf": -float("inf"), "results": results}
    best["results"] = results
    return best
