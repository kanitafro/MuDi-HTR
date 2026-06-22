"""Offline handwriting image preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _augment_grayscale(
    image: np.ndarray,
    rng: np.random.Generator,
    max_rotation_deg: float = 5.0,
    max_scale_delta: float = 0.10,
) -> np.ndarray:
    """Apply light geometric augmentation for training robustness."""
    if cv2 is None:
        return image

    h, w = image.shape
    angle = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
    scale = float(rng.uniform(1.0 - max_scale_delta, 1.0 + max_scale_delta))
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    augmented = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return augmented


def preprocess_image(
    path: str | Path,
    image_size: tuple[int, int] = (128, 512),
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load, grayscale, resize, optionally augment, and normalize a handwriting image."""
    source = str(path)
    width, height = image_size[1], image_size[0]

    if cv2 is not None:
        image = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to load image: {path}")
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    else:
        image = Image.open(source).convert("L")
        resized = np.array(image.resize((width, height)))

    if augment:
        local_rng = rng if rng is not None else np.random.default_rng()
        resized = _augment_grayscale(resized, local_rng)

    return resized.astype(np.float32) / 255.0

