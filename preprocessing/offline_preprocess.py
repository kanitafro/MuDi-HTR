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
    max_shear_deg: float = 7.0,
    max_translation_frac: float = 0.06,
) -> np.ndarray:
    """Apply light geometric augmentation for training robustness."""
    if cv2 is None:
        return image

    h, w = image.shape
    angle = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
    scale_x = float(rng.uniform(1.0 - max_scale_delta, 1.0 + max_scale_delta))
    scale_y = float(rng.uniform(1.0 - max_scale_delta, 1.0 + max_scale_delta))
    shear_x = np.tan(np.deg2rad(float(rng.uniform(-max_shear_deg, max_shear_deg))))
    translate_x = float(rng.uniform(-max_translation_frac, max_translation_frac) * w)
    translate_y = float(rng.uniform(-max_translation_frac, max_translation_frac) * h)

    center_x = w / 2.0
    center_y = h / 2.0
    theta = np.deg2rad(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    to_origin = np.array(
        [
            [1.0, 0.0, -center_x],
            [0.0, 1.0, -center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scale_matrix = np.array(
        [
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    shear_matrix = np.array(
        [
            [1.0, shear_x, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rotation_matrix = np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    back_to_center = np.array(
        [
            [1.0, 0.0, center_x + translate_x],
            [0.0, 1.0, center_y + translate_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    affine = back_to_center @ rotation_matrix @ shear_matrix @ scale_matrix @ to_origin
    matrix = affine[:2, :]

    augmented = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return augmented


def _otsu_binarize(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale handwriting image to a binary foreground/background mask."""
    if cv2 is not None:
        _, binary = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return binary

    threshold = float(image.mean())
    return np.where(image > threshold, 255, 0).astype(np.uint8)


def preprocess_image(
    path: str | Path,
    image_size: tuple[int, int] = (128, 512),
    augment: bool = False,
    binarize: bool = True,
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

    if binarize:
        resized = _otsu_binarize(resized)

    return resized.astype(np.float32) / 255.0

