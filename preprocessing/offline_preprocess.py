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


def _distortion_free_resize(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize while preserving aspect ratio and pad to the target canvas."""
    target_h, target_w = image_size
    source_h, source_w = image.shape[:2]
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"Invalid image shape for resizing: {image.shape}")

    scale = min(target_w / source_w, target_h / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))

    if cv2 is not None:
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    else:
        resized = np.array(Image.fromarray(image).resize((resized_w, resized_h), Image.Resampling.BILINEAR))

    canvas = np.full((target_h, target_w), 255, dtype=resized.dtype)
    top = max(0, (target_h - resized_h) // 2)
    left = max(0, (target_w - resized_w) // 2)
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas


def preprocess_image_from_array(
    image: np.ndarray,
    image_size: tuple[int, int] = (128, 512),
    augment: bool = False,
    binarize: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Preprocess an in-memory grayscale image using the offline training pipeline."""
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image array, got shape {image.shape}")

    working = image.astype(np.uint8, copy=False)

    if augment:
        local_rng = rng if rng is not None else np.random.default_rng()
        working = _augment_grayscale(working, local_rng)

    working = _distortion_free_resize(working, image_size=image_size)

    if binarize:
        working = _otsu_binarize(working)

    return working.astype(np.float32) / 255.0


def preprocess_image_from_pil(
    image: Image.Image,
    image_size: tuple[int, int] = (128, 512),
    augment: bool = False,
    binarize: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Preprocess a PIL image using the offline training pipeline."""
    grayscale = image.convert("L")
    return preprocess_image_from_array(
        np.array(grayscale, dtype=np.uint8),
        image_size=image_size,
        augment=augment,
        binarize=binarize,
        rng=rng,
    )


def preprocess_image(
    path: str | Path,
    image_size: tuple[int, int] = (128, 512),
    augment: bool = False,
    binarize: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load and preprocess a handwriting image with distortion-free resizing."""
    source = str(path)
    if cv2 is not None:
        image = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to load image: {path}")
        return preprocess_image_from_array(
            image,
            image_size=image_size,
            augment=augment,
            binarize=binarize,
            rng=rng,
        )

    image = Image.open(source)
    return preprocess_image_from_pil(
        image,
        image_size=image_size,
        augment=augment,
        binarize=binarize,
        rng=rng,
    )

