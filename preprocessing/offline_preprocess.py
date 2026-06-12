"""Offline handwriting image preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def preprocess_image(path: str | Path, image_size: tuple[int, int] = (128, 512)) -> np.ndarray:
    """Load, grayscale, resize, and normalize a handwriting image."""
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

    return resized.astype(np.float32) / 255.0
