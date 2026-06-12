"""Online handwriting preprocessing utilities."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

Point = tuple[float, float]
Stroke = list[Point]


def _normalize_strokes(strokes: list[Stroke]) -> list[Stroke]:
    """Normalize points in all strokes to [0, 1] range."""
    if not strokes:
        return []

    xs = [x for stroke in strokes for x, _ in stroke]
    ys = [y for stroke in strokes for _, y in stroke]
    if not xs or not ys:
        return []

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max(max_x - min_x, 1e-9)
    range_y = max(max_y - min_y, 1e-9)

    normalized: list[Stroke] = []
    for stroke in strokes:
        normalized.append([
            ((x - min_x) / range_x, (y - min_y) / range_y) for x, y in stroke
        ])
    return normalized


def _parse_json_strokes(payload: dict) -> list[Stroke]:
    """Parse strokes from a JSON payload."""
    strokes_data = payload.get("strokes", payload)
    parsed: list[Stroke] = []
    for stroke in strokes_data:
        points: Iterable = stroke.get("points", stroke) if isinstance(stroke, dict) else stroke
        parsed_points: Stroke = []
        for point in points:
            if isinstance(point, dict):
                x, y = point.get("x"), point.get("y")
            else:
                x, y = point[0], point[1]
            if x is None or y is None:
                continue
            parsed_points.append((float(x), float(y)))
        if parsed_points:
            parsed.append(parsed_points)
    return parsed


def _parse_inkml_strokes(root: ET.Element) -> list[Stroke]:
    """Parse trace points from an InkML root element."""
    parsed: list[Stroke] = []
    for trace in root.findall(".//{*}trace"):
        text = (trace.text or "").strip()
        if not text:
            continue
        points: Stroke = []
        for raw_point in text.split(","):
            coords = [c for c in raw_point.strip().split() if c]
            if len(coords) < 2:
                continue
            try:
                x, y = float(coords[0]), float(coords[1])
            except ValueError:
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        if points:
            parsed.append(points)
    return parsed


def parse_didi(path: str | Path) -> list[Stroke]:
    """Load DIDI JSON/InkML strokes and return normalized stroke sequences."""
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        strokes = _parse_json_strokes(payload)
    elif suffix in {".inkml", ".xml"}:
        root = ET.parse(source).getroot()
        strokes = _parse_inkml_strokes(root)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return _normalize_strokes(strokes)
