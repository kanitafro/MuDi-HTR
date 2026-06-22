"""Shared preprocessing utilities for online handwriting datasets."""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch

def normalize_stroke_points(xs: List[float], ys: List[float], 
                            guide_width: float, guide_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize coordinates using the writing guide dimensions."""
    norm_xs = (np.array(xs) / guide_width) * 2 - 1
    norm_ys = (np.array(ys) / guide_height) * 2 - 1
    return norm_xs, norm_ys

def normalize_timestamps(ts: List[float]) -> np.ndarray:
    """Normalize timestamps to [0, 1] range."""
    ts_array = np.array(ts, dtype=np.float32)
    if ts_array.max() > ts_array.min():
        return (ts_array - ts_array.min()) / (ts_array.max() - ts_array.min())
    return np.zeros_like(ts_array)

def process_strokes(drawing: List, writing_guide: Dict) -> List[np.ndarray]:
    """
    Process raw strokes into normalized stroke sequences.
    
    Args:
        drawing: List of strokes, each stroke is [xs, ys, ts]
        writing_guide: Dict with 'width' and 'height'
    
    Returns:
        List of numpy arrays, each shape (L_s, 3) with (norm_x, norm_y, norm_t)
    """
    width = writing_guide['width']
    height = writing_guide['height']
    
    processed_strokes = []
    for stroke in drawing:
        xs, ys, ts = stroke[0], stroke[1], stroke[2]
        norm_xs, norm_ys = normalize_stroke_points(xs, ys, width, height)
        norm_ts = normalize_timestamps(ts)
        stroke_data = np.stack([norm_xs, norm_ys, norm_ts], axis=1)
        processed_strokes.append(stroke_data)
    
    return processed_strokes

def save_processed_data(data: List[Dict], output_dir: Path, split: str):
    """Save processed data to a .pt file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{split}.pt"
    torch.save(data, out_file)
    print(f"Saved {len(data)} entries to {out_file}")
    return out_file

def get_text_length(text: str) -> int:
    """Get the length of a text string for CTC."""
    return len(text)