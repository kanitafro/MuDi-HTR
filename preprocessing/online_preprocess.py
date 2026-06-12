"""Preprocessing for DIDI dataset (online handwriting)."""
import os
import json
import numpy as np
from pathlib import Path
from torch import save as torch_save
from typing import List, Tuple, Dict, Any

def load_didi_ndjson(file_path: Path) -> List[Dict[str, Any]]:
    """Load all entries from a DIDI .ndjson file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def normalize_stroke_points(xs: List[float], ys: List[float], 
                            guide_width: float, guide_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize coordinates using the writing guide dimensions.
    Maps [0, width] -> [-1, 1] and [0, height] -> [-1, 1].

    Parameters:
    - xs: List of x coordinates for a stroke
    - ys: List of y coordinates for a stroke
    - guide_width: Width of the writing guide
    - guide_height: Height of the writing guide
    Returns:
    - norm_xs: Normalized x coordinates
    - norm_ys: Normalized y coordinates
    """
    norm_xs = (np.array(xs) / guide_width) * 2 - 1
    norm_ys = (np.array(ys) / guide_height) * 2 - 1
    return norm_xs, norm_ys

def preprocess_didi_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one DIDI entry (JSON line) into a sequence of normalized strokes.
    Parameters:
    - entry: A dict representing one 1 from the DIDI .ndjson file, containing:
        - 'key': unique identifier for the sample
        - 'split': which dataset split it belongs to (train/valid/test)
        - 'drawing': list of strokes, where each stroke is [xs, ys, ts]
        - 'writing_guide': dict with 'width' and 'height' for normalization

    
    Returns: {
        'key': str,
        'split': str,
        'strokes': List of (norm_x, norm_y, t, pressure_if_any),
        'label': str (optional, if text present)
    }
    """
    drawing = entry['drawing']  # list of strokes, each is [xs, ys, ts]
    guide = entry['writing_guide']
    width = guide['width']
    height = guide['height']
    
    processed_strokes = []
    for stroke in drawing:
        xs, ys, ts = stroke[0], stroke[1], stroke[2]
        # Normalize coordinates
        norm_xs, norm_ys = normalize_stroke_points(xs, ys, width, height)
        # Timestamps: normalize to [0,1] per stroke (optional)
        ts_array = np.array(ts, dtype=np.float32)
        if ts_array.max() > ts_array.min():
            ts_norm = (ts_array - ts_array.min()) / (ts_array.max() - ts_array.min())
        else:
            ts_norm = np.zeros_like(ts_array)
        # Combine into (x, y, t) per point
        stroke_data = np.stack([norm_xs, norm_ys, ts_norm], axis=1)
        processed_strokes.append(stroke_data)
    
    return {
        'key': entry['key'],
        'split': entry['split'],
        'strokes': processed_strokes,
        'label_id': entry.get('label_id'),
        # If text label exists (in diagrams with text), add it
        'text': entry.get('text', '')
    }

def save_processed_dataset(input_ndjson: Path, output_dir: Path):
    """Load DIDI ndjson, preprocess, and save by split."""
    data = load_didi_ndjson(input_ndjson)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        split_data = [preprocess_didi_entry(e) for e in data if e.get('split') == split]
        out_file = output_dir / f"{split}.pt"
        torch_save(split_data, out_file)  # or use np.savez
        print(f"Saved {len(split_data)} entries to {out_file}")

def main():
    
    print("Current working directory:", os.getcwd())
    print("Looking for file at:", Path("../data/raw/didi_dataset/diagrams_20200131.ndjson").absolute())

    data_path = Path("../data/raw/didi_dataset/diagrams_20200131.ndjson")
    
    # Check if file exists
    if not data_path.exists():
        print(f"ERROR: File not found at {data_path.absolute()}")
        return
    
    # Load first few entries to inspect
    with open(data_path, 'r') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        print("Keys in first entry:", sample.keys())
        print("Split value:", sample.get('split'))
        print("Writing guide:", sample.get('writing_guide'))
    
    # Then run your save function
    output_path = Path("../data/processed/online")
    save_processed_dataset(data_path, output_path)

if __name__ == "__main__":
    main()