# preprocessing/utils.py
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

# ---------------------------------------------------------
# Extract rare files if needed (for ISGL dataset)
# ---------------------------------------------------------

import os
from pathlib import Path

# Try to import patool for universal archive extraction
try:
    import patoolib
    HAS_PATOOL = True
except ImportError:
    HAS_PATOOL = False
    print("patool not found. Installing...")
    os.system("pip install patool")
    try:
        import patoolib
        HAS_PATOOL = True
    except ImportError:
        print("Failed to install patool. Trying rarfile...")

# Fallback: try rarfile
if not HAS_PATOOL:
    try:
        import rarfile
        HAS_RARFILE = True
    except ImportError:
        HAS_RARFILE = False
        print("rarfile not found. Run: pip install rarfile")


def extract_archives(root_dir: Path):
    """
    Extract all .rar and .zip files in the directory tree.
    """
    extraction_base = root_dir / "extracted"
    extraction_base.mkdir(parents=True, exist_ok=True)
    
    # Find all archives
    archives = list(root_dir.rglob("*.rar")) + list(root_dir.rglob("*.zip"))
    
    if not archives:
        print("No archives found to extract.")
        return extraction_base
    
    print(f"Found {len(archives)} archives to extract.")
    
    for archive_path in archives:
        # Create output folder name
        rel_path = archive_path.relative_to(root_dir)
        output_name = rel_path.with_suffix("").stem
        output_dir = extraction_base / rel_path.parent / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if already extracted (check if there are files)
        if any(output_dir.iterdir()):
            print(f"⏭️  Already extracted: {archive_path.name}")
            continue
        
        print(f"Extracting: {archive_path.name} -> {output_dir}")
        
        if HAS_PATOOL:
            try:
                patoolib.extract_archive(str(archive_path), outdir=str(output_dir))
                print(f"✅ Extracted: {archive_path.name}")
            except Exception as e:
                print(f"❌ Error extracting {archive_path}: {e}")
        elif HAS_RARFILE:
            try:
                with rarfile.RarFile(archive_path) as rf:
                    rf.extractall(output_dir)
                print(f"✅ Extracted: {archive_path.name}")
            except Exception as e:
                print(f"❌ Error extracting {archive_path}: {e}")
        else:
            print("❌ No archive extraction library available.")
            print("   Please install: pip install patool")
            print("   Or install rarfile: pip install rarfile")
            return extraction_base
    
    return extraction_base


def extract_main():
    isgl_dir = Path("data/raw/ISGL")
    
    if not isgl_dir.exists():
        print(f"❌ ISGL directory not found: {isgl_dir}")
        return
    
    print("📦 Extracting ISGL archives...")
    extracted_dir = extract_archives(isgl_dir)
    
    # Show what was extracted
    print("\n📁 Extracted files:")
    for f in extracted_dir.rglob("*"):
        if f.is_file():
            print(f"  {f.relative_to(extracted_dir)}")
