# preprocessing/isgl_preprocess.py
"""
Preprocessing for ISGL Online Handwriting Dataset.
Assumes .txt files are already extracted in the original folder structure.
"""

import re
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

from preprocessing.utils import process_strokes, save_processed_data


def find_all_txt_files(root_dir: Path) -> List[Path]:
    """Find all .txt files in the ISGL directory tree."""
    txt_files = []
    # Look in the ONLINE folder
    online_dir = root_dir / "ONLINE"
    if online_dir.exists():
        txt_files.extend(list(online_dir.rglob("*.txt")))
    # Also look in extracted folder if it exists (from manual extraction)
    extracted_dir = root_dir / "extracted"
    if extracted_dir.exists():
        txt_files.extend(list(extracted_dir.rglob("*.txt")))
    # Also look in the root itself
    txt_files.extend(list(root_dir.rglob("*.txt")))
    return list(set(txt_files))


def parse_isgl_txt(file_path: Path) -> Dict[str, Any]:
    """Parse a single ISGL .txt file containing stroke data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    lines = content.strip().split('\n')
    text = lines[0].strip() if lines else ""

    strokes = []
    current_stroke = []
    in_stroke = False

    for line in lines:
        line = line.strip()
        if line == "Pen Down (x,y)":
            in_stroke = True
            current_stroke = []
            continue
        if line == "Pen Up":
            if current_stroke:
                strokes.append(current_stroke)
                current_stroke = []
            in_stroke = False
            continue
        if in_stroke and line and not line.startswith("TimeTaken"):
            parts = line.split('_')
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    current_stroke.append([x, y, 0])
                except ValueError:
                    pass

    # Fallback parsing
    if not strokes:
        pattern = r'Pen Down \(x,y\)\n([\s\S]*?)\nPen Up'
        matches = re.findall(pattern, content)
        for match in matches:
            stroke = []
            for line in match.strip().split('\n'):
                if '_' in line:
                    parts = line.split('_')
                    if len(parts) == 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            stroke.append([x, y, 0])
                        except ValueError:
                            pass
            if stroke:
                strokes.append(stroke)

    if not strokes:
        return None

    # Estimate writing guide
    all_x = [p[0] for stroke in strokes for p in stroke]
    all_y = [p[1] for stroke in strokes for p in stroke]
    width = max(all_x) - min(all_x) + 100 if all_x else 1000
    height = max(all_y) - min(all_y) + 100 if all_y else 1000
    writing_guide = {'width': width, 'height': height}

    return {
        'text': text,
        'drawing': strokes,
        'writing_guide': writing_guide,
        'key': file_path.stem
    }


def preprocess_isgl_dataset(input_dir: Path, output_dir: Path):
    """Preprocess ISGL dataset by scanning for .txt files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = find_all_txt_files(input_dir)
    print(f"Found {len(txt_files)} .txt files total.")

    if not txt_files:
        print("❌ No .txt files found. Please extract the RAR files manually.")
        print("   Use WinRAR or 7-Zip to extract all .rar files in the ISGL folder.")
        return

    print("Sample files:")
    for f in txt_files[:5]:
        print(f"  {f}")

    all_samples = []
    for txt_file in tqdm(txt_files, desc="Processing samples"):
        try:
            sample = parse_isgl_txt(txt_file)
            if sample and sample['text'] and sample['drawing']:
                processed_strokes = process_strokes(sample['drawing'], sample['writing_guide'])
                all_samples.append({
                    'key': sample['key'],
                    'strokes': processed_strokes,
                    'text': sample['text'],
                    'dataset': 'isgl'
                })
        except Exception as e:
            print(f"Error parsing {txt_file}: {e}")

    if not all_samples:
        print("❌ No valid samples found!")
        return

    print(f"✅ Total valid samples: {len(all_samples)}")

    # Split into train/valid/test (80/10/10)
    import random
    random.seed(42)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    splits = {
        'train': all_samples[:n_train],
        'valid': all_samples[n_train:n_train + n_val],
        'test': all_samples[n_train + n_val:]
    }

    for split_name, samples in splits.items():
        if samples:
            save_processed_data(samples, output_dir, split_name)

    # Statistics
    print(f"\n📊 ISGL Dataset Statistics:")
    for split_name in ['train', 'valid', 'test']:
        file_path = output_dir / f"{split_name}.pt"
        if file_path.exists():
            data = torch.load(file_path, weights_only=False)
            texts = [s['text'] for s in data]
            unique_texts = len(set(texts))
            text_lengths = [len(t) for t in texts]
            avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            print(f"  {split_name}: {len(data)} samples, {unique_texts} unique texts, avg len: {avg_len:.1f}")


def main():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    input_dir = repo_root / "data" / "raw" / "ISGL"
    output_dir = repo_root / "data" / "processed" / "online" / "isgl"

    print(f"Looking for ISGL data at: {input_dir}")
    print(f"Output directory: {output_dir}")

    if not input_dir.exists():
        print(f"❌ ERROR: ISGL directory not found at {input_dir}")
        return

    preprocess_isgl_dataset(input_dir, output_dir)
    print("\n✅ ISGL preprocessing complete!")


if __name__ == "__main__":
    main()