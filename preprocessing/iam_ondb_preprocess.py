# preprocessing/iam_ondb_preprocess.py
"""IAM-OnDB dataset preprocessing for online handwriting recognition."""

import os
import re
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
from tqdm import tqdm
import random

from preprocessing.utils import process_strokes, save_processed_data


def parse_line_stroke_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a line stroke file from IAM-OnDB."""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        if content.startswith('<?xml') or content.startswith('<'):
            return parse_line_stroke_xml(content, file_path)
        else:
            return parse_line_stroke_text(content, file_path)
            
    except Exception as e:
        return None


def parse_line_stroke_xml(content: str, file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse line stroke XML file (WhiteboardCaptureSession format)."""
    try:
        root = ET.fromstring(content)
        
        strokes = []
        for stroke_elem in root.findall('.//Stroke'):
            xs = []
            ys = []
            ts = []
            for point_elem in stroke_elem.findall('.//Point'):
                x = point_elem.get('x')
                y = point_elem.get('y')
                if x is not None and y is not None:
                    try:
                        xs.append(float(x))
                        ys.append(float(y))
                        ts.append(0.0)
                    except ValueError:
                        pass
            if xs:
                strokes.append([xs, ys, ts])
        
        if not strokes:
            # Fallback: try Trace elements
            for trace_elem in root.findall('.//Trace'):
                if trace_elem.text:
                    xs, ys, ts = [], [], []
                    for point_str in trace_elem.text.strip().split(','):
                        if point_str.strip():
                            coords = [float(x) for x in point_str.strip().split() if x]
                            if len(coords) >= 2:
                                xs.append(coords[0])
                                ys.append(coords[1])
                                ts.append(coords[2] if len(coords) > 2 else 0.0)
                    if xs:
                        strokes.append([xs, ys, ts])
            if not strokes:
                return None
        
        # Get writing guide (canvas dimensions)
        writing_guide = {'width': 1000, 'height': 1000}
        desc = root.find('.//WhiteboardDescription')
        if desc is not None:
            coords = desc.find('.//DiagonallyOppositeCoords')
            if coords is not None:
                x = coords.get('x')
                y = coords.get('y')
                if x and y:
                    writing_guide = {'width': float(x), 'height': float(y)}
        
        return {
            'drawing': strokes,
            'writing_guide': writing_guide,
            'key': file_path.stem
        }
        
    except ET.ParseError:
        return None


def parse_line_stroke_text(content: str, file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse line stroke files in text format."""
    strokes = []
    current_stroke = []
    
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            if current_stroke:
                xs, ys, ts = [], [], []
                for p in current_stroke:
                    xs.append(p[0])
                    ys.append(p[1])
                    ts.append(p[2] if len(p) > 2 else 0.0)
                strokes.append([xs, ys, ts])
                current_stroke = []
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            if len(parts) > 2:
                for i in range(0, len(parts) - 1, 2):
                    try:
                        x = float(parts[i])
                        y = float(parts[i+1])
                        current_stroke.append([x, y, 0])
                    except ValueError:
                        pass
            else:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    current_stroke.append([x, y, 0])
                except ValueError:
                    pass
    
    if current_stroke:
        xs, ys, ts = [], [], []
        for p in current_stroke:
            xs.append(p[0])
            ys.append(p[1])
            ts.append(p[2] if len(p) > 2 else 0.0)
        strokes.append([xs, ys, ts])
    
    if not strokes:
        return None
    
    writing_guide = {'width': 1000, 'height': 1000}
    
    return {
        'drawing': strokes,
        'writing_guide': writing_guide,
        'key': file_path.stem
    }


def get_transcription(ascii_dir: Path, form_id: str) -> str:
    """
    Get transcription for a form from the ascii directory.
    """
    # Remove the trailing -XX suffix if present
    base_form_id = form_id
    if '-' in form_id:
        parts = form_id.split('-')
        if len(parts) >= 2 and parts[-1].isdigit():
            base_form_id = '-'.join(parts[:-1])
    
    parts = base_form_id.split('-')
    if len(parts) < 2:
        return ""
    
    writer = parts[0]
    form_part = '-'.join(parts[1:])
    
    if form_part and form_part[-1].isalpha():
        folder_name = f"{writer}-{form_part[:-1]}"
    else:
        folder_name = f"{writer}-{form_part}"
    
    ascii_file = ascii_dir / writer / folder_name / f"{base_form_id}.txt"
    
    if ascii_file.exists():
        try:
            with open(ascii_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                content = content.replace('\n', ' ').replace('\r', ' ')
                content = re.sub(r'\s+', ' ', content)
                content = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\'\"]', '', content)
                content = content.strip()
                return content
        except Exception:
            return ""
    
    return ""


def get_writer_from_path(file_path: Path) -> Optional[str]:
    """Extract writer ID from file path (folder structure)."""
    # The writer is the parent directory name (e.g., 'a01')
    parent_dir = file_path.parent.name
    if parent_dir.startswith(('a','b','c','d','e','f','g','h','j','k','l','m','n','p','r','z')):
        if len(parent_dir) <= 4:
            try:
                int(parent_dir[1:])
                return parent_dir
            except ValueError:
                pass
    
    # Fallback: try to extract from filename
    filename = file_path.stem
    if len(filename) >= 3 and filename[0].isalpha() and filename[1:3].isdigit():
        return filename[:3]
    
    return None


def get_iam_splits(line_strokes_dir: Path) -> Dict[str, List[Path]]:
    """Get train/validation/test splits for IAM-OnDB using 80/10/10 by writer."""
    all_files = []
    for root, dirs, files in os.walk(line_strokes_dir):
        for file in files:
            if file.startswith('.'):
                continue
            file_path = Path(root) / file
            if file_path.is_file() and file_path.suffix == '.xml':
                all_files.append(file_path)
    
    if not all_files:
        print("⚠️  No XML files found in lineStrokes directory!")
        return {'train': [], 'valid': [], 'test': []}
    
    print(f"Found {len(all_files)} XML files in lineStrokes directory")
    
    # Group by writer
    writer_files = {}
    for file_path in all_files:
        writer = get_writer_from_path(file_path)
        if writer:
            if writer not in writer_files:
                writer_files[writer] = []
            writer_files[writer].append(file_path)
    
    if not writer_files:
        print("⚠️  Could not extract writer IDs! Using random split by file.")
        random.seed(42)
        random.shuffle(all_files)
        n = len(all_files)
        return {
            'train': all_files[:int(0.8*n)],
            'valid': all_files[int(0.8*n):int(0.9*n)],
            'test': all_files[int(0.9*n):]
        }
    
    print(f"Found {len(writer_files)} writers")
    
    # Shuffle writer list with fixed seed
    writer_list = sorted(writer_files.keys())
    random.seed(42)
    random.shuffle(writer_list)
    
    n_writers = len(writer_list)
    n_train = int(0.8 * n_writers)
    n_val = int(0.1 * n_writers)
    
    train_writers = set(writer_list[:n_train])
    val_writers = set(writer_list[n_train:n_train + n_val])
    test_writers = set(writer_list[n_train + n_val:])
    
    train_files = []
    valid_files = []
    test_files = []
    
    for writer, files in writer_files.items():
        if writer in train_writers:
            train_files.extend(files)
            print(f"  {writer} -> TRAIN ({len(files)} files)")
        elif writer in val_writers:
            valid_files.extend(files)
            print(f"  {writer} -> VALID ({len(files)} files)")
        elif writer in test_writers:
            test_files.extend(files)
            print(f"  {writer} -> TEST ({len(files)} files)")
    
    print(f"\nSplit sizes - Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
    
    return {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }


def preprocess_iam_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a single IAM-OnDB sample."""
    drawing = sample['drawing']
    writing_guide = sample['writing_guide']
    
    strokes = process_strokes(drawing, writing_guide)
    
    return {
        'key': sample.get('key', ''),
        'strokes': strokes,
        'text': sample.get('text', ''),
        'dataset': 'iam_ondb'
    }


def preprocess_iam_dataset(line_strokes_dir: Path, ascii_dir: Path, output_dir: Path, 
                          split_files: Dict[str, List[Path]]):
    """Preprocess IAM-OnDB dataset."""
    print(f"\n📁 Processing IAM-OnDB dataset from {line_strokes_dir}")
    
    total_samples = 0
    total_with_text = 0
    text_lengths = []
    
    for split, file_list in split_files.items():
        if not file_list:
            print(f"⚠️  No files found for {split} split!")
            continue
            
        print(f"\nProcessing {split} split ({len(file_list)} files)...")
        processed = []
        
        for file_path in tqdm(file_list, desc=f"Processing {split}"):
            sample = parse_line_stroke_file(file_path)
            if sample is None:
                continue
            
            form_id = file_path.stem
            text = get_transcription(ascii_dir, form_id)
            
            if not text or len(text) < 2:
                continue
            
            sample['text'] = text
            processed_sample = preprocess_iam_sample(sample)
            processed.append(processed_sample)
            total_samples += 1
            total_with_text += 1
            text_lengths.append(len(text))
        
        if processed:
            save_processed_data(processed, output_dir, split)
            print(f"  Saved {len(processed)} samples with text labels")
        else:
            print(f"  ⚠️  No valid samples found for {split}")
    
    print(f"\n📊 IAM-OnDB Dataset Statistics:")
    print(f"  Total samples with text: {total_with_text}")
    if text_lengths:
        avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        min_len = min(text_lengths) if text_lengths else 0
        max_len = max(text_lengths) if text_lengths else 0
        print(f"  Avg text length: {avg_len:.1f} chars")
        print(f"  Min text length: {min_len}")
        print(f"  Max text length: {max_len}")
    
    for split in ['train', 'valid', 'test']:
        file_path = output_dir / f"{split}.pt"
        if file_path.exists():
            data = torch.load(file_path, weights_only=False)
            print(f"  {split}: {len(data)} samples")


def main():
    """Main entry point for IAM-OnDB preprocessing."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    data_dir = repo_root / "data" / "raw" / "IAM-OnDB"
    line_strokes_dir = data_dir / "lineStrokes"
    ascii_dir = data_dir / "ascii"
    output_dir = repo_root / "data" / "processed" / "iam_ondb"
    
    print(f"Script directory: {script_dir}")
    print(f"Repo root: {repo_root}")
    print(f"Looking for IAM-OnDB lineStrokes at: {line_strokes_dir}")
    print(f"Looking for IAM-OnDB ascii at: {ascii_dir}")
    
    if not line_strokes_dir.exists():
        print(f"❌ ERROR: lineStrokes directory not found at {line_strokes_dir}")
        return
    
    if not ascii_dir.exists():
        print(f"⚠️  Warning: ascii directory not found at {ascii_dir}")
    
    all_files = list(line_strokes_dir.rglob("*.xml"))
    
    if all_files:
        print(f"✅ Found {len(all_files)} XML files in lineStrokes")
    else:
        print(f"❌ No XML files found in {line_strokes_dir}")
        return
    
    split_files = get_iam_splits(line_strokes_dir)
    
    total_files = sum(len(f) for f in split_files.values())
    if total_files == 0:
        print("❌ No files found in any split!")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_iam_dataset(line_strokes_dir, ascii_dir, output_dir, split_files)
    
    print("\n✅ IAM-OnDB preprocessing complete!")


if __name__ == "__main__":
    main()