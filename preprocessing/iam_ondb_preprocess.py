"""IAM-OnDB dataset preprocessing for online handwriting recognition."""

import json
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch

from preprocessing.utils import process_strokes, save_processed_data, get_text_length

def parse_inkml(file_path: Path) -> Dict[str, Any]:
    """
    Parse an InkML file from IAM-OnDB dataset.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Get trace groups (strokes)
        traces = []
        for trace in root.findall('.//{http://www.w3.org/2003/InkML}trace'):
            points = []
            for point_str in trace.text.strip().split(','):
                if point_str.strip():
                    coords = [float(x) for x in point_str.strip().split() if x]
                    if len(coords) >= 2:
                        # Sometimes there are more than 2 coords, we take x, y
                        points.append([coords[0], coords[1], 0])  # No timestamp
            if points:
                traces.append(points)
        
        # Get text label (transcription)
        label = "unknown"
        for annotation in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
            if annotation.get('type') == 'truth':
                label = annotation.text.strip()
                break
        
        # Get writing guide if available
        writing_guide = {'width': 1000, 'height': 1000}  # Default
        for trace_format in root.findall('.//{http://www.w3.org/2003/InkML}traceFormat'):
            for canvas in trace_format.findall('.//{http://www.w3.org/2003/InkML}canvas'):
                width = canvas.get('width')
                height = canvas.get('height')
                if width and height:
                    writing_guide = {'width': float(width), 'height': float(height)}
                    break
        
        return {
            'drawing': traces,
            'writing_guide': writing_guide,
            'text': label,
            'key': file_path.stem
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def preprocess_iam_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single IAM-OnDB sample.
    """
    drawing = sample['drawing']
    writing_guide = sample['writing_guide']
    
    # Process strokes
    strokes = process_strokes(drawing, writing_guide)
    
    return {
        'key': sample.get('key', ''),
        'strokes': strokes,
        'text': sample.get('text', ''),
        'dataset': 'iam_ondb'
    }

def preprocess_iam_dataset(input_dir: Path, output_dir: Path, split_files: Dict[str, List[str]]):
    """
    Preprocess IAM-OnDB dataset.
    
    Args:
        input_dir: Directory containing InkML files
        output_dir: Output directory for processed files
        split_files: Dict mapping 'train', 'valid', 'test' to list of file paths
    """
    print(f"Processing IAM-OnDB dataset from {input_dir}")
    
    for split, file_list in split_files.items():
        print(f"\nProcessing {split} split ({len(file_list)} files)...")
        processed = []
        
        for file_path in tqdm(file_list, desc=f"Processing {split}"):
            file_path = Path(file_path)
            if not file_path.exists():
                continue
            
            sample = parse_inkml(file_path)
            if sample is None:
                continue
            
            processed_sample = preprocess_iam_sample(sample)
            if processed_sample['text'] and processed_sample['text'] != 'unknown':
                processed.append(processed_sample)
        
        if processed:
            save_processed_data(processed, output_dir, split)
            print(f"  Saved {len(processed)} samples with text labels")
        else:
            print(f"  ⚠️  No valid samples found for {split}")
    
    # Print statistics
    print("\n📊 IAM-OnDB Dataset Statistics:")
    for split in ['train', 'valid', 'test']:
        file_path = output_dir / f"{split}.pt"
        if file_path.exists():
            data = torch.load(file_path, weights_only=False)
            avg_len = sum(len(s['text']) for s in data) / len(data) if data else 0
            print(f"  {split}: {len(data)} samples, avg text length: {avg_len:.1f}")

def main():
    """Main entry point for IAM-OnDB preprocessing."""
    input_dir = Path("../data/raw/iam_ondb")
    output_dir = Path("../data/processed/iam_ondb")
    
    print(f"Current directory: {Path.cwd()}")
    print(f"Looking for IAM-OnDB data at: {input_dir.absolute()}")
    
    if not input_dir.exists():
        print(f"❌ ERROR: IAM-OnDB directory not found at {input_dir.absolute()}")
        print("Please download IAM-OnDB dataset first.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement proper IAM-OnDB split loading
    # This is a placeholder - you'll need to adapt based on actual IAM-OnDB structure
    inkml_files = list(input_dir.rglob("*.inkml"))
    
    # Simple split (80/10/10)
    import random
    random.seed(42)
    random.shuffle(inkml_files)
    n = len(inkml_files)
    train_files = inkml_files[:int(0.8*n)]
    valid_files = inkml_files[int(0.8*n):int(0.9*n)]
    test_files = inkml_files[int(0.9*n):]
    
    split_files = {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }
    
    preprocess_iam_dataset(input_dir, output_dir, split_files)
    
    print("\n✅ IAM-OnDB preprocessing complete!")

if __name__ == "__main__":
    main()