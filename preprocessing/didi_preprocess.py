"""DIDI dataset preprocessing for online handwriting recognition."""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import torch
from tqdm import tqdm

from preprocessing.utils import process_strokes, save_processed_data


def clean_text(text: str) -> str:
    """
    Clean text by removing unwanted characters and normalizing whitespace.
    """
    if not text:
        return text
    
    # Replace newlines and carriage returns with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Keep only alphanumeric, spaces, and common punctuation
    # This should match your alphabet
    allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:-_()[]{}<>/\\'\""
    text = ''.join(c for c in text if c in allowed_chars or c.isspace())
    
    return text.strip()


def load_didi_ndjson(file_path: Path) -> List[Dict[str, Any]]:
    """Load all entries from a DIDI .ndjson file."""
    data = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading DIDI data"):
            if line.strip():
                data.append(json.loads(line))
    return data


def get_dot_content(label_id: str, dot_dir: Path) -> str:
    """
    Get the content of the dot file for a given label_id.
    Falls back to label_id if dot file not found.
    """
    dot_path = dot_dir / f"{label_id}.dot"
    if dot_path.exists():
        try:
            with open(dot_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return clean_text(content)
        except Exception:
            pass
    return label_id  # Fallback to label_id


def preprocess_didi_sample(sample: Dict[str, Any], dot_dir: Path) -> Dict[str, Any]:
    """
    Preprocess a single DIDI sample.
    
    Args:
        sample: Single entry from DIDI NDJSON
        dot_dir: Directory containing .dot files
    
    Returns:
        Processed sample with strokes and text label
    """
    drawing = sample['drawing']
    writing_guide = sample['writing_guide']
    
    # Process strokes
    strokes = process_strokes(drawing, writing_guide)
    
    # Get text from dot file
    label_id = sample.get('label_id', '')
    text = get_dot_content(label_id, dot_dir)
    
    return {
        'key': sample['key'],
        'split': sample['split'],
        'strokes': strokes,
        'label_id': label_id,
        'text': text,
        'dataset': 'didi'
    }


def preprocess_didi_dataset(data_path: Path, dot_dir: Path, output_dir: Path):
    """
    Preprocess the entire DIDI dataset.
    """
    print("Loading DIDI dataset...")
    data = load_didi_ndjson(data_path)
    print(f"Loaded {len(data)} samples")
    
    # Process by split
    for split in ['train', 'valid', 'test']:
        split_samples = [s for s in data if s.get('split') == split]
        print(f"Processing {len(split_samples)} samples for {split} split...")
        
        processed = []
        for sample in tqdm(split_samples, desc=f"Processing {split}"):
            try:
                processed_sample = preprocess_didi_sample(sample, dot_dir)
                processed.append(processed_sample)
            except Exception as e:
                print(f"Error processing sample {sample.get('key')}: {e}")
                continue
        
        save_processed_data(processed, output_dir, split)
    
    # Print statistics
    print("\n📊 DIDI Dataset Statistics:")
    for split in ['train', 'valid', 'test']:
        file_path = output_dir / f"{split}.pt"
        if file_path.exists():
            data = torch.load(file_path, weights_only=False)
            with_text = sum(1 for s in data if s['text'] and s['text'] != s['label_id'])
            total = len(data)
            # Check text lengths
            lengths = [len(s['text']) for s in data]
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            print(f"  {split}: {total} samples, {with_text} with dot text ({with_text/total*100:.1f}%)")
            print(f"  Avg text length: {avg_len:.1f} chars")


def main():
    """Main entry point for DIDI preprocessing."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent  # preprocessing/
    repo_root = script_dir.parent       # mudi/
    
    # Build paths relative to repo root
    base_dir = repo_root / "data" / "raw" / "didi_dataset"
    data_path = base_dir / "diagrams_20200131.ndjson"
    
    # The dot files are inside diagrams_20200131_prompts/
    prompt_dir = base_dir / "diagrams_20200131_prompts"
    dot_dir = prompt_dir / "dot"
    
    # If dot_dir doesn't exist, try alternate location
    if not dot_dir.exists():
        dot_dir = base_dir / "dot"
    
    output_dir = repo_root / "data" / "processed" / "online" / "didi"
    
    print(f"Script directory: {script_dir}")
    print(f"Repo root: {repo_root}")
    print(f"Looking for data at: {data_path}")
    print(f"Looking for dot files at: {dot_dir}")
    
    if not data_path.exists():
        print(f"❌ ERROR: Data file not found at {data_path}")
        return
    
    if not dot_dir.exists():
        print(f"⚠️  Warning: Dot directory not found at {dot_dir}")
        print("   Using label_id as fallback text...")
    else:
        # Check if there are dot files
        dot_files = list(dot_dir.glob("*.dot"))
        print(f"✅ Found {len(dot_files)} .dot files in {dot_dir}")
        if dot_files:
            # Show a sample
            try:
                with open(dot_files[0], 'r') as f:
                    sample_content = f.read().strip()
                    print(f"   Sample dot content (cleaned): {clean_text(sample_content)[:100]}...")
            except Exception as e:
                print(f"   Could not read sample: {e}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_didi_dataset(data_path, dot_dir, output_dir)
    
    print("\n✅ DIDI preprocessing complete!")


if __name__ == "__main__":
    main()