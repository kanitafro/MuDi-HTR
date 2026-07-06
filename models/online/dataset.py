# models/online/dataset.py
"""
PyTorch Dataset for online handwriting recognition.
Supports both DIDI and IAM-OnDB datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class OnlineHandwritingDataset(Dataset):
    """
    Dataset for online handwriting recognition.
    Loads preprocessed .pt files from the preprocessing step.
    Supports both DIDI and IAM-OnDB datasets.
    """
    
    def __init__(self, data_path: Path, split: str, max_seq_len: int = None, dataset_name: str = None):
        self.data_path = Path(data_path)
        self.split = split
        self.max_seq_len = max_seq_len
        self.dataset_name = dataset_name or data_path.parent.name
        
        # Load preprocessed data
        file_path = self.data_path / f"{split}.pt"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        self.data = torch.load(file_path, weights_only=False)
        print(f"Loaded {len(self.data)} samples from {self.dataset_name} {split} split")
        
        # --- Load global feature statistics (if they exist) ---
        stats_path = Path(__file__).parent / "feature_stats.pt"
        if stats_path.exists():
            stats = torch.load(stats_path, weights_only=False)
            self.global_mean = stats['mean']
            self.global_std = stats['std']
            print(f"✅ Loaded global stats: mean={self.global_mean}, std={self.global_std}")
        else:
            self.global_mean = None
            self.global_std = None
            print("⚠️ Global stats not found; skipping normalization")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Get strokes: list of arrays/tensors, each (L_s, 3)
        strokes = sample['strokes']
        
        # Convert numpy arrays to tensors if needed
        tensor_strokes = []
        for stroke in strokes:
            if isinstance(stroke, np.ndarray):
                tensor_strokes.append(torch.from_numpy(stroke).float())
            elif isinstance(stroke, torch.Tensor):
                tensor_strokes.append(stroke.float())
            else:
                # If it's a list or other type, convert
                tensor_strokes.append(torch.tensor(stroke, dtype=torch.float32))
        
        ## Concatenate all strokes into a single sequence
        # Shape: (total_points, 3)
        sequence = torch.cat(tensor_strokes, dim=0)  # (seq_len, 3)
        
        # --- EXTRACT RAW FEATURES ---
        x = sequence[:, 0]          # (seq_len,)
        y = sequence[:, 1]          # (seq_len,)
        pen = sequence[:, 2]        # (seq_len,) - 0=down, 1=up
        
        # --- COMPUTE DERIVATIVES (Step 1) ---
        # 1. Differential coordinates (dx, dy)
        dx = torch.diff(x, prepend=x[0:1])   # Pad first value with itself
        dy = torch.diff(y, prepend=y[0:1])   # Pad first value with itself
        
        # 2. Log magnitude (add epsilon to avoid log(0))
        r = torch.sqrt(dx**2 + dy**2) + 1e-8
        log_r = torch.log(r)
        
        # 3. Angle (direction of pen movement)
        angle = torch.atan2(dy, dx)
        
        # 4. Pen state flags (explicit up/down)
        pen_up = (pen == 1).float()          # 1 if pen is up
        pen_down = (pen == 0).float()        # 1 if pen is down
        
        # --- STACK INTO NEW FEATURE VECTOR (6 features) ---
        sequence = torch.stack([dx, dy, log_r, angle, pen_up, pen_down], dim=1)
        
        # --- NORMALIZE USING GLOBAL STATS (NOT per-sample) ---
        if self.global_mean is not None and self.global_std is not None:
            # Normalize only features 0-3 (dx, dy, log_r, angle)
            # Keep pen flags (4,5) unchanged (0 or 1)
            sequence[:, :4] = (sequence[:, :4] - self.global_mean) / self.global_std
        
        # Shape: (seq_len, 6)
        
        # Optional: truncate long sequences
        #if self.max_seq_len is not None and sequence.shape[0] > self.max_seq_len:
        #    sequence = sequence[:self.max_seq_len]

        # --- DATA AUGMENTATION (only for training) ---
        max_angle_deg = 8   # in degrees
        max_angle = max_angle_deg * 3.14159 / 180   # in radians
        angle_rot = (torch.rand(1) - 0.5) * 2 * max_angle
        
        if self.split == 'train':
            # random scaling
            scale = 0.9 + 0.2 * torch.rand(1) # scale 0.9-1.1
            sequence[:, :2] *= scale
            # random rotation
            #angle_rot = torch.randn(1) * 0.15
            angle_rot = (torch.rand(1) - 0.5) * 2 * max_angle
            rot_matrix = torch.tensor([[torch.cos(angle_rot), -torch.sin(angle_rot)],
                                       [torch.sin(angle_rot), torch.cos(angle_rot)]])
            sequence[:, :2] = sequence[:, :2] @ rot_matrix
            # small noise
            sequence[:, :2] += torch.randn_like(sequence[:, :2]) * 0.005
            
        # Get text label
        text = sample.get('text', '')
        
        return {
            'sequence': sequence,          # (seq_len, 3)
            'text': text,                  # string
            'key': sample.get('key', ''),  # unique identifier
            'original_len': sequence.shape[0],  # for tracking
            'dataset': sample.get('dataset', self.dataset_name)  # dataset name
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for variable-length sequences.
        Pads sequences to max length in batch.
        """
        sequences = [item['sequence'] for item in batch]
        texts = [item['text'] for item in batch]
        keys = [item.get('key', '') for item in batch]
        original_lens = [item['original_len'] for item in batch]
        
        # Find max length in batch
        max_len = max(seq.shape[0] for seq in sequences)
        
        # Pad sequences to max_len
        padded_sequences = []
        for seq in sequences:
            if seq.shape[0] < max_len:
                pad = torch.zeros(max_len - seq.shape[0], seq.shape[1])
                padded = torch.cat([seq, pad], dim=0)
            else:
                padded = seq
            padded_sequences.append(padded)
        
        # Stack into batch tensor
        sequences = torch.stack(padded_sequences, dim=0)  # (batch, max_len, 3)
        
        # Original lengths for CTC loss
        lengths = torch.tensor(original_lens, dtype=torch.long)
        
        return {
            'sequences': sequences,
            'texts': texts,
            'keys': keys,
            'lengths': lengths
        }


class CTCLabelEncoder:
    """
    Encoder for CTC labels.
    Converts text strings to integer indices (with blank at index 0).
    """
    
    def __init__(self, alphabet):
        """
        Args:
            alphabet: List of characters (e.g., [' ', 'a', 'b', ..., 'z'])
        """
        self.alphabet = alphabet
        self.char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
        self.idx_to_char = {idx: char for idx, char in enumerate(alphabet)}
        self.blank_idx = 0  # blank token is always index 0
        
    def encode(self, text):
        """
        Convert text string to list of integer indices.
        
        Args:
            text: String of characters
        
        Returns:
            List of indices
        """
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Skip unknown characters
                # Optionally, map to a special UNK token
                print(f"Warning: Character '{char}' not in alphabet, skipping")
        return indices
    
    def decode(self, indices):
        """
        Convert indices back to text (for debugging).
        
        Args:
            indices: List of integer indices
        
        Returns:
            String of characters
        """
        return ''.join([self.idx_to_char[idx] for idx in indices if idx != self.blank_idx])
    
    def collate_labels(self, texts):
        """
        Prepare labels for CTC loss.
        
        Returns:
            labels: Concatenated label indices for all samples
            label_lengths: Length of each label sequence
        """
        encoded_labels = [torch.tensor(self.encode(text), dtype=torch.long) 
                         for text in texts]
        label_lengths = torch.tensor([len(label) for label in encoded_labels], 
                                     dtype=torch.long)
        labels = torch.cat(encoded_labels, dim=0)
        return labels, label_lengths


if __name__ == "__main__":
    # Test the dataset
    data_dir = Path("../data/processed/online/didi")
    
    # Define alphabet (including blank at index 0)
    alphabet = [' '] + [chr(i) for i in range(97, 123)] + [chr(i) for i in range(65, 91)] + \
               [str(i) for i in range(10)] + ['!', '?', '.', ',', '-', "'", '"']
    
    # Create dataset
    dataset = OnlineHandwritingDataset(data_dir, 'train', max_seq_len=3000)
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, 
                       collate_fn=dataset.collate_fn)
    
    batch = next(iter(loader))
    print(f"Batch sequences shape: {batch['sequences'].shape}")
    print(f"Batch lengths: {batch['lengths']}")
    print(f"Batch texts: {batch['texts']}")
    print("✅ Dataset and collate function working!")