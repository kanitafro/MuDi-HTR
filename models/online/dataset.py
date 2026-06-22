"""
PyTorch Dataset for online handwriting recognition.
Supports both DIDI and IAM-OnDB datasets.
"""

import torch
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
        """
        Args:
            data_path: Path to directory containing split .pt files
            split: One of 'train', 'validation', 'test'
            max_seq_len: Maximum sequence length (for truncation/padding)
            dataset_name: Optional name for logging ('didi' or 'iam_ondb')
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_seq_len = max_seq_len
        self.dataset_name = dataset_name or data_path.parent.name
        
        # Load the preprocessed data
        file_path = self.data_path / f"{split}.pt"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load with weights_only=False because our .pt files contain data, not model weights
        self.data = torch.load(file_path, weights_only=False)
        print(f"Loaded {len(self.data)} samples from {self.dataset_name} {split} split")
        
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
        
        # Concatenate all strokes into a single sequence
        # Shape: (total_points, 3)
        sequence = torch.cat(tensor_strokes, dim=0)
        
        # Optional: truncate long sequences
        if self.max_seq_len is not None and sequence.shape[0] > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        
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