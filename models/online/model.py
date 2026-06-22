# models/online/model.py
"""
Online handwriting recognition model: BiLSTM + CTC.
Processes stroke sequences (variable length) and outputs character probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineHTRModel(nn.Module):
    """
    Bidirectional LSTM with CTC for online handwriting recognition.
    
    Input: (batch, seq_len, 3) - normalized (x, y, t) features
    Output: (seq_len, batch, num_classes) - logits for CTC loss
    """
    
    def __init__(self, input_size=3, hidden_size=256, num_layers=3, 
                 num_classes=80, dropout=0.3):
        """
        Args:
            input_size: Number of features per point (x, y, t) = 3
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output characters (including blank for CTC)
            dropout: Dropout probability between LSTM layers
        """
        super(OnlineHTRModel, self).__init__()
        
        # Optional: Linear projection before LSTM to increase capacity
        self.projection = nn.Linear(input_size, hidden_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer: maps to character classes
        # +1 for CTC blank token (handled by CTCLoss)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            lengths: Optional tensor of actual sequence lengths for packed sequences
        
        Returns:
            logits: Shape (seq_len, batch, num_classes) - ready for CTC loss
        """
        # Project input to hidden_size
        x = self.projection(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Pack sequences for variable length (optional but efficient)
        if lengths is not None:
            # Sort by length for packing
            lengths, sorted_idx = lengths.sort(descending=True)
            x = x[sorted_idx]
            
            # Pack the sequences
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
            x, _ = self.lstm(x)
            # Unpack
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            
            # Restore original order
            _, original_idx = sorted_idx.sort()
            x = x[original_idx]
        else:
            # Standard forward without packing
            x, _ = self.lstm(x)
        
        # Apply dropout and linear layer
        x = self.dropout(x)
        logits = self.fc(x)
        
        # CTC expects (seq_len, batch, num_classes)
        logits = logits.permute(1, 0, 2)
        
        return logits


class CTCDecoder:
    """
    CTC decoder for converting logits to text.
    Supports greedy decoding and beam search (optional).
    """
    
    def __init__(self, alphabet, blank_idx=0):
        """
        Args:
            alphabet: List of characters (e.g., [' ', 'a', 'b', ..., 'z'])
            blank_idx: Index of blank token (usually 0)
        """
        self.alphabet = alphabet
        self.blank_idx = blank_idx
        
    def greedy_decode(self, logits, lengths=None):
        """
        Greedy decoding: take argmax at each timestep and collapse repeats/blank.
        
        Args:
            logits: (seq_len, batch, num_classes)
            lengths: Optional sequence lengths
        
        Returns:
            List of decoded strings
        """
        # Get predictions (seq_len, batch)
        preds = logits.argmax(dim=-1)  # (seq_len, batch)
        
        # Transpose to (batch, seq_len)
        preds = preds.transpose(0, 1)
        
        # Apply CTC collapse (remove blanks and duplicates)
        decoded = []
        for batch_idx in range(preds.size(0)):
            seq = preds[batch_idx]
            # Get valid length
            if lengths is not None:
                seq = seq[:lengths[batch_idx]]
            
            # Collapse: remove blank, remove consecutive duplicates
            chars = []
            prev = -1
            for idx in seq:
                if idx != self.blank_idx and idx != prev:
                    chars.append(idx)
                prev = idx
            
            # Convert indices to characters
            text = ''.join([self.alphabet[idx] for idx in chars])
            decoded.append(text)
        
        return decoded
    
    def get_confidence(self, logits):
        """
        Extract confidence from CTC logits.
        
        Returns:
            confidence: Max softmax probability averaged over timesteps
        """
        probs = F.softmax(logits, dim=-1)  # (seq_len, batch, num_classes)
        max_probs, _ = probs.max(dim=-1)   # (seq_len, batch)
        avg_confidence = max_probs.mean(dim=0)  # (batch,)
        return avg_confidence


if __name__ == "__main__":
    # Smoke test
    model = OnlineHTRModel(input_size=3, hidden_size=256, num_layers=3, 
                          num_classes=80, dropout=0.3)
    
    # Simulate batch: (batch=4, seq_len=100, features=3)
    test_input = torch.randn(4, 100, 3)
    output = model(test_input)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape} (seq_len, batch, classes)")
    print("✅ OnlineHTRModel initialized successfully!")