# models/online/model.py
"""
Online handwriting recognition model: BiLSTM + CTC.
Processes stroke sequences (variable length) and outputs character probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineHTRModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, num_layers=2,
                 num_classes=96, dropout=0.5):
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_size)
        x = self.projection(x)
        x = F.relu(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)          # (batch, seq_len, hidden_size*2)
        x = self.dropout(x)
        logits = self.fc(x)           # (batch, seq_len, num_classes)
        logits = logits.permute(1, 0, 2)  # (seq_len, batch, num_classes)
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

    def beam_search(self, logits, lengths=None, beam_width=10, top_k=5, blank_idx=0):
        """
        CTC beam search decoder.
        
        Args:
            logits: (seq_len, batch, num_classes) or (seq_len, num_classes)
            lengths: optional tensor of valid sequence lengths per batch (seq_len,)
            beam_width: number of top paths to keep at each step
            top_k: number of final hypotheses to return per sample
            blank_idx: index of the blank token (default 0)
        
        Returns:
            List of lists: for each batch sample, a list of (hypothesis_string, score) tuples,
                           sorted by score (descending).
                           Score is the average log-probability per character (length-normalized).
        """
        # Handle batch dimension
        if logits.dim() == 3:
            seq_len, batch_size, num_classes = logits.shape
        else:  # (seq_len, num_classes) -> add batch dim
            logits = logits.unsqueeze(1)
            seq_len, batch_size, num_classes = logits.shape
    
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)  # (seq_len, batch, num_classes)
    
        # If lengths are provided, truncate to valid length (optional, but good)
        if lengths is not None:
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)
            max_len = int(lengths.max().item())
            log_probs = log_probs[:max_len]  # truncate
    
        all_hypotheses = []
    
        for b in range(batch_size):
            # Initialize beams: dict mapping prefix_tuple -> accumulated log score
            beams = {(): 0.0}  # empty prefix
    
            # Iterate over time steps for this batch sample
            for t in range(log_probs.size(0)):
                probs = log_probs[t, b]  # (num_classes,)
                new_beams = {}
    
                for prefix, score in beams.items():
                    # Option 1: predict blank -> prefix unchanged
                    blank_score = score + probs[blank_idx]
                    if prefix in new_beams:
                        new_beams[prefix] = torch.logsumexp(
                            torch.tensor([new_beams[prefix], blank_score]), dim=0
                        ).item()
                    else:
                        new_beams[prefix] = blank_score
    
                    # Option 2: predict a non-blank character
                    for c in range(num_classes):
                        if c == blank_idx:
                            continue
                        char_score = score + probs[c]
                        # If last char is same as c, merge (CTC rule)
                        if prefix and prefix[-1] == c:
                            new_prefix = prefix
                        else:
                            new_prefix = prefix + (c,)
    
                        if new_prefix in new_beams:
                            new_beams[new_prefix] = torch.logsumexp(
                                torch.tensor([new_beams[new_prefix], char_score]), dim=0
                            ).item()
                        else:
                            new_beams[new_prefix] = char_score
    
                # Prune to beam_width
                if len(new_beams) > beam_width:
                    # Keep only top beam_width prefixes by score
                    sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
                    beams = dict(sorted_beams[:beam_width])
                else:
                    beams = new_beams
    
            # After all time steps, normalize scores by length and return top_k
            # Convert prefixes to strings and normalize scores
            candidates = []
            for prefix, score in beams.items():
                # Length-normalize (average log prob per character)
                if len(prefix) > 0:
                    norm_score = score / len(prefix)
                else:
                    norm_score = score  # empty string
                text = ''.join([self.alphabet[idx] for idx in prefix])
                candidates.append((text, norm_score))
    
            # Sort by score descending and take top_k
            candidates.sort(key=lambda x: x[1], reverse=True)
            all_hypotheses.append(candidates[:top_k])
    
        return all_hypotheses
    
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

"""
class OnlineHTRModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, num_layers=2,
                 num_classes=96, dropout=0.4):
        super().__init__()
        # --- 1D CNN (temporal) ---
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
        )
        # After CNN: (batch, 64, seq_len/4)

        # Projection to hidden_size (optional)
        self.projection = nn.Linear(64, hidden_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)          # (batch, input_size, seq_len)
        x = self.cnn(x)                 # (batch, 64, seq_len/4)
        x = x.permute(0, 2, 1)          # (batch, seq_len/4, 64)
        x = self.projection(x)          # (batch, seq_len/4, hidden_size)
        x = F.relu(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)             # (batch, seq_len/4, hidden_size*2)
        x = self.dropout(x)
        logits = self.fc(x)             # (batch, seq_len/4, num_classes)
        logits = logits.permute(1, 0, 2) # (seq_len/4, batch, num_classes)
        return logits
"""

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