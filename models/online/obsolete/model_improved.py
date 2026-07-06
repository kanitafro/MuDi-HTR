# models/online/model_improved.py
"""
Improved online handwriting recognition model: CNN + BiLSTM + CTC.
Processes stroke sequences with downsampling and attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedOnlineHTRModel(nn.Module):
    """
    Improved model for long-sequence handwriting recognition.
    
    Architecture:
    1. Temporal CNN (downsampling) → reduces sequence length
    2. Bidirectional LSTM (deep) → captures long-range dependencies
    3. CTC output layer
    """
    
    def __init__(self, input_size=3, hidden_size=512, num_layers=4,
                 num_classes=95, dropout=0.3):
        super(ImprovedOnlineHTRModel, self).__init__()
        
        # 1. Temporal CNN: Downsample the sequence
        # Input: (batch, seq_len, 3) → Permute to (batch, 3, seq_len)
        self.temporal_cnn = nn.Sequential(
            # Conv1D: (batch, 3, seq_len) → (batch, 64, seq_len/2)
            nn.Conv1d(input_size, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            
            # Conv1D: (batch, 64, seq_len/2) → (batch, 128, seq_len/4)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
        )
        
        # Calculate the output dimension after CNN
        # For a 5000-point sequence: 5000 → 2500 → 1250
        self.cnn_output_size = 128  # features per timestep
        
        # 2. Projection layer: CNN features → LSTM input
        self.projection = nn.Linear(128, hidden_size)
        
        # 3. Deep Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 4. Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            lengths: (batch,) original sequence lengths
        Returns:
            logits: (seq_len, batch, num_classes)
        """
        # Save original lengths for later
        orig_lengths = lengths
        
        # 1. CNN downsample
        # (batch, seq_len, 3) → (batch, 3, seq_len)
        x = x.permute(0, 2, 1)
        x = self.temporal_cnn(x)  # (batch, 128, seq_len/4)
        
        # (batch, 128, seq_len/4) → (batch, seq_len/4, 128)
        x = x.permute(0, 2, 1)
        
        # Update lengths (approximate)
        if lengths is not None:
            # Each conv layer with stride=2 roughly halves the length
            lengths = torch.ceil(lengths / 4).long()
            # Ensure at least 1
            lengths = torch.clamp(lengths, min=1)
        
        # 2. Project to hidden_size
        x = self.projection(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 3. LSTM
        if lengths is not None:
            # Pack sequences
            lengths, sorted_idx = lengths.sort(descending=True)
            x = x[sorted_idx]
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
            x, _ = self.lstm(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            
            # Restore order
            _, original_idx = sorted_idx.sort()
            x = x[original_idx]
        else:
            x, _ = self.lstm(x)
        
        # 4. Output
        x = self.dropout(x)
        logits = self.fc(x)
        
        # CTC expects (seq_len, batch, num_classes)
        logits = logits.permute(1, 0, 2)
        
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerHTRModel(nn.Module):
    """
    Transformer-based model for online handwriting recognition.
    Better at capturing long-range dependencies than LSTMs.
    """
    
    def __init__(self, input_size=3, hidden_size=512, num_layers=4,
                 num_classes=95, dropout=0.3, nhead=8):
        super(TransformerHTRModel, self).__init__()
        
        # 1. Temporal CNN (downsampling)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
        )
        
        # 2. Projection
        self.projection = nn.Linear(128, hidden_size)
        
        # 3. Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            lengths: (batch,) original sequence lengths
        """
        # 1. CNN downsample
        x = x.permute(0, 2, 1)
        x = self.temporal_cnn(x)
        x = x.permute(0, 2, 1)
        
        # 2. Project
        x = self.projection(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # 3. Positional encoding
        x = self.pos_encoder(x)
        
        # 4. Transformer (generate attention mask for padding)
        if lengths is not None:
            # Approximate lengths after downsampling
            lengths = torch.ceil(lengths / 4).long()
            lengths = torch.clamp(lengths, min=1)
            
            # Create padding mask
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # 5. Output
        x = self.dropout(x)
        logits = self.fc(x)
        
        # CTC expects (seq_len, batch, num_classes)
        logits = logits.permute(1, 0, 2)
        
        return logits