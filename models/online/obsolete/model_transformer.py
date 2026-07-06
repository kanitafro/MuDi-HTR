"""
Transformer-based model for online handwriting recognition.
Uses a Transformer encoder with positional encoding and temporal downsampling.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
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
    """
    def __init__(self, input_size=3, d_model=512, nhead=8, num_layers=6,
                 num_classes=95, dropout=0.3, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # 1. Temporal CNN (downsampling)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
        )

        # 2. Projection to d_model
        self.projection = nn.Linear(128, d_model)

        # 3. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Output layer
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_size)
        # 1. CNN downsampling
        x = x.permute(0, 2, 1)          # (batch, input_size, seq_len)
        x = self.cnn(x)                 # (batch, 128, seq_len/4)
        x = x.permute(0, 2, 1)          # (batch, seq_len/4, 128)

        # Update lengths approx (factor 4)
        if lengths is not None:
            lengths = torch.ceil(lengths.float() / 4).long()
            lengths = torch.clamp(lengths, min=1)

        # 2. Project
        x = self.projection(x)           # (batch, T, d_model)
        x = self.dropout(x)

        # 3. Positional encoding
        x = self.pos_encoder(x)

        # 4. Transformer (mask padding)
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)

        # 5. Output
        x = self.dropout(x)
        logits = self.fc(x)              # (batch, T, num_classes)
        logits = logits.permute(1, 0, 2) # (T, batch, num_classes) for CTC

        return logits