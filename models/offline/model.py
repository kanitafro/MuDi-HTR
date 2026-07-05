from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, pool: tuple[int, int]) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: tuple[int, int], dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.pool = nn.MaxPool2d(pool)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.pool(x)
        residual = self.pool(residual)
        x = torch.relu(x + residual)
        return x


class CRNN(nn.Module):
    """Residual CRNN for offline handwriting recognition."""

    def __init__(self, num_classes: int = 80, hidden_size: int = 256) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            ResidualConvBlock(1, 64, (2, 2), dropout=0.05),
            ResidualConvBlock(64, 128, (2, 2), dropout=0.05),
            ResidualConvBlock(128, 256, (2, 1), dropout=0.10),
            ResidualConvBlock(256, 384, (2, 1), dropout=0.10),
            ResidualConvBlock(384, 512, (2, 1), dropout=0.10),
        )
        self.final_pool = nn.MaxPool2d((4, 1))  # 4x128 -> 1x128

        self.sequence_projection = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.35,
        )
        self.sequence_dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ CRITICAL: Validate input at entry point
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {x.shape}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 channel (grayscale), got {x.shape[1]}")
        if x.shape[2] != 128 or x.shape[3] != 512:
            raise ValueError(f"Expected (128, 512) input size, got {x.shape[2:]}")
        if x.dtype != torch.float32:
            raise TypeError(f"Expected float32 input, got {x.dtype}")
        
        features = self.cnn(x)
        features = self.final_pool(features)

        batch_size, channels, height, width = features.shape
        if height != 1:
            raise ValueError(f"Expected pooled feature height of 1, got {height}.")

        sequence = features.squeeze(2).permute(0, 2, 1)  # (batch, width, channels)
        sequence = self.sequence_projection(sequence)
        sequence, _ = self.rnn(sequence)
        sequence = self.sequence_dropout(sequence)
        logits = self.classifier(sequence)
        return logits.permute(1, 0, 2)

if __name__ == "__main__":
    model = CRNN(num_classes=80)
    test_batch = torch.randn(4, 1, 128, 512)
    output = model(test_batch)
    print("CRNN architecture successfully loaded!")
    print("Output shape for CTC Loss (Sequence Length, Batch, Classes):", output.shape)
