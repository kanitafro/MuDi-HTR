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


class CRNN(nn.Module):
    """Compact CRNN baseline for offline handwriting recognition."""

    def __init__(self, num_classes: int = 80, hidden_size: int = 256) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(1, 64, (2, 2)),   # 128x512 -> 64x256
            ConvBlock(64, 128, (2, 2)), # 64x256 -> 32x128
            ConvBlock(128, 256, (2, 1)), # 32x128 -> 16x128
            ConvBlock(256, 256, (2, 1)), # 16x128 -> 8x128
            ConvBlock(256, 512, (2, 1)), # 8x128 -> 4x128
        )
        self.final_pool = nn.MaxPool2d((4, 1))  # 4x128 -> 1x128

        self.sequence_projection = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = self.final_pool(features)

        batch_size, channels, height, width = features.shape
        if height != 1:
            raise ValueError(f"Expected pooled feature height of 1, got {height}.")

        sequence = features.squeeze(2).permute(0, 2, 1)  # (batch, width, channels)
        sequence = self.sequence_projection(sequence)
        sequence, _ = self.rnn(sequence)
        logits = self.classifier(sequence)
        return logits.permute(1, 0, 2)

if __name__ == "__main__":
    model = CRNN(num_classes=80)
    test_batch = torch.randn(4, 1, 128, 512)
    output = model(test_batch)
    print("CRNN architecture successfully loaded!")
    print("Output shape for CTC Loss (Sequence Length, Batch, Classes):", output.shape)
