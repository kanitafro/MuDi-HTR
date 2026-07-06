from __future__ import annotations

import numpy as np
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

    @staticmethod
    def _logsumexp(values: list[float]) -> float:
        finite_values = [value for value in values if value > -np.inf]
        if not finite_values:
            return -np.inf
        max_value = max(finite_values)
        return float(max_value + np.log(sum(np.exp(value - max_value) for value in finite_values)))

    @classmethod
    def _ctc_beam_search_single(
        cls,
        log_probs: np.ndarray,
        alphabet: list[str],
        beam_width: int = 10,
        blank_index: int = 0,
    ) -> tuple[str, float]:
        if log_probs.ndim != 2:
            raise ValueError(f"Expected 2D log-probabilities (T, C), got shape {log_probs.shape}")
        if log_probs.shape[1] != len(alphabet):
            raise ValueError(f"Alphabet size {len(alphabet)} does not match logits classes {log_probs.shape[1]}")

        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -np.inf)}
        candidate_width = min(max(beam_width * 2, 10), max(1, log_probs.shape[1] - 1))

        for timestep in log_probs:
            next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
            top_indices = np.argsort(timestep)[::-1][:candidate_width]

            def update(
                prefix: tuple[int, ...],
                blank_score: float | None = None,
                nonblank_score: float | None = None,
            ) -> None:
                existing_blank, existing_nonblank = next_beams.get(prefix, (-np.inf, -np.inf))
                if blank_score is not None:
                    existing_blank = cls._logsumexp([existing_blank, blank_score])
                if nonblank_score is not None:
                    existing_nonblank = cls._logsumexp([existing_nonblank, nonblank_score])
                next_beams[prefix] = (existing_blank, existing_nonblank)

            for prefix, (p_blank, p_nonblank) in beams.items():
                total = cls._logsumexp([p_blank, p_nonblank])
                update(prefix, blank_score=total + float(timestep[blank_index]))

                for class_index in top_indices:
                    if class_index == blank_index:
                        continue
                    score = float(timestep[class_index])
                    if prefix and prefix[-1] == class_index:
                        update(prefix, nonblank_score=p_nonblank + score)
                        update(prefix + (class_index,), nonblank_score=p_blank + score)
                    else:
                        update(prefix + (class_index,), nonblank_score=total + score)

            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda item: cls._logsumexp([item[1][0], item[1][1]]),
                    reverse=True,
                )[:beam_width]
            )

        best_prefix, (best_blank, best_nonblank) = max(
            beams.items(),
            key=lambda item: cls._logsumexp([item[1][0], item[1][1]]),
        )
        best_score = cls._logsumexp([best_blank, best_nonblank])
        total_score = cls._logsumexp([cls._logsumexp([blank, nonblank]) for blank, nonblank in beams.values()])

        characters: list[str] = []
        last_token: int | None = None
        for token in best_prefix:
            if token == blank_index or token == last_token:
                last_token = token
                continue
            last_token = token
            if 0 <= token < len(alphabet):
                token_text = alphabet[token]
                if token_text not in {"<BLANK>", "<UNK>"}:
                    characters.append(token_text)

        confidence = float(np.exp(best_score - total_score)) if total_score > -np.inf else 0.0
        return "".join(characters), confidence

    def decode_beam_search(
        self,
        logits: torch.Tensor,
        alphabet: list[str],
        beam_width: int = 10,
        blank_index: int = 0,
    ) -> list[tuple[str, float]]:
        """Decode CTC logits with a pure Python/NumPy beam search.

        Args:
            logits: CTC logits with shape (T, B, C) or (T, C).
            alphabet: Token list where index 0 is the blank symbol.
            beam_width: Beam width for the prefix search.
            blank_index: Index of the blank token.

        Returns:
            A list of (text, probability) tuples, one per batch item.
        """

        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        if logits.ndim != 3:
            raise ValueError(f"Expected logits with shape (T, B, C) or (T, C), got {tuple(logits.shape)}")

        log_probs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
        decoded: list[tuple[str, float]] = []
        for batch_index in range(log_probs.shape[1]):
            text, probability = self._ctc_beam_search_single(
                log_probs[:, batch_index, :],
                alphabet=alphabet,
                beam_width=beam_width,
                blank_index=blank_index,
            )
            decoded.append((text, probability))
        return decoded

if __name__ == "__main__":
    model = CRNN(num_classes=80)
    test_batch = torch.randn(4, 1, 128, 512)
    output = model(test_batch)
    print("CRNN architecture successfully loaded!")
    print("Output shape for CTC Loss (Sequence Length, Batch, Classes):", output.shape)
