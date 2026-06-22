import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes=80, hidden_size=256):
        super(CRNN, self).__init__()
        
        # 1. CNN Feature Extractor (Processes the 128x512 normalized tensor)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # Dim: 128x512 -> 64x256
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # Dim: 64x256 -> 32x128
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)) # Shrink only height: 32x128 -> 16x128
        )
        
        # 2. Linear Feature Bridge 
        # Our CNN output map has a height of 16 and 256 feature channels (16 * 256 = 4096)
        self.linear = nn.Linear(4096, hidden_size)
        
        # 3. RNN Sequence Processor (Bidirectional LSTM to read left-to-right text slices)
        self.rnn = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True, 
            dropout=0.3
        )
        
        # 4. Character Classifier (Maps features to your text characters output)
        self.fc = nn.Linear(hidden_size * 2, num_classes) # Bidirectional means hidden_size * 2

    def forward(self, x):
        # Expected input shape: (batch_size, 1, 128, 512)
        features = self.cnn(x)
        
        # Collapse dimensions for Sequence processing
        b, c, h, w = features.size()
        features = features.view(b, c * h, w) # Shape: (batch_size, 4096, width_steps)
        features = features.permute(0, 2, 1)  # Shape: (batch_size, width_steps, 4096)
        
        # Apply mapping to hidden sizes
        rnn_input = self.linear(features)
        
        # Send through sequential BiLSTM layers
        rnn_output, _ = self.rnn(rnn_input)
        
        # Get character token predictions
        logits = self.fc(rnn_output)
        
        # Return permuted for PyTorch's Connectionist Temporal Classification (CTC) Loss: 
        # Output shape: (sequence_length, batch_size, num_classes)
        return logits.permute(1, 0, 2)

if __name__ == "__main__":
    # Smoke-test to verify math dimensions work out perfectly!
    model = CRNN(num_classes=80)
    # Mimic a batch of 4 processed images matching Kanita's shape
    test_batch = torch.randn(4, 1, 128, 512) 
    output = model(test_batch)
    print("🚀 CRNN architecture successfully loaded!")
    print("Output shape for CTC Loss (Sequence Length, Batch, Classes):", output.shape)