# models/online/utils.py
"""
Utility functions for online branch: confidence extraction, model loading, etc.
"""

import torch
import torch.nn.functional as F
from pathlib import Path


from models.online.model import OnlineHTRModel, CTCDecoder


def get_confidence(logits):
    """
    Extract confidence from CTC logits.
    
    Args:
        logits: (seq_len, batch, num_classes)
    
    Returns:
        confidence: (batch,) average max softmax probability
    """
    probs = F.softmax(logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)  # (seq_len, batch)
    avg_confidence = max_probs.mean(dim=0)  # (batch,)
    return avg_confidence


def load_online_model(model_path, device='cpu'):
    """
    Load trained online model for inference.
    
    Args:
        model_path: Path to saved .pth file
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Loaded model in eval mode
        decoder: CTCDecoder instance
        alphabet: List of characters
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    alphabet = checkpoint['alphabet']
    
    # Initialize model
    model = OnlineHTRModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize decoder
    decoder = CTCDecoder(alphabet, blank_idx=0)
    
    return model, decoder, alphabet


def preprocess_stroke_for_inference(strokes, writing_guide):
    """
    Preprocess raw stroke data for inference (same as training preprocessing).
    
    Args:
        strokes: Raw strokes from drawing canvas
        writing_guide: Dict with 'width' and 'height'
    
    Returns:
        sequence: Tensor of shape (seq_len, 3)
    """
    import numpy as np
    
    width = writing_guide['width']
    height = writing_guide['height']
    
    processed_strokes = []
    for stroke in strokes:
        xs, ys, ts = stroke[0], stroke[1], stroke[2]
        
        # Normalize coordinates
        norm_xs = (np.array(xs) / width) * 2 - 1
        norm_ys = (np.array(ys) / height) * 2 - 1
        
        # Normalize timestamps
        ts_array = np.array(ts, dtype=np.float32)
        if ts_array.max() > ts_array.min():
            ts_norm = (ts_array - ts_array.min()) / (ts_array.max() - ts_array.min())
        else:
            ts_norm = np.zeros_like(ts_array)
        
        # Combine
        stroke_data = np.stack([norm_xs, norm_ys, ts_norm], axis=1)
        processed_strokes.append(stroke_data)
    
    # Concatenate all strokes
    sequence = torch.tensor(np.concatenate(processed_strokes, axis=0), dtype=torch.float32)
    
    return sequence


def predict_online(model, decoder, sequence, device='cpu'):
    """
    Run inference on a single sequence.
    
    Args:
        model: Trained OnlineHTRModel
        decoder: CTCDecoder instance
        sequence: Tensor of shape (seq_len, 3)
        device: 'cpu' or 'cuda'
    
    Returns:
        text: Predicted text string
        confidence: Confidence score
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        sequence = sequence.unsqueeze(0).to(device)  # (1, seq_len, 3)
        seq_len = torch.tensor([sequence.shape[1]], dtype=torch.long).to(device)
        
        # Forward pass
        logits = model(sequence, seq_len)  # (seq_len, 1, num_classes)
        
        # Decode
        text = decoder.greedy_decode(logits, seq_len)[0]
        
        # Get confidence
        confidence = get_confidence(logits)[0].item()
    
    return text, confidence


if __name__ == "__main__":
    # Test model loading
    model_path = Path("./models/online/checkpoints/best_online.pth")
    if model_path.exists():
        model, decoder, alphabet = load_online_model(model_path)
        print(f"Loaded model with alphabet size: {len(alphabet)}")
        print("✅ Model loading test passed!")
    else:
        print("Model not found. Train first.")