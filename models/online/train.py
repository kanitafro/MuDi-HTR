# models/online/train.py
"""
Training script for online handwriting recognition model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import yaml
import sys
from tqdm import tqdm
import wandb

# pip install torch numpy pyyaml tqdm wandb pathlib matplotlib seaborn tensorboard scikit-learn wandb

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.online.model import OnlineHTRModel, CTCDecoder
from models.online.dataset import OnlineHandwritingDataset, CTCLabelEncoder
from models.online.visualize import TrainingVisualizer, generate_report_summary


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_cer(pred_text, true_text):
    """
    Compute Character Error Rate (CER).
    
    Args:
        pred_text: Predicted string
        true_text: Ground truth string
    
    Returns:
        CER as float
    """
    if len(true_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0
    
    # Simple Levenshtein distance
    m, n = len(pred_text), len(true_text)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_text[i-1] == true_text[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1, dp[i-1][j] + 1, dp[i][j-1] + 1)
    
    return dp[m][n] / len(true_text)


def compute_wer(pred_text, true_text):
    """
    Compute Word Error Rate (WER).
    Simple word-level split and comparison.
    """
    pred_words = pred_text.split()
    true_words = true_text.split()
    
    if len(true_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    # Use Levenshtein distance at word level
    m, n = len(pred_words), len(true_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == true_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1, dp[i-1][j] + 1, dp[i][j-1] + 1)
    
    return dp[m][n] / len(true_words)


def evaluate(model, dataloader, decoder, device, label_encoder):
    """Evaluate model on validation/test set."""
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0
    
    # Debug: print first few samples
    print("\n🔍 DEBUG: Checking first 3 predictions...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            sequences = batch['sequences'].to(device)
            lengths = batch['lengths'].to(device)
            texts = batch['texts']
            
            # Forward pass
            logits = model(sequences, lengths)
            
            # Greedy decode
            decoded_texts = decoder.greedy_decode(logits, lengths)
            
            # Debug: print first batch
            if batch_idx == 0:
                for i in range(min(3, len(decoded_texts))):
                    print(f"  Sample {i+1}:")
                    print(f"    Ground Truth: '{texts[i]}'")
                    print(f"    Predicted:    '{decoded_texts[i]}'")
                    print(f"    GT length: {len(texts[i])}, Pred length: {len(decoded_texts[i])}")
            
            # Compute metrics
            for pred, true in zip(decoded_texts, texts):
                cer = compute_cer(pred, true)
                wer = compute_wer(pred, true)
                total_cer += cer
                total_wer += wer
                num_samples += 1
    
    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples
    
    print(f"\n📊 Evaluation complete: {num_samples} samples")
    print(f"   Avg CER: {avg_cer:.4f}, Avg WER: {avg_wer:.4f}")
    
    return avg_cer, avg_wer


def train_one_epoch(model, dataloader, optimizer, criterion, device, label_encoder, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequences'].to(device)
        lengths = batch['lengths'].to(device)
        texts = batch['texts']
        
        # Encode labels
        labels, label_lengths = label_encoder.collate_labels(texts)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward pass
        logits = model(sequences, lengths)  # (seq_len, batch, num_classes)
        
        # CTC loss expects log_probs
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Compute CTC loss
        loss = criterion(log_probs, labels, lengths, label_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    visualizer = TrainingVisualizer(Path("./experiments/figures"))
    
    # Initialize wandb (optional)
    # wandb.init(project="mudi-htr", config=config)
    
    # Define alphabet
    alphabet = list(config['data']['alphabet'])
    print(f"Alphabet size (including blank at index 0): {len(alphabet)}")
    
    # Create dataset and dataloader
    data_dir = Path(config['data']['data_dir'])
    max_seq_len = config['training']['max_seq_len']
    
    train_dataset = OnlineHandwritingDataset(data_dir, 'train', max_seq_len)
    val_dataset = OnlineHandwritingDataset(data_dir, 'valid', max_seq_len)
    test_dataset = OnlineHandwritingDataset(data_dir, 'test', max_seq_len)
    
    # Use num_workers=0 for Windows to avoid multiprocessing issues
    num_workers = 0  # You can change to 4 on Linux/Mac
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                            shuffle=True, collate_fn=train_dataset.collate_fn,
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                          shuffle=False, collate_fn=val_dataset.collate_fn,
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                           shuffle=False, collate_fn=test_dataset.collate_fn,
                           num_workers=num_workers, pin_memory=True)
    
    # Initialize model
    model = OnlineHTRModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize label encoder and decoder
    label_encoder = CTCLabelEncoder(alphabet)
    decoder = CTCDecoder(alphabet, blank_idx=0)
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), 
                        lr=float(config['training']['learning_rate']),
                        weight_decay=float(config['training']['weight_decay']))

    # Also fix any other numeric conversions if needed
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    factor=float(config['training']['scheduler_factor']),
                                                    patience=int(config['training']['scheduler_patience']))
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Training loop
    best_val_cer = float('inf')
    writer = SummaryWriter(log_dir)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, 
                                criterion, device, label_encoder, config)
        
        # Evaluate on validation
        val_cer, val_wer = evaluate(model, val_loader, decoder, device, label_encoder)
        
        # Update learning rate
        scheduler.step(val_cer)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update visualizer
        visualizer.update(epoch, train_loss, val_cer, val_wer, current_lr)
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('CER/val', val_cer, epoch)
        writer.add_scalar('WER/val', val_wer, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # wandb.log({'train_loss': train_loss, 'val_cer': val_cer, 
        #           'val_wer': val_wer, 'lr': current_lr})
        
        # Save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_model_path = checkpoint_dir / "best_online.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_wer': val_wer,
                'config': config,
                'alphabet': alphabet
            }, best_model_path)
            print(f"Saved best model with CER: {val_cer:.4f}")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_wer': val_wer,
                'config': config,
                'alphabet': alphabet
            }, checkpoint_path)
    
    # Final evaluation on test set
    print("\n=== Final Test Evaluation ===")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    test_cer, test_wer = evaluate(model, test_loader, decoder, device, label_encoder)
    print(f"Test CER: {test_cer:.4f}")
    print(f"Test WER: {test_wer:.4f}")

    # Generate all plots
    print("\n📊 Generating visualization plots...")
    visualizer.plot_all(save=True, show=False)

    # Print summary
    summary = generate_report_summary(visualizer.metrics)
    print(summary)

    # Save summary to file
    summary_path = Path("./experiments/figures/training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"✅ Saved summary to {summary_path}")

    writer.close()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()