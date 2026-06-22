# models/online/pretrain.py
"""
Pretraining script for online handwriting recognition on DIDI dataset.
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.online.model import OnlineHTRModel, CTCDecoder
from models.online.dataset import OnlineHandwritingDataset, CTCLabelEncoder
from models.online.train import evaluate, train_one_epoch  # Reuse from train.py


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Load configuration
    config_path = Path(__file__).parent / "config_pretrain.yaml"
    config = load_config(config_path)['pretrain']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {config['data_dir']}")
    
    # Get project root
    script_dir = Path(__file__).parent  # models/online/
    repo_root = script_dir.parent.parent  # MuDi-HTR/
    
    # Create directories with absolute paths
    data_dir = repo_root / config['data_dir']
    checkpoint_dir = repo_root / config['paths']['checkpoint_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = repo_root / config['paths']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for data at: {data_dir}")
    
    # Define alphabet
    alphabet = list(config['alphabet'])
    print(f"Alphabet size (including blank at index 0): {len(alphabet)}")
    
    # Create dataset and dataloader
    data_dir = Path(config['data_dir'])
    max_seq_len = config['training']['max_seq_len']
    
    train_dataset = OnlineHandwritingDataset(data_dir, 'train', max_seq_len, dataset_name='didi')
    val_dataset = OnlineHandwritingDataset(data_dir, 'valid', max_seq_len, dataset_name='didi')
    test_dataset = OnlineHandwritingDataset(data_dir, 'test', max_seq_len, dataset_name='didi')
    
    # Use num_workers=0 for Windows, can increase on Linux/Mac
    num_workers = 4
    
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
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    factor=float(config['training']['scheduler_factor']),
                                                    patience=int(config['training']['scheduler_patience']))
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Training loop
    best_val_cer = float('inf')
    writer = SummaryWriter(log_dir)
    
    print(f"\n🚀 Starting pretraining on DIDI for {config['training']['epochs']} epochs...")
    
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
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('CER/val', val_cer, epoch)
        writer.add_scalar('WER/val', val_wer, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_model_path = checkpoint_dir / "best_pretrain.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_wer': val_wer,
                'config': config,
                'alphabet': alphabet
            }, best_model_path)
            print(f"✅ Saved best pretrained model with CER: {val_cer:.4f}")
    
    # Final evaluation on test set
    print("\n=== Final Test Evaluation ===")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_cer, test_wer = evaluate(model, test_loader, decoder, device, label_encoder)
    print(f"Test CER: {test_cer:.4f}")
    print(f"Test WER: {test_wer:.4f}")
    
    writer.close()
    print("\n✅ Pretraining complete!")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main()