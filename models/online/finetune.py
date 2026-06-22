"""
Fine-tuning script for online handwriting recognition on IAM-OnDB dataset.
Loads pretrained weights from DIDI and fine-tunes on IAM-OnDB.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.online.model import OnlineHTRModel, CTCDecoder
from models.online.dataset import OnlineHandwritingDataset, CTCLabelEncoder
from models.online.train import evaluate, train_one_epoch


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Load configuration
    config_path = Path(__file__).parent / "config_pretrain.yaml"
    full_config = load_config(config_path)
    config = full_config['finetune']
    pretrain_config = full_config['pretrain']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {config['data_dir']}")
    
    # Create directories
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define alphabet (same as pretraining)
    alphabet = list(pretrain_config['alphabet'])
    print(f"Alphabet size (including blank at index 0): {len(alphabet)}")
    
    # Create dataset and dataloader
    data_dir = Path(config['data_dir'])
    max_seq_len = config['training']['max_seq_len']
    
    train_dataset = OnlineHandwritingDataset(data_dir, 'train', max_seq_len, dataset_name='iam_ondb')
    val_dataset = OnlineHandwritingDataset(data_dir, 'valid', max_seq_len, dataset_name='iam_ondb')
    test_dataset = OnlineHandwritingDataset(data_dir, 'test', max_seq_len, dataset_name='iam_ondb')
    
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
    
    # Initialize model (same architecture as pretraining)
    model = OnlineHTRModel(
        input_size=pretrain_config['model']['input_size'],
        hidden_size=pretrain_config['model']['hidden_size'],
        num_layers=pretrain_config['model']['num_layers'],
        num_classes=pretrain_config['model']['num_classes'],
        dropout=pretrain_config['model']['dropout']
    ).to(device)
    
    # Load pretrained weights
    pretrained_path = Path(config['paths']['pretrained_model'])
    if pretrained_path.exists():
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded pretrained model from epoch {checkpoint['epoch']}")
        print(f"   Pretrained CER: {checkpoint['val_cer']:.4f}")
    else:
        print(f"⚠️  Pretrained model not found at: {pretrained_path}")
        print("   Starting from scratch...")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize label encoder and decoder
    label_encoder = CTCLabelEncoder(alphabet)
    decoder = CTCDecoder(alphabet, blank_idx=0)
    
    # Use lower learning rate for fine-tuning
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
    
    print(f"\n🚀 Starting fine-tuning on IAM-OnDB for {config['training']['epochs']} epochs...")
    
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
            best_model_path = checkpoint_dir / "best_finetune.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_wer': val_wer,
                'config': config,
                'alphabet': alphabet
            }, best_model_path)
            print(f"✅ Saved best fine-tuned model with CER: {val_cer:.4f}")
    
    # Final evaluation on test set
    print("\n=== Final Test Evaluation ===")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_cer, test_wer = evaluate(model, test_loader, decoder, device, label_encoder)
    print(f"Test CER: {test_cer:.4f}")
    print(f"Test WER: {test_wer:.4f}")
    
    writer.close()
    print("\n✅ Fine-tuning complete!")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main()