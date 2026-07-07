"""
Training script for IAM-OnDB dataset (from scratch, no pretraining).
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
from models.online.visualize import TrainingVisualizer, generate_report_summary


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Load configuration
    config_path = Path(__file__).parent / "config_iam.yaml"
    config = load_config(config_path)['iam']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get project root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    # Build paths
    data_dir = repo_root / config['data_dir']
    checkpoint_dir = repo_root / config['paths']['checkpoint_dir']
    log_dir = repo_root / config['paths']['log_dir']

    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Alphabet
    alphabet = list(config['alphabet'])
    print(f"Alphabet size (including blank at index 0): {len(alphabet)}")

    # Load datasets
    max_seq_len = config['training']['max_seq_len']
    print("\nLoading IAM-OnDB datasets...")
    train_dataset = OnlineHandwritingDataset(data_dir, 'train', max_seq_len, dataset_name='iam_ondb')
    val_dataset = OnlineHandwritingDataset(data_dir, 'valid', max_seq_len, dataset_name='iam_ondb')
    test_dataset = OnlineHandwritingDataset(data_dir, 'test', max_seq_len, dataset_name='iam_ondb')

    num_workers = 0 if sys.platform == 'win32' else 4

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, collate_fn=train_dataset.collate_fn,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, collate_fn=val_dataset.collate_fn,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                             shuffle=False, collate_fn=test_dataset.collate_fn,
                             num_workers=num_workers, pin_memory=True)

    # Initialize model (from scratch)
    model = OnlineHTRModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Model initialized from scratch (no pretraining)")

    # Initialize label encoder and decoder
    label_encoder = CTCLabelEncoder(alphabet)
    decoder = CTCDecoder(alphabet, blank_idx=0)

    # Optimizer with higher initial learning rate
    optimizer = optim.Adam(model.parameters(),
                           lr=float(config['training']['learning_rate']),
                           weight_decay=float(config['training']['weight_decay']))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=float(config['training']['scheduler_factor']),
                                                     patience=int(config['training']['scheduler_patience']))

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Visualizer
    visualizer = TrainingVisualizer(Path("experiments/figures_iam"))
    writer = SummaryWriter(log_dir)

    best_val_cer = float('inf')
    print(f"\n🚀 Training IAM-OnDB from scratch for {config['training']['epochs']} epochs...")

    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, label_encoder, config)

        val_cer, val_wer = evaluate(model, val_loader, decoder, device, label_encoder)

        scheduler.step(val_cer)
        current_lr = optimizer.param_groups[0]['lr']

        visualizer.update(epoch, train_loss, val_cer, val_wer, current_lr)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('CER/val', val_cer, epoch)
        writer.add_scalar('WER/val', val_wer, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_model_path = checkpoint_dir / "best_iam.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_wer': val_wer,
                'config': config,
                'alphabet': alphabet
            }, best_model_path)
            print(f"✅ Saved best model with CER: {val_cer:.4f}")

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

    # Final test
    print("\n=== Final Test Evaluation ===")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_cer, test_wer = evaluate(model, test_loader, decoder, device, label_encoder)
    print(f"Test CER: {test_cer:.4f}")
    print(f"Test WER: {test_wer:.4f}")

    visualizer.plot_all(save=True, show=False)
    summary = generate_report_summary(visualizer.metrics)
    print(summary)

    writer.close()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()