# models/online/train_isgl.py
"""
Training script for ISGL dataset (online handwriting recognition).
Uses the improved CNN-BiLSTM architecture with temporal downsampling.
Handles short sequences correctly.
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

from models.online.model import OnlineHTRModel as ImprovedOnlineHTRModel
from models.online.dataset import OnlineHandwritingDataset, CTCLabelEncoder
from models.online.model import CTCDecoder
from models.online.visualize import TrainingVisualizer, generate_report_summary


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


#def compute_output_lengths(input_lengths, downsample_factor=4):
#    """Compute output sequence lengths after CNN downsampling."""
#    output_lens = torch.ceil(input_lengths.float() / downsample_factor).long()
#    output_lens = torch.clamp(output_lens, min=1)
#    return output_lens

def compute_output_lengths(input_lengths, downsample_factor=1):
    """No downsampling – output length = input length."""
    return input_lengths.clone()


def train_one_epoch(model, dataloader, optimizer, criterion, device, label_encoder, config):
    """Custom training loop with correct output lengths."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequences'].to(device)
        lengths = batch['lengths'].to(device)          # original input lengths
        texts = batch['texts']

        # Encode labels
        labels, label_lengths = label_encoder.collate_labels(texts)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        # Forward pass – model returns logits (seq_len, batch, classes)
        logits = model(sequences, lengths)

        # Compute output lengths after downsampling (factor 4)
        output_lengths = compute_output_lengths(lengths, downsample_factor=1)
        # Clamp to actual output sequence length (logits.size(0))
        max_out_len = logits.size(0)
        output_lengths = torch.clamp(output_lengths, max=max_out_len)

        log_probs = torch.log_softmax(logits, dim=-1)
        loss = criterion(log_probs, labels, output_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, decoder, device, label_encoder):
    """Custom evaluation with correct output lengths."""
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            sequences = batch['sequences'].to(device)
            lengths = batch['lengths'].to(device)
            texts = batch['texts']

            logits = model(sequences, lengths)

            # Compute output lengths
            output_lengths = compute_output_lengths(lengths, downsample_factor=1)
            max_out_len = logits.size(0)
            output_lengths = torch.clamp(output_lengths, max=max_out_len)

            decoded = decoder.greedy_decode(logits, output_lengths)

            for pred, true in zip(decoded, texts):
                cer = compute_cer(pred, true)
                wer = compute_wer(pred, true)
                total_cer += cer
                total_wer += wer
                num_samples += 1

    return total_cer / num_samples, total_wer / num_samples


# Simple CER/WER helpers (copied from train.py)
def compute_cer(pred, true):
    if len(true) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    m, n = len(pred), len(true)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == true[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1, dp[i-1][j] + 1, dp[i][j-1] + 1)
    return dp[m][n] / len(true)


def compute_wer(pred, true):
    pred_words = pred.split()
    true_words = true.split()
    if len(true_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
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


def main():
    # Load configuration
    config_path = Path(__file__).parent / "config_isgl.yaml"
    #if not config_path.exists():
    #    config_path = Path(__file__).parent / "config_iam.yaml"
    config = load_config(config_path)['iam']

    # Override data directory for ISGL
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    # Use ISGL data
    data_dir = repo_root / "data" / "processed" / "online" / "isgl"
    checkpoint_dir = repo_root / "models" / "online" / "checkpoints_final" / "isgl"
    log_dir = repo_root / "runs" / "isgl"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Alphabet
    alphabet = list(config['alphabet'])
    print(f"Alphabet size (including blank at index 0): {len(alphabet)}")

    # Load datasets
    max_seq_len = config['training']['max_seq_len']
    print("\nLoading ISGL datasets...")
    train_dataset = OnlineHandwritingDataset(data_dir, 'train', max_seq_len, dataset_name='isgl')
    val_dataset = OnlineHandwritingDataset(data_dir, 'valid', max_seq_len, dataset_name='isgl')
    test_dataset = OnlineHandwritingDataset(data_dir, 'test', max_seq_len, dataset_name='isgl')

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

    # Check for missing characters
    all_chars = set()
    for i in range(len(train_dataset)):
        all_chars.update(train_dataset[i]['text'])
    print("All characters in training labels:", sorted(all_chars))
    print("Alphabet characters:", list(config['alphabet']))
    missing = all_chars - set(config['alphabet'])
    if missing:
        print(f"❌ Characters missing from alphabet: {missing}")
    else:
        print("✅ All characters are in alphabet")

    """
    #--- OVERFIT TEST ---
    tiny_train = torch.utils.data.Subset(train_dataset, range(10))
    tiny_loader = DataLoader(tiny_train, batch_size=10, shuffle=True, 
                             collate_fn=train_dataset.collate_fn)

    print("\n🔍 Running overfit test on 10 samples...")

    model = ImprovedOnlineHTRModel(
            input_size=config['model']['input_size'],
            hidden_size=512,
            num_layers=3,
            num_classes=config['model']['num_classes'],
            dropout=0.0
        ).to(device)
    label_encoder = CTCLabelEncoder(alphabet)
    decoder = CTCDecoder(alphabet, blank_idx=0)
    model.train()
    # Override dropout to 0.0 (temporarily)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    
    # Use a much lower learning rate
    #optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = None
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    for epoch in range(200):  # 50 epochs
        model.train()
        for batch in tiny_loader:
            sequences = batch['sequences'].to(device)
            lengths = batch['lengths'].to(device)
            texts = batch['texts']
            labels, label_lengths = label_encoder.collate_labels(texts)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            logits = model(sequences, lengths)
            output_lengths = compute_output_lengths(lengths, downsample_factor=1)
            max_out = logits.size(0)
            output_lengths = torch.clamp(output_lengths, max=max_out)

            
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, labels, output_lengths, label_lengths)
            
            print(f"logits shape: {logits.shape}")  # (seq_len, batch, num_classes)
            print(f"Loss: {loss.item():.4f}")
            print(f"log_probs shape: {log_probs.shape}   |    labels shape: {labels.shape}")  # (seq_len, batch, num_classes)   |   (total_label_length,)
            print(f"output_lengths: {output_lengths}")
            print(f"label_lengths: {label_lengths}")
            print(f"max label length: {label_lengths.max().item()}    |    max output length: {output_lengths.max().item()}")
            print("Ground truth texts:", texts)
            print("Label lengths:", label_lengths)
            print()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        # Decode predictions after each epoch
        model.eval()
        with torch.no_grad():
            logits = model(sequences, lengths)
            decoded = decoder.greedy_decode(logits, output_lengths)
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Preds={decoded}")
            if all(p == t for p, t in zip(decoded, texts)):
                print("✅ Perfect overfit achieved!")
                break
    # --- OVERFIT TEST OVER---
    """
    # Initialize the improved model
    model = ImprovedOnlineHTRModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Improved model initialized from scratch")

    # Initialize label encoder and decoder
    label_encoder = CTCLabelEncoder(alphabet)
    decoder = CTCDecoder(alphabet, blank_idx=0)

    # Optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=float(config['training']['learning_rate']),
                           weight_decay=float(config['training']['weight_decay']))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=float(config['training']['scheduler_factor']),
                                                     patience=int(config['training']['scheduler_patience']))

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Visualizer
    visualizer = TrainingVisualizer(Path("experiments_final/figures_isgl_final")) ############## HERE ##################
    writer = SummaryWriter(log_dir)

    best_val_cer = float('inf')
    print(f"\n🚀 Training ISGL from scratch for {config['training']['epochs']} epochs...")

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
            best_model_path = checkpoint_dir / "best_isgl_final.pth" ############## HERE ##################
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
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main()
