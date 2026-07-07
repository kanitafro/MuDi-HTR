"""
Train transformer model on IAM-OnDB with curriculum learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from models.online.model_transformer import TransformerHTRModel
from models.online.dataset import OnlineHandwritingDataset, CTCLabelEncoder
from models.online.model import CTCDecoder
from models.online.visualize import TrainingVisualizer, generate_report_summary


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_output_lengths(input_lengths, factor=4):
    return torch.clamp(torch.ceil(input_lengths.float() / factor).long(), min=1)


def train_one_epoch(model, dataloader, optimizer, criterion, device, label_encoder, config, factor=4):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        seqs = batch['sequences'].to(device)
        lens = batch['lengths'].to(device)
        texts = batch['texts']

        labels, label_lens = label_encoder.collate_labels(texts)
        labels = labels.to(device)
        label_lens = label_lens.to(device)

        logits = model(seqs, lens)
        out_lens = compute_output_lengths(lens, factor)
        max_out = logits.size(0)
        out_lens = torch.clamp(out_lens, max=max_out)

        log_probs = torch.log_softmax(logits, dim=-1)
        loss = criterion(log_probs, labels, out_lens, label_lens)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, decoder, device, label_encoder, factor=4):
    model.eval()
    total_cer, total_wer, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            seqs = batch['sequences'].to(device)
            lens = batch['lengths'].to(device)
            texts = batch['texts']

            logits = model(seqs, lens)
            out_lens = compute_output_lengths(lens, factor)
            max_out = logits.size(0)
            out_lens = torch.clamp(out_lens, max=max_out)

            preds = decoder.greedy_decode(logits, out_lens)
            for p, t in zip(preds, texts):
                cer = compute_cer(p, t)
                wer = compute_wer(p, t)
                total_cer += cer
                total_wer += wer
                n += 1
    return total_cer / n, total_wer / n


def compute_cer(p, t):
    if len(t) == 0:
        return 0.0 if len(p) == 0 else 1.0
    m, n = len(p), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            if p[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1)
    return dp[m][n] / len(t)


def compute_wer(p, t):
    pw, tw = p.split(), t.split()
    if len(tw) == 0:
        return 0.0 if len(pw) == 0 else 1.0
    m, n = len(pw), len(tw)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            if pw[i-1] == tw[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1)
    return dp[m][n] / len(tw)


def main():
    config_path = Path(__file__).parent / "config_iam.yaml"
    config = load_config(config_path)['iam']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    repo_root = Path(__file__).parent.parent.parent
    data_dir = repo_root / config['data_dir']
    checkpoint_dir = repo_root / config['paths']['checkpoint_dir'] / "transformer"
    log_dir = repo_root / config['paths']['log_dir'] / "transformer"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    alphabet = list(config['alphabet'])
    print(f"Alphabet size: {len(alphabet)}")

    # Load full datasets
    max_seq_len = config['training']['max_seq_len']
    train_full = OnlineHandwritingDataset(data_dir, 'train', max_seq_len, dataset_name='iam_ondb')
    val_dataset = OnlineHandwritingDataset(data_dir, 'valid', max_seq_len, dataset_name='iam_ondb')
    test_dataset = OnlineHandwritingDataset(data_dir, 'test', max_seq_len, dataset_name='iam_ondb')

    # Curriculum lengths: start small, increase
    curriculum_lengths = [500, 1000, 1500, 2000, 3000, 5000]
    current_max_len = curriculum_lengths[0]

    # Dataloader with dynamic filtering
    def filter_by_len(dataset, max_len):
        indices = [i for i, sample in enumerate(dataset.data) if sum(len(s) for s in sample['strokes']) <= max_len]
        return Subset(dataset, indices)

    num_workers = 0 if sys.platform == 'win32' else 4

    model = TransformerHTRModel(
        input_size=config['model']['input_size'],
        d_model=512,          # larger
        nhead=8,
        num_layers=6,
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        max_len=5000
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    label_encoder = CTCLabelEncoder(alphabet)
    decoder = CTCDecoder(alphabet, blank_idx=0)

    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    visualizer = TrainingVisualizer(Path("experiments/figures_transformer"))
    writer = SummaryWriter(log_dir)

    best_val_cer = float('inf')
    total_epochs = 60
    factor = 4  # CNN downsampling factor

    print(f"\n🚀 Starting Transformer training with curriculum learning...")

    for epoch in range(1, total_epochs+1):
        # Advance curriculum every 10 epochs
        idx = min(epoch // 10, len(curriculum_lengths)-1)
        current_max_len = curriculum_lengths[idx]
        print(f"\nEpoch {epoch}/{total_epochs} - Max sequence length: {current_max_len}")

        # Build subset
        train_subset = filter_by_len(train_full, current_max_len)
        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'],
                                  shuffle=True, collate_fn=train_full.collate_fn,
                                  num_workers=num_workers, pin_memory=True)

        # Validation loader uses full validation set (no filtering)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                                shuffle=False, collate_fn=val_dataset.collate_fn,
                                num_workers=num_workers, pin_memory=True)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, label_encoder, config, factor)

        # Validate
        val_cer, val_wer = evaluate(model, val_loader, decoder, device, label_encoder, factor)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f}")
        print(f"LR: {current_lr:.6f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('CER/val', val_cer, epoch)
        writer.add_scalar('WER/val', val_wer, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('Curriculum_len', current_max_len, epoch)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(model.state_dict(), checkpoint_dir / "best_transformer.pth")
            print(f"✅ Saved best model with CER: {val_cer:.4f}")

    # Final test
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                             shuffle=False, collate_fn=test_dataset.collate_fn,
                             num_workers=num_workers, pin_memory=True)
    model.load_state_dict(torch.load(checkpoint_dir / "best_transformer.pth", weights_only=False))
    test_cer, test_wer = evaluate(model, test_loader, decoder, device, label_encoder, factor)
    print(f"\nTest CER: {test_cer:.4f}, Test WER: {test_wer:.4f}")

    writer.close()
    print("✅ Training complete.")


if __name__ == "__main__":
    main()