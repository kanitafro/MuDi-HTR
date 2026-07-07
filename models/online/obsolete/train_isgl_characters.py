"""
Training script for ISGL dataset - CHARACTERS ONLY (single letters/digits).
Filters out words (text length > 1) and trains on individual characters.
Uses the improved CNN-BiLSTM model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from models.online.model_improved import ImprovedOnlineHTRModel
from models.online.dataset import OnlineHandwritingDataset, CTCLabelEncoder
from models.online.model import CTCDecoder
from models.online.visualize import TrainingVisualizer, generate_report_summary


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def filter_characters(dataset):
    """Return indices of samples where text length == 1 (single character)."""
    indices = []
    for i in range(len(dataset)):
        sample = dataset.data[i]  # Access raw data
        text = sample.get('text', '')
        if len(text) == 1:
            indices.append(i)
    return indices


def compute_cer(p, t):
    if len(t) == 0:
        return 0.0 if len(p) == 0 else 1.0
    m, n = len(p), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
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
    for i in range(1, m+1):
        for j in range(1, n+1):
            if pw[i-1] == tw[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1)
    return dp[m][n] / len(tw)


def compute_output_lengths(input_lengths, factor=4):
    return torch.clamp(torch.ceil(input_lengths.float() / factor).long(), min=1)


def main():
    config_path = Path(__file__).parent / "config_isgl.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / "config_iam.yaml"
    config = load_config(config_path)['iam']

    repo_root = Path(__file__).parent.parent.parent
    data_dir = repo_root / "data" / "processed" / "online" / "isgl"
    checkpoint_dir = repo_root / "models" / "online" / "checkpoints" / "isgl_characters"
    log_dir = repo_root / "runs" / "isgl_characters"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    alphabet = list(config['alphabet'])
    print(f"Alphabet size: {len(alphabet)}")

    max_seq_len = config['training']['max_seq_len']
    print("\nLoading ISGL datasets...")
    full_train = OnlineHandwritingDataset(data_dir, 'train', max_seq_len, dataset_name='isgl')
    full_val = OnlineHandwritingDataset(data_dir, 'valid', max_seq_len, dataset_name='isgl')
    full_test = OnlineHandwritingDataset(data_dir, 'test', max_seq_len, dataset_name='isgl')

    # Filter to characters only (text length == 1)
    train_indices = filter_characters(full_train)
    val_indices = filter_characters(full_val)
    test_indices = filter_characters(full_test)

    print(f"Train: {len(train_indices)} characters out of {len(full_train)}")
    print(f"Val: {len(val_indices)} characters out of {len(full_val)}")
    print(f"Test: {len(test_indices)} characters out of {len(full_test)}")

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)
    test_dataset = Subset(full_test, test_indices)

    num_workers = 0 if sys.platform == 'win32' else 4

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, collate_fn=full_train.collate_fn,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, collate_fn=full_val.collate_fn,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                             shuffle=False, collate_fn=full_test.collate_fn,
                             num_workers=num_workers, pin_memory=True)

    # Use the IMPROVED model (CNN-BiLSTM) – it performed slightly better
    model = ImprovedOnlineHTRModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Improved CNN-BiLSTM model initialized (characters only)")

    label_encoder = CTCLabelEncoder(alphabet)
    decoder = CTCDecoder(alphabet, blank_idx=0)

    optimizer = optim.Adam(model.parameters(),
                           lr=float(config['training']['learning_rate']),
                           weight_decay=float(config['training']['weight_decay']))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=float(config['training']['scheduler_factor']),
                                                     patience=int(config['training']['scheduler_patience']))

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    visualizer = TrainingVisualizer(Path("experiments/figures_isgl_characters"))
    writer = SummaryWriter(log_dir)

    best_val_cer = float('inf')
    print(f"\n🚀 Training ISGL (characters only) for {config['training']['epochs']} epochs...")

    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")

        # Training
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            seqs = batch['sequences'].to(device)
            lens = batch['lengths'].to(device)
            texts = batch['texts']

            labels, label_lens = label_encoder.collate_labels(texts)
            labels = labels.to(device)
            label_lens = label_lens.to(device)

            logits = model(seqs, lens)
            out_lens = compute_output_lengths(lens, factor=4)
            max_out = logits.size(0)
            out_lens = torch.clamp(out_lens, max=max_out)

            log_probs = torch.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, labels, out_lens, label_lens)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_cer, total_wer, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                seqs = batch['sequences'].to(device)
                lens = batch['lengths'].to(device)
                texts = batch['texts']

                logits = model(seqs, lens)
                out_lens = compute_output_lengths(lens, factor=4)
                max_out = logits.size(0)
                out_lens = torch.clamp(out_lens, max=max_out)

                preds = decoder.greedy_decode(logits, out_lens)
                for p, t in zip(preds, texts):
                    cer = compute_cer(p, t)
                    wer = compute_wer(p, t)
                    total_cer += cer
                    total_wer += wer
                    n += 1

        val_cer = total_cer / n if n > 0 else 1.0
        val_wer = total_wer / n if n > 0 else 1.0

        scheduler.step(val_cer)
        current_lr = optimizer.param_groups[0]['lr']

        visualizer.update(epoch, train_loss, val_cer, val_wer, current_lr)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f}")
        print(f"LR: {current_lr:.6f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('CER/val', val_cer, epoch)
        writer.add_scalar('WER/val', val_wer, epoch)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_model_path = checkpoint_dir / "best_isgl_characters.pth"
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
    print("\n=== Final Test ===")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    total_cer, total_wer, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            seqs = batch['sequences'].to(device)
            lens = batch['lengths'].to(device)
            texts = batch['texts']
            logits = model(seqs, lens)
            out_lens = compute_output_lengths(lens, factor=4)
            max_out = logits.size(0)
            out_lens = torch.clamp(out_lens, max=max_out)
            preds = decoder.greedy_decode(logits, out_lens)
            for p, t in zip(preds, texts):
                cer = compute_cer(p, t)
                wer = compute_wer(p, t)
                total_cer += cer
                total_wer += wer
                n += 1
    test_cer = total_cer / n if n > 0 else 1.0
    test_wer = total_wer / n if n > 0 else 1.0
    print(f"Test CER: {test_cer:.4f}")
    print(f"Test WER: {test_wer:.4f}")

    visualizer.plot_all(save=True, show=False)
    writer.close()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()