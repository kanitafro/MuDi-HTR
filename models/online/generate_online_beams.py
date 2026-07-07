"""
generate_online_beams.py
Generates CTC beam search hypotheses for the ISGL test set.
Outputs a JSON file with top-K hypotheses and scores for each sample.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.online.model import OnlineHTRModel, CTCDecoder
from models.online.dataset import OnlineHandwritingDataset, CTCLabelEncoder
from scripts.train_online import load_config, compute_output_lengths


def main():
    # ==================== CONFIG ====================
    # Paths
    repo_root = Path(__file__).parent.parent.parent  # /home/jovyan/mudi
    config_path = Path(__file__).parent / "config_isgl.yaml"
    checkpoint_path = repo_root / "models" / "online" / "checkpoints_final" / "isgl" / "best_isgl_final.pth"
    output_json_path = Path(__file__).parent / "online_beam_results.json"
    
    # Beam search parameters
    BEAM_WIDTH = 30          # number of paths to keep at each step
    TOP_K = 10               # number of final hypotheses per sample
    
    # ==================== LOAD CONFIG ====================
    config = load_config(config_path)['iam']
    alphabet = list(config['alphabet'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== LOAD DATASET ====================
    data_dir = repo_root / "data" / "processed" / "online" / "isgl"
    max_seq_len = config['training']['max_seq_len']
    
    print("Loading test dataset...")
    test_dataset = OnlineHandwritingDataset(data_dir, 'test', max_seq_len, dataset_name='isgl')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # batch size 1 for easier key handling
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True
    )
    print(f"Test set size: {len(test_dataset)}")
    
    # ==================== LOAD MODEL ====================
    print("Loading model...")
    model = OnlineHTRModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']  # dropout doesn't matter for eval
    ).to(device)
    
    # Load best checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")
    print(f"   Best CER: {checkpoint.get('val_cer', 'N/A')} at epoch {checkpoint.get('epoch', 'N/A')}")
    
    # ==================== DECODER ====================
    decoder = CTCDecoder(alphabet, blank_idx=0)
    label_encoder = CTCLabelEncoder(alphabet)  # not needed for decoding, but available
    
    # ==================== GENERATE BEAMS ====================
    print(f"\n🔍 Generating beam search predictions (beam_width={BEAM_WIDTH}, top_k={TOP_K})...")
    
    results = {}  # key -> list of (hypothesis, score)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing test samples"):
            sequences = batch['sequences'].to(device)
            lengths = batch['lengths'].to(device)
            keys = batch['keys']
            
            # Forward pass
            logits = model(sequences, lengths)
            
            # Compute output lengths (downsample_factor=1 for pure LSTM)
            output_lengths = compute_output_lengths(lengths, downsample_factor=1)
            max_out_len = logits.size(0)
            output_lengths = torch.clamp(output_lengths, max=max_out_len)
            
            # Beam search
            beams = decoder.beam_search(logits, output_lengths, beam_width=BEAM_WIDTH, top_k=TOP_K)
            
            # Store results
            for key, beam_list in zip(keys, beams):
                # beam_list is list of (hypothesis_string, score)
                results[key] = beam_list
    
    # ==================== SAVE RESULTS ====================
    print(f"\n💾 Saving results to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total samples processed: {len(results)}")
    print(f"Top hypotheses per sample: {TOP_K}")
    print(f"Output file: {output_json_path}")
    
    # Show a few examples
    print("\n📝 Sample outputs:")
    sample_keys = list(results.keys())[:3]
    for key in sample_keys:
        print(f"\n  Key: {key}")
        for i, (hyp, score) in enumerate(results[key][:3]):
            print(f"    {i+1}: {hyp} (score: {score:.4f})")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()