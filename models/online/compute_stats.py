import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.online.dataset import OnlineHandwritingDataset

def compute_global_stats():
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "online" / "isgl"
    
    # Load training data
    train_dataset = OnlineHandwritingDataset(data_dir, 'train', max_seq_len=None, dataset_name='isgl')

    
    # Collect all features
    all_features = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        sequence = sample['sequence']  # (seq_len, 6)
        all_features.append(sequence)
    
    # Concatenate all sequences
    all_features = torch.cat(all_features, dim=0)  # (total_points, 6)
    
    # After all_features = torch.cat(all_features, dim=0)
    print("Feature shapes:", all_features.shape)
    print("Feature 0 (dx) stats: mean={:.4f}, std={:.4f}".format(all_features[:,0].mean(), all_features[:,0].std()))
    print("Feature 1 (dy) stats: mean={:.4f}, std={:.4f}".format(all_features[:,1].mean(), all_features[:,1].std()))
    print("Feature 2 (log_r) stats: mean={:.4f}, std={:.4f}".format(all_features[:,2].mean(), all_features[:,2].std()))
    print("Feature 3 (angle) stats: mean={:.4f}, std={:.4f}".format(all_features[:,3].mean(), all_features[:,3].std()))
    
    # Compute mean and std for each feature (ignore pen flags 4,5)
    mean = all_features[:, :4].mean(dim=0)
    std = all_features[:, :4].std(dim=0)
    
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    # Save to file
    save_path = Path(__file__).parent / "feature_stats.pt"
    torch.save({'mean': mean, 'std': std}, save_path)
    print(f"✅ Saved stats to {save_path}")

if __name__ == "__main__":
    compute_global_stats()