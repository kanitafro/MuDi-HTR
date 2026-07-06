import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.online.model import OnlineHTRModel, CTCDecoder
from models.online.dataset import OnlineHandwritingDataset, CTCLabelEncoder
from models.online.train_isgl import load_config, compute_output_lengths
from torch.utils.data import DataLoader

# Load config and model
config = load_config(Path(__file__).parent / "config_isgl.yaml")['iam']
alphabet = list(config['alphabet'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = OnlineHTRModel(
    input_size=config['model']['input_size'],
    hidden_size=config['model']['hidden_size'],
    num_layers=config['model']['num_layers'],
    num_classes=config['model']['num_classes'],
    dropout=config['model']['dropout']
).to(device)

# Load best checkpoint
checkpoint = torch.load("/home/jovyan/mudi/models/online/checkpoints/isgl/best_isgl_final.pth", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load a test sample
data_dir = Path("/home/jovyan/mudi/data/processed/online/isgl")
test_dataset = OnlineHandwritingDataset(data_dir, 'valid', max_seq_len=500, dataset_name='isgl')
loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)

# Get one batch
batch = next(iter(loader))
sequences = batch['sequences'].to(device)
lengths = batch['lengths'].to(device)

# Forward pass
logits = model(sequences, lengths)

# Decode with greedy (baseline)
decoder = CTCDecoder(alphabet, blank_idx=0)
greedy_pred = decoder.greedy_decode(logits, lengths)
print("Greedy:", greedy_pred[0])

# Decode with beam search (top 5)
beam_preds = decoder.beam_search(logits, lengths, beam_width=20, top_k=5)
print("\nBeam search (top 5):")
for i, (text, score) in enumerate(beam_preds[0]):
    print(f"  {i+1}: {text} (score: {score:.4f})")