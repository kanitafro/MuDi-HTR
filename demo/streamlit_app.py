import sys
from pathlib import Path
from dataclasses import dataclass

# ---- 1. Set project root ----
project_root = Path(__file__).resolve().parent.parent   # points to MuDi-HTR/
sys.path.insert(0, str(project_root))
print("Project root:", project_root)

# ---- 2. Imports ----
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from streamlit_drawable_canvas import st_canvas
import json
import re
from PIL import Image, ImageOps

from models.online.model import OnlineHTRModel, CTCDecoder
from models.online.dataset import CTCLabelEncoder
from models.online.train_isgl import load_config, compute_output_lengths

try:
    from models.offline import CRNN
except ImportError:
    CRNN = None

# -------------------- Helper Functions --------------------
def logsumexp_pair(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    return float(np.logaddexp(a, b))

@dataclass(frozen=True)
class BranchVocabulary:
    tokens: list[str]
    blank_index: int
    unknown_index: int | None = None

    @property
    def actual_tokens(self) -> list[str]:
        if self.unknown_index is None:
            return self.tokens[1:]
        return self.tokens[self.unknown_index + 1 :]

def build_branch_vocabularies(online_alphabet: list[str], offline_vocab: list[str]):
    online_vocab = BranchVocabulary(tokens=online_alphabet, blank_index=0)
    offline_vocab_map = BranchVocabulary(tokens=offline_vocab, blank_index=0,
                                         unknown_index=1 if len(offline_vocab)>1 and offline_vocab[1].startswith("<UNK") else None)
    unified_tokens = []
    seen = set()
    for token in online_vocab.actual_tokens:
        if token not in seen:
            unified_tokens.append(token); seen.add(token)
    for token in offline_vocab_map.actual_tokens:
        if token not in seen:
            unified_tokens.append(token); seen.add(token)
    return online_vocab, offline_vocab_map, unified_tokens

def branch_probs_to_unified_probs(branch_probs: np.ndarray, branch_vocab: BranchVocabulary,
                                  unified_tokens: list[str]) -> np.ndarray:
    T, _ = branch_probs.shape
    unified = np.zeros((T, len(unified_tokens)+1), dtype=np.float64)
    unified[:,0] = branch_probs[:, branch_vocab.blank_index]
    if branch_vocab.unknown_index is not None and branch_vocab.unknown_index < branch_probs.shape[1]:
        unified[:,0] += branch_probs[:, branch_vocab.unknown_index]
    start = 1 if branch_vocab.unknown_index is None else branch_vocab.unknown_index+1
    token_to_unified = {tok:i+1 for i,tok in enumerate(unified_tokens)}
    for branch_off, token in enumerate(branch_vocab.tokens[start:], start=start):
        unif_idx = token_to_unified.get(token)
        if unif_idx is None:
            unified[:,0] += branch_probs[:, branch_off]
        else:
            unified[:, unif_idx] += branch_probs[:, branch_off]
    row_sums = unified.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    unified /= row_sums
    return unified

def resample_time_axis(probs: np.ndarray, target_len: int) -> np.ndarray:
    T, C = probs.shape
    if T == target_len:
        return probs.copy()
    if T == 1:
        return np.repeat(probs, target_len, axis=0)
    x_old = np.linspace(0,1,T)
    x_new = np.linspace(0,1,target_len)
    resampled = np.empty((target_len, C), dtype=np.float64)
    for c in range(C):
        resampled[:,c] = np.interp(x_new, x_old, probs[:,c])
    resampled = np.clip(resampled, 1e-12, None)
    resampled /= resampled.sum(axis=1, keepdims=True)
    return resampled

def ctc_prefix_beam_search(
    log_probs: np.ndarray,
    idx_to_char: list[str],
    beam_width: int = 50,
    blank_index: int = 0,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    time_steps, num_classes = log_probs.shape
    if num_classes != len(idx_to_char):
        raise ValueError(f"Vocabulary mismatch: got {num_classes} logits classes but {len(idx_to_char)} tokens.")

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -np.inf)}
    for t in range(time_steps):
        timestep = log_probs[t]
        blank_logp = float(timestep[blank_index])
        # Candidate pruning
        candidate_indices = np.argsort(timestep[1:])[-min(beam_width*2, num_classes-1):] + 1
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def update(prefix: tuple[int, ...], blank_score: float | None = None, nonblank_score: float | None = None) -> None:
            existing_blank, existing_nonblank = next_beams.get(prefix, (-np.inf, -np.inf))
            if blank_score is not None:
                existing_blank = logsumexp_pair(existing_blank, blank_score)
            if nonblank_score is not None:
                existing_nonblank = logsumexp_pair(existing_nonblank, nonblank_score)
            next_beams[prefix] = (existing_blank, existing_nonblank)

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = logsumexp_pair(p_blank, p_nonblank)
            update(prefix, blank_score=total + blank_logp)
            for class_idx in candidate_indices:
                class_logp = float(timestep[class_idx])
                if prefix and prefix[-1] == class_idx:
                    update(prefix, nonblank_score=p_nonblank + class_logp)
                    update(prefix + (class_idx,), nonblank_score=p_blank + class_logp)
                else:
                    update(prefix + (class_idx,), nonblank_score=total + class_logp)

        scored = sorted(next_beams.items(), key=lambda item: logsumexp_pair(item[1][0], item[1][1]), reverse=True)[:beam_width]
        beams = dict(scored)

    hypotheses = []
    for prefix, (p_blank, p_nonblank) in beams.items():
        score = logsumexp_pair(p_blank, p_nonblank)
        if len(prefix) > 0:
            score /= len(prefix)
        chars = []
        for idx in prefix:
            if idx != blank_index and idx < len(idx_to_char):
                chars.append(idx_to_char[idx])
        hypotheses.append((''.join(chars), score))

    hypotheses.sort(key=lambda x: x[1], reverse=True)
    return hypotheses[:top_k]

def max_softmax_confidence(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    return float(probs.max(dim=-1).values.mean().item())

def online_logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).squeeze(1).detach().cpu().numpy().astype(np.float64)

def offline_logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).squeeze(1).detach().cpu().numpy().astype(np.float64)

# -------------------- Load Config and Global Stats --------------------
CONFIG_PATH = project_root / "models" / "online" / "config_isgl.yaml"
STATS_PATH = project_root / "models" / "online" / "feature_stats.pt"

config = load_config(CONFIG_PATH)['iam']
alphabet = list(config['alphabet'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if STATS_PATH.exists():
    stats = torch.load(STATS_PATH, weights_only=False)
    global_mean = stats['mean'].to(device)
    global_std = stats['std'].to(device)
    st.write(f"Global mean: {global_mean.cpu().numpy()}")
    st.write(f"Global std: {global_std.cpu().numpy()}")
else:
    st.error("feature_stats.pt not found. Please run compute_stats.py first.")
    st.stop()

# -------------------- Load Models --------------------
@st.cache_resource
def load_online():
    model = OnlineHTRModel(input_size=config['model']['input_size'],
                           hidden_size=config['model']['hidden_size'],
                           num_layers=config['model']['num_layers'],
                           num_classes=config['model']['num_classes'],
                           dropout=0.0)
    ckpt = torch.load(ONLINE_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model

ONLINE_CKPT = project_root / "models" / "checkpoints" / "isgl" / "best_isgl_final.pth"
online_model = load_online()
decoder = CTCDecoder(alphabet, blank_idx=0)

@st.cache_resource
def load_offline():
    if not OFFLINE_CKPT.exists():
        return None, None, None
    ckpt = torch.load(OFFLINE_CKPT, map_location=device, weights_only=False)
    
    # Extract vocabulary
    vocab = ckpt.get('encoder_vocab', [])
    if not vocab:
        vocab = ckpt.get('config', {}).get('vocab', [])
    if not vocab:
        st.error("Offline checkpoint missing vocabulary.")
        return None, None, None

    # Get state dict (handle different checkpoint formats)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Determine num_classes from classifier weight
    classifier_weight = state_dict.get('classifier.weight')
    if classifier_weight is not None:
        num_classes = classifier_weight.shape[0]
    else:
        num_classes = len(vocab)

    # Infer hidden size from rnn weights
    rnn_weight = None
    for key in state_dict:
        if 'rnn.weight_ih_l0' in key:
            rnn_weight = state_dict[key]
            break
    if rnn_weight is not None:
        # For bidirectional LSTM, weight_ih_l0 shape is (4*hidden, input)
        hidden_size = rnn_weight.shape[0] // 4
    else:
        hidden_size = 256  # fallback

    # Initialize model
    model = CRNN(num_classes=num_classes, hidden_size=hidden_size).to(device)

    # Load with strict=False to handle naming mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        st.warning(f"Missing keys in offline checkpoint: {missing[:3]}...")
    if unexpected:
        st.warning(f"Unexpected keys in offline checkpoint: {unexpected[:3]}...")
    
    model.eval()
    return model, vocab, ckpt

OFFLINE_CKPT = project_root / "models" / "checkpoints" / "offline" / "pretrained.pth"
offline_model, offline_vocab, offline_ckpt = load_offline()

if offline_model is not None:
    online_vocab, offline_vocab_map, unified_tokens = build_branch_vocabularies(alphabet, offline_vocab)
    st.success("✅ Both online and offline models loaded - fusion available.")
else:
    st.info("ℹ️ Offline model not found – using online-only.")
    offline_vocab_map = None
    unified_tokens = []

# -------------------- Preprocessing: strokes to 6 features --------------------
def strokes_to_features(strokes, max_len=500, target_size=100):
    import re
    from scipy.ndimage import gaussian_filter1d  # you may need to install scipy

    stroke_raw = []
    for stroke in strokes:
        path = stroke.get('path', [])
        if not path:
            continue
        if isinstance(path, list):
            path_str = ' '.join(str(item) for item in path)
        else:
            path_str = str(path)
        numbers = re.findall(r'[-+]?\d*\.?\d+', path_str)
        if len(numbers) < 2:
            continue
        coords = []
        for i in range(0, len(numbers) - 1, 2):
            try:
                x = float(numbers[i])
                y = float(numbers[i+1])
                coords.append((x, y))
            except ValueError:
                continue
        if coords:
            stroke_raw.append(np.array(coords, dtype=np.float32))

    if not stroke_raw:
        return None

    # Step 1: Scale all coordinates so that max extent == target_size
    all_pts = np.vstack(stroke_raw)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]
    scale = max(width, height) / target_size
    if scale < 1e-6:
        scale = 1.0

    scaled_strokes = [(pts / scale) for pts in stroke_raw]

    # Step 2: Smooth each stroke with a Gaussian filter (sigma=1)
    smoothed_strokes = []
    for pts in scaled_strokes:
        if len(pts) > 3:
            pts_smooth = gaussian_filter1d(pts, sigma=1, axis=0)
        else:
            pts_smooth = pts
        smoothed_strokes.append(pts_smooth)

    # Step 3: Build sequence with pen-up between strokes
    all_points = []
    for idx, pts in enumerate(smoothed_strokes):
        for px, py in pts:
            all_points.append([px, py, 0])
        if idx < len(smoothed_strokes) - 1:
            last_x, last_y = pts[-1]
            all_points.append([last_x, last_y, 1])

    if len(all_points) == 0:
        return None

    points = np.array(all_points, dtype=np.float32)
    if len(points) > max_len:
        points = points[:max_len]
    if len(points) == 0:
        return None

    seq = torch.from_numpy(points).to(device)
    x = seq[:, 0]
    y = seq[:, 1]
    pen = seq[:, 2]

    dx = torch.diff(x, prepend=x[0:1])
    dy = torch.diff(y, prepend=y[0:1])
    r = torch.sqrt(dx**2 + dy**2) + 1e-8
    log_r = torch.log(r)
    angle = torch.atan2(dy, dx)
    pen_up = (pen == 1).float()
    pen_down = (pen == 0).float()

    features = torch.stack([dx, dy, log_r, angle, pen_up, pen_down], dim=1)

    # OPTIONAL: Apply global stats (if they were computed on similar scale)
    # features[:, :4] = (features[:, :4] - global_mean) / (global_std + 1e-8)

    features = features.unsqueeze(0)
    lengths = torch.tensor([features.size(1)], dtype=torch.long).to(device)
    return features, lengths

# -------------------- Prediction with Fusion --------------------
def predict(strokes, image_data, threshold, return_beams=False):
    # If offline model is available, use it
    if offline_model is not None and image_data is not None:
        offline_input = canvas_to_offline_input(image_data)
        with torch.no_grad():
            offline_logits = offline_model(offline_input)  # shape: (seq_len, batch, num_classes)
            # Remove batch dimension (batch=1)
            # Logits shape: (seq_len, batch, classes) -> squeeze(1) -> (seq_len, classes)
            offline_log_probs = torch.log_softmax(offline_logits, dim=-1).squeeze(1).detach().cpu().numpy()
            # Ensure we have 2D
            if offline_log_probs.ndim != 2:
                st.error(f"Unexpected shape after squeeze: {offline_log_probs.shape}")
                return [] if return_beams else ""
            # offline_vocab has blank at index 0
            idx_to_char = offline_vocab
            beams = ctc_prefix_beam_search(offline_log_probs, idx_to_char, beam_width=50, top_k=5, blank_index=0)
            if return_beams:
                return beams
            else:
                return beams[0][0] if beams else ""

    # Fallback to online model
    online_result = strokes_to_features(strokes)
    if online_result is None:
        return [] if return_beams else ""

    features, lengths = online_result
    with torch.no_grad():
        online_logits = online_model(features, lengths)
        online_log_probs = torch.log_softmax(online_logits, dim=-1).squeeze(1).detach().cpu().numpy()
        output_lengths = compute_output_lengths(lengths, downsample_factor=1)
        max_out = online_logits.size(0)
        output_lengths = torch.clamp(output_lengths, max=max_out)
        valid_len = int(output_lengths[0].item())
        online_log_probs = online_log_probs[:valid_len, :]
        idx_to_char = alphabet   # blank at 0
        beams = ctc_prefix_beam_search(online_log_probs, idx_to_char, beam_width=50, top_k=5, blank_index=0)

    if return_beams:
        return beams
    else:
        return beams[0][0] if beams else ""



# -------------------- Image Preprocessing for Offline --------------------
def canvas_to_offline_input(image_data: np.ndarray, target_h: int = 128, target_w: int = 512) -> torch.Tensor:
    if image_data.shape[-1] == 4:
        image = Image.fromarray(image_data).convert('L')
    else:
        image = Image.fromarray(image_data).convert('L')
    image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    img_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
    img_tensor = img_tensor.unsqueeze(0)  # (1, H, W)
    return img_tensor.unsqueeze(0).to(device)  # (1, 1, H, W)


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Fused Handwriting Recognition", layout="centered")
st.title("✍️ Online + Offline Fusion Demo")
threshold = st.slider("Online confidence threshold", 0.0, 1.0, 0.6, 0.05)
show_beams = st.checkbox("Show top-5 beam hypotheses")
st.markdown("Draw a word below. The prediction uses fusion if offline model is loaded.")

canvas_result = st_canvas(
    fill_color="rgba(255,255,255,0)",
    stroke_width=3,
    stroke_color="#000000",
    background_color="#ffffff",
    height=300,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.json_data is not None:
    strokes = canvas_result.json_data["objects"]
    stroke_list = []
    for obj in strokes:
        if obj.get("type") == "path":
            path = obj.get("path", [])
            if isinstance(path, list) and len(path) > 0:
                stroke_list.append({"path": path})
    if stroke_list:
        if show_beams:
            beams = predict(stroke_list, canvas_result.image_data, threshold, return_beams=True)
            st.write("Top-5 hypotheses:")
            for i, (hyp, score) in enumerate(beams):
                st.write(f"{i+1}: {hyp} (score: {score:.4f})")
            pred = beams[0][0] if beams else ""
            st.success(f"Best: **{pred}**")
        else:
            pred = predict(stroke_list, canvas_result.image_data, threshold, return_beams=False)
            st.success(f"Prediction: **{pred}**")
    else:
        st.info("No strokes detected.")