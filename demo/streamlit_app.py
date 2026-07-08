import sys
from pathlib import Path
from dataclasses import dataclass

# ---- 1. Set project root ----
project_root = Path(__file__).resolve().parent.parent   # points to MuDi-HTR/
sys.path.insert(0, str(project_root))

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
from scripts.train_online import load_config, compute_output_lengths
from inference.fusion import perform_beam_fusion
from preprocessing.inference_preprocess import try_preprocess_variants, select_best_candidate
from preprocessing.offline_preprocess import preprocess_image_from_pil

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

# Diagnostics expander (defined early so load-time diagnostics can use it)
diag = st.sidebar.expander("Diagnostics & Model Load", expanded=False)

if STATS_PATH.exists():
    stats = torch.load(STATS_PATH, weights_only=False)
    global_mean = stats['mean'].to(device)
    global_std = stats['std'].to(device)
    # Put numeric diagnostics into the diagnostics expander
    try:
        diag.write("Global mean:")
        diag.write(global_mean.cpu().numpy())
        diag.write("Global std:")
        diag.write(global_std.cpu().numpy())
    except Exception:
        diag.write((global_mean.cpu().numpy(), global_std.cpu().numpy()))
else:
    st.error("feature_stats.pt not found. Please run compute_stats.py first.")
    st.stop()

# -------------------- Load Models --------------------
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

@st.cache_resource
def cached_load_online(ckpt_path_str: str):
    path = Path(ckpt_path_str)
    model = OnlineHTRModel(input_size=config['model']['input_size'],
                           hidden_size=config['model']['hidden_size'],
                           num_layers=config['model']['num_layers'],
                           num_classes=config['model']['num_classes'],
                           dropout=0.0)
    if path.exists():
        ckpt = torch.load(path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt['model_state_dict'])
        except Exception:
            pass
    model.to(device).eval()
    decoder = CTCDecoder(alphabet, blank_idx=0)
    return model, decoder

ONLINE_CKPT = project_root / "models" / "checkpoints" / "isgl" / "best_isgl_final.pth"
online_model, decoder = cached_load_online(str(ONLINE_CKPT))

def load_offline():
    if not OFFLINE_CKPT.exists():
        return None, None, None, {"error": "offline_ckpt_missing"}
    ckpt = torch.load(OFFLINE_CKPT, map_location=device, weights_only=False)
    
    # Extract vocabulary
    vocab = ckpt.get('encoder_vocab', [])
    if not vocab:
        vocab = ckpt.get('config', {}).get('vocab', [])
    if not vocab:
        return None, None, None, {"error": "missing_vocab"}

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
    info = {}
    if missing:
        info['missing'] = missing
    if unexpected:
        info['unexpected'] = unexpected

    model.eval()
    return model, vocab, ckpt, info


@st.cache_resource
def cached_load_offline(ckpt_path_str: str):
    """Load offline model into a cached resource without calling Streamlit layout."""
    path = Path(ckpt_path_str)
    if not path.exists():
        return None, None, None, {"error": "offline_ckpt_missing"}
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = ckpt.get('encoder_vocab', []) or ckpt.get('config', {}).get('vocab', [])
    if not vocab:
        return None, None, None, {"error": "missing_vocab"}
    state_dict = ckpt.get('model_state_dict', ckpt)

    classifier_weight = state_dict.get('classifier.weight')
    if classifier_weight is not None:
        num_classes = int(classifier_weight.shape[0])
    else:
        num_classes = len(vocab)

    # infer hidden size
    hidden = 256
    for k in state_dict:
        if 'rnn.weight_ih_l0' in k:
            hidden = int(state_dict[k].shape[0] // 4)
            break

    model = CRNN(num_classes=num_classes, hidden_size=hidden)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    info = {}
    if missing:
        info['missing'] = missing
    if unexpected:
        info['unexpected'] = unexpected
    model.to(device).eval()
    return model, vocab, ckpt, info

# Prefer finetuned checkpoint when available
ckpt_dir = project_root / "models" / "checkpoints" / "offline"
finetuned_ckpt = ckpt_dir / "finetuned.pth"
pretrained_ckpt = ckpt_dir / "pretrained.pth"
if finetuned_ckpt.exists():
    OFFLINE_CKPT = finetuned_ckpt
else:
    OFFLINE_CKPT = pretrained_ckpt

# Use cached loader to avoid reloading the offline model on every rerun
offline_model, offline_vocab, offline_ckpt, offline_diag = cached_load_offline(str(OFFLINE_CKPT))

offline_vocab_map = None
unified_tokens = []
if offline_model is not None:
    online_vocab, offline_vocab_map, unified_tokens = build_branch_vocabularies(alphabet, offline_vocab)
    diag.success("Both online and offline models loaded — fusion available.")
    try:
        diag.write(f"Offline checkpoint used: {OFFLINE_CKPT}")
    except Exception:
        pass
else:
    # show diagnostic messages returned from load_offline()
    if offline_diag is None:
        diag.info("Offline model not found — using online-only.")
    else:
        if offline_diag.get('error') == 'offline_ckpt_missing':
            diag.info("Offline checkpoint file not found — using online-only.")
        elif offline_diag.get('error') == 'missing_vocab':
            diag.error("Offline checkpoint is missing encoder vocabulary — using online-only.")
        else:
            # Any load-time warnings (missing/unexpected keys)
            if 'missing' in offline_diag:
                diag.warning(f"Missing keys in offline checkpoint: {offline_diag['missing'][:6]}...")
            if 'unexpected' in offline_diag:
                diag.warning(f"Unexpected keys in offline checkpoint: {offline_diag['unexpected'][:6]}...")

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
    try:
        # features shape: (1, T, 6) -> normalize first 4 channels (dx, dy, log_r, angle)
        gm = global_mean.view(1, 1, -1) if hasattr(global_mean, 'view') else torch.tensor(global_mean).view(1, 1, -1)
        gs = global_std.view(1, 1, -1) if hasattr(global_std, 'view') else torch.tensor(global_std).view(1, 1, -1)
        features[:, :, :4] = (features[:, :, :4] - gm.to(device)) / (gs.to(device) + 1e-8)
    except Exception:
        # If normalization fails, proceed without it
        pass

    features = features.unsqueeze(0)
    lengths = torch.tensor([features.size(1)], dtype=torch.long).to(device)
    return features, lengths

# -------------------- Prediction with Fusion --------------------
def predict(strokes, image_data, fusion_alpha, return_beams=False):
    # fast canvas path uses greedy decoding and smaller beams
    def _greedy_decode_from_logits(logits: torch.Tensor, idx_to_char: list[str]):
        probs = torch.softmax(logits, dim=-1).squeeze(1).detach().cpu().numpy()
        indices = probs.argmax(axis=-1)
        # collapse repeats and remove blanks (index 0)
        chars = []
        prev = None
        for idx in indices:
            if idx != prev and idx != 0 and idx < len(idx_to_char):
                chars.append(idx_to_char[idx])
            prev = idx
        return ''.join(chars)
    # If offline model is available and an image is provided, prefer fusion when possible
    if offline_model is not None and image_data is not None:
        # Try to get online representation if strokes present
        online_result = strokes_to_features(strokes) if strokes else None
        # Use faster preprocessing for canvas inputs when appropriate
        offline_input = canvas_to_offline_input(image_data)

        with torch.no_grad():
            offline_logits = offline_model(offline_input)
            if online_result is not None:
                features, lengths = online_result
                online_logits = online_model(features, lengths)
                # For canvas (interactive) predictions use a fast greedy fusion path
                if strokes and len(strokes) > 0:
                    # Build small beam lists from both branches and fuse their candidate scores.
                    # Convert logits to log-prob arrays for beam search helper.
                    online_log_probs = torch.log_softmax(online_logits, dim=-1).squeeze(1).detach().cpu().numpy()
                    offline_log_probs = torch.log_softmax(offline_logits, dim=-1).squeeze(1).detach().cpu().numpy()

                    online_beams = ctc_prefix_beam_search(online_log_probs, alphabet, beam_width=BEAM_WIDTH_CANVAS, top_k=5, blank_index=0)
                    offline_beams = ctc_prefix_beam_search(offline_log_probs, offline_vocab, beam_width=BEAM_WIDTH_CANVAS, top_k=5, blank_index=0)

                    candidates = {}
                    for hyp, score in online_beams:
                        entry = candidates.setdefault(hyp, {'online': -np.inf, 'offline': -np.inf})
                        entry['online'] = score
                    for hyp, score in offline_beams:
                        entry = candidates.setdefault(hyp, {'online': -np.inf, 'offline': -np.inf})
                        entry['offline'] = score

                    # Convert log-scores to probabilities per-branch (softmax over candidate set)
                    online_scores = np.array([candidates[h]['online'] for h in candidates], dtype=np.float64)
                    offline_scores = np.array([candidates[h]['offline'] for h in candidates], dtype=np.float64)
                    # Replace -inf with a large negative number for stability
                    large_neg = -1e9
                    online_scores = np.where(np.isneginf(online_scores), large_neg, online_scores)
                    offline_scores = np.where(np.isneginf(offline_scores), large_neg, offline_scores)

                    # Softmax
                    def softmax(x):
                        x = x - x.max()
                        ex = np.exp(x)
                        s = ex / (ex.sum() + 1e-12)
                        return s

                    online_probs = softmax(online_scores)
                    offline_probs = softmax(offline_scores)

                    hyps = list(candidates.keys())
                    combined_scores = fusion_alpha * online_probs + (1.0 - fusion_alpha) * offline_probs
                    best_idx = int(np.argmax(combined_scores))
                    fused_hyp = hyps[best_idx]
                    if return_beams:
                        return fused_hyp, online_beams, offline_beams, online_log_probs, offline_log_probs
                    return fused_hyp

                fused, online_beams, offline_beams = perform_beam_fusion(
                    online_logits, offline_logits, online_alphabet=alphabet, offline_alphabet=offline_vocab,
                    alpha_weight=fusion_alpha,
                )
                # convert logits to log-probs for diagnostics
                online_log_probs = torch.log_softmax(online_logits, dim=-1).squeeze(1).detach().cpu().numpy()
                offline_log_probs = torch.log_softmax(offline_logits, dim=-1).squeeze(1).detach().cpu().numpy()
                if return_beams:
                    return fused, online_beams, offline_beams, online_log_probs, offline_log_probs
                if return_beams:
                    return fused
                return fused[0][0] if fused else ""

            # If no online features, fall back to offline-only decoding
            offline_log_probs = torch.log_softmax(offline_logits, dim=-1).squeeze(1).detach().cpu().numpy()
            if offline_log_probs.ndim != 2:
                st.error(f"Unexpected shape after squeeze: {offline_log_probs.shape}")
                return [] if return_beams else ""
            idx_to_char = offline_vocab
            beams = ctc_prefix_beam_search(offline_log_probs, idx_to_char, beam_width=50, top_k=5, blank_index=0)
            if return_beams:
                return beams, None, None, None, offline_log_probs
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
        return beams, None, None, None, None
    else:
        return beams[0][0] if beams else ""



# -------------------- Image Preprocessing for Offline --------------------
def canvas_to_offline_input(image_data: np.ndarray, target_h: int = 128, target_w: int = 512) -> torch.Tensor:
    # Convert to PIL and apply the same preprocessing used during offline training
    if image_data is None:
        raise ValueError("canvas image_data is None")
    pil = Image.fromarray(image_data).convert('L')
    proc = preprocess_image_from_pil(pil, image_size=(target_h, target_w), augment=False, binarize=True)
    img_tensor = torch.from_numpy(proc.astype(np.float32)).unsqueeze(0)  # (1, H, W)
    return img_tensor.unsqueeze(0).to(device)  # (1, 1, H, W)


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Online + Offline Handwriting Recognition", layout="wide")
st.title("Online + Offline Handwriting Recognition")
st.caption("MuDi-HTR: multi-branch handwriting recognition demo")

# Diagnostics expander in the sidebar (collapsed by default)
# (already defined earlier to allow load-time diagnostics)

# -------------------- Upload & Segmentation UI --------------------
@st.cache_resource
def get_easyocr_reader():
    try:
        import easyocr
    except Exception:
        return None
    return easyocr.Reader(["en"], gpu=False)

with st.sidebar:
    st.markdown("## Demo")
    # Mode selector allows switching between Canvas and Upload regardless of whether a file
    mode = st.radio(
        "Mode",
        ("Canvas", "Upload"),
        index=1 if 'upload' in st.session_state and st.session_state.get('upload') else 0,
        horizontal=True,
    )

    uploaded_file = None
    with st.expander("Upload & Run", expanded=(mode == "Upload")):
        if mode == "Upload":
            uploaded_file = st.file_uploader("Upload handwritten image", type=["png","jpg","jpeg","tif","tiff"], key='upload')
            # allow clearing the upload to return to canvas easily
            if st.session_state.get('upload') is not None:
                if st.button("Clear upload"):
                    st.session_state['upload'] = None
                    uploaded_file = None
        else:
            st.caption("Switch to Upload mode to load an image.")

    with st.expander("Inference Settings", expanded=False):
        fusion_weight = st.slider("Online weight (α)", 0.0, 1.0, 0.55, 0.05)
        show_beams = st.checkbox("Show top-5 beam hypotheses", value=False)

    # Removed unused toggles per user request
    invert_images = False
    auto_segment = False


def is_blank_crop(pil_crop: Image.Image, image_size=(128,512), blank_thresh: float = 0.02, min_height: int = 8) -> bool:
    """Return True if crop appears blank or too small to be meaningful.

    Uses offline preprocessing binarization to estimate foreground fraction.
    """
    try:
        if pil_crop.size[1] < min_height:
            return True
        proc = preprocess_image_from_pil(pil_crop, image_size=image_size, augment=False, binarize=True)
        # proc is float array in [0,1] after binarize; compute fraction of foreground (non-zero)
        fg_frac = float((proc > 0.5).mean())
        return fg_frac < blank_thresh
    except Exception:
        return False

if uploaded_file is not None:
    st.image(uploaded_file, width=300)
    # Place the run button in the main UI so it's visible when an upload exists
    run_button = st.button("Recognize Upload")
else:
    run_button = False

# Note: canvas toolbar already has a clear button; no explicit clear button needed here.

BEAM_WIDTH = 5

# Larger beam for interactive canvas to improve spacing/ambiguous char decoding
BEAM_WIDTH_CANVAS = 12

# Offline target size used for preprocessing
OFFLINE_TARGET_H = 128
OFFLINE_TARGET_W = 512
# UI scale factor to make the canvas larger for comfortable writing
CANVAS_UI_SCALE = 2

# Canvas and inference (only show canvas when not in upload mode)
if uploaded_file is None:
    st.subheader("Write something below.")
    canvas_width = OFFLINE_TARGET_W * CANVAS_UI_SCALE
    canvas_height = OFFLINE_TARGET_H * CANVAS_UI_SCALE
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result is not None and canvas_result.json_data is not None:
        strokes = canvas_result.json_data["objects"]
        stroke_list = []
        for obj in strokes:
            if obj.get("type") == "path":
                path = obj.get("path", [])
                if isinstance(path, list) and len(path) > 0:
                    stroke_list.append({"path": path})
        if stroke_list:
            # Show preprocessed canvas preview in Diagnostics
            try:
                pil_preview = Image.fromarray(canvas_result.image_data).convert('L')
                proc_preview = preprocess_image_from_pil(pil_preview, image_size=(128,512), augment=False, binarize=True)
                diag.image((proc_preview * 255).astype('uint8'), caption="Preprocessed canvas (128x512)")
            except Exception:
                pass

            if show_beams:
                with st.spinner("Running fusion inference..."):
                    fused, online_beams, offline_beams, online_log_probs, offline_log_probs = predict(stroke_list, canvas_result.image_data, fusion_weight, return_beams=True)
                st.write("Top fused hypothesis:")
                st.success(f"{fused if isinstance(fused,str) else fused[0]}")
                # Show per-branch beams in Diagnostics
                try:
                    diag.markdown("**Online beams (top-5)**")
                    for hyp, score in online_beams[:5]:
                        diag.write(f"{hyp} — {score:.4f}")
                    diag.markdown("**Offline beams (top-5)**")
                    for hyp, score in offline_beams[:5]:
                        diag.write(f"{hyp} — {score:.4f}")
                except Exception:
                    pass
            else:
                with st.spinner("Running fusion inference..."):
                    pred = predict(stroke_list, canvas_result.image_data, fusion_weight, return_beams=False)
                st.success(f"Prediction: **{pred}**")
    # when no strokes, don't show a notification to avoid clutter

# -------------------- Upload run handler --------------------
if run_button:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        from io import BytesIO
        from pathlib import Path
        import time

        image_bytes = uploaded_file.getvalue()
        pil = Image.open(BytesIO(image_bytes)).convert('L')

        out_dir = project_root / "debug_segments" / f"run_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # segmentation
        reader = get_easyocr_reader()
        crops_meta = []
        if reader is None:
            diag.info("EasyOCR not available — using horizontal projection segmentation")
            from scripts.segment_with_craft import horizontal_projection_segment
            img_np = np.array(pil).astype(np.float32) / 255.0
            crops_meta = horizontal_projection_segment(img_np, out_dir, expand=8, min_area=100)
            # Filter out blank / tiny crops
            filtered = []
            for e in crops_meta:
                try:
                    p = Path(e['path'])
                    im = Image.open(p).convert('L')
                    if is_blank_crop(im):
                        continue
                    filtered.append(e)
                except Exception:
                    continue
            crops_meta = filtered
        else:
            diag.info("Using EasyOCR for segmentation")
            results = reader.readtext(np.array(pil), detail=1)
            used = 0
            for (bbox, text, conf) in results:
                xs = np.array([p[0] for p in bbox])
                ys = np.array([p[1] for p in bbox])
                x0 = int(max(0, np.floor(xs.min()) - 8))
                x1 = int(min(pil.size[0], np.ceil(xs.max()) + 8))
                y0 = int(max(0, np.floor(ys.min()) - 8))
                y1 = int(min(pil.size[1], np.ceil(ys.max()) + 8))
                crop = pil.crop((x0, y0, x1, y1))
                # Skip blank / tiny crops
                if is_blank_crop(crop):
                    continue
                out_path = out_dir / f"crop_{used:03d}.png"
                crop.save(out_path)
                crops_meta.append({"id": used, "bbox": [x0, y0, x1, y1], "path": str(out_path), "conf": float(conf), "text": text})
                used += 1

        results = []
        if not crops_meta:
            st.warning("No crops found; attempting to run offline model on the full image.")
            if offline_model is None:
                st.error("Offline model not loaded — cannot run image inference.")
            else:
                proc = preprocess_image_from_pil(pil if not invert_images else ImageOps.invert(pil), image_size=(128,512), augment=False, binarize=True)
                tensor = torch.tensor(proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = offline_model(tensor)
                    decoded = offline_model.decode_beam_search(logits, alphabet=offline_vocab, beam_width=BEAM_WIDTH)
                    best_text, best_conf = decoded[0] if decoded else ("", -1.0)
                st.markdown(f"**Full-image prediction:** {best_text} (conf={best_conf:.3f})")
                results.append({"id": 0, "path": "full_image", "text": best_text, "conf": float(best_conf)})
        else:
            st.markdown(f"### Found {len(crops_meta)} crops — running per-crop inference")
            from preprocessing.inference_preprocess import try_preprocess_variants, select_best_candidate
            for entry in crops_meta:
                crop_path = Path(entry["path"])
                pil_crop = Image.open(crop_path).convert('L')
                if invert_images:
                    pil_crop = ImageOps.invert(pil_crop)
                candidates = try_preprocess_variants(pil_crop)
                if offline_model is None:
                    st.error("Offline model not loaded — cannot run crop inference.")
                    break
                best = select_best_candidate(offline_model, offline_vocab, candidates, device=device, beam_width=BEAM_WIDTH)
                st.image(str(crop_path), width=400)
                pred_text = best.get('text','')
                st.markdown(f"**{pred_text}** — conf={best.get('conf',-1.0):.3f} (method={best.get('method')} invert={best.get('invert')})")
                # Show raw repr and candidate list for debugging (sanitized to avoid circular refs)
                with st.expander("Prediction details"):
                    safe_candidates = []
                    for r in best.get('results', []):
                        safe_candidates.append({
                            'method': r.get('method'),
                            'invert': bool(r.get('invert')),
                            'text': r.get('text'),
                            'conf': float(r.get('conf', -1.0)),
                        })
                    st.write({'repr': repr(pred_text), 'candidates': safe_candidates})
                results.append({"id": entry['id'], "path": str(crop_path), "text": best.get('text',''), "conf": float(best.get('conf',-1.0)), "method": best.get('method'), "invert": best.get('invert')})

        # Save results and provide download
        if results:
            out_json = out_dir / "results.json"
            import json
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            with open(out_json, 'rb') as f:
                st.download_button("Download results.json", f, file_name="results.json")


