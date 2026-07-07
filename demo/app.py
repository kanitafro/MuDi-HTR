

# demo/app.py
import sys
from pathlib import Path

# Fix the ModuleNotFoundError by adding the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import yaml
import math

# Now these imports will work beautifully!
from models.online.model import OnlineHTRModel, CTCDecoder
from models.offline.model import CRNN

# Define device globally so preprocessing functions can safely access it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

offline_ckpt_path = project_root / "models" / "checkpoints" / "offline" / "pretrained.pth"
online_ckpt_path = project_root / "models" / "checkpoints" / "isgl" / "best_isgl_final.pth"
stats_path = project_root / "models" / "online" / "feature_stats.pt"
config_path = project_root / "models" / "online" / "config_isgl.yaml"


# --- Configuration & Cache Loading ---
@st.cache_resource
def load_models_and_configs():
    # Load configuration from the project root area
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['iam']
    
    online_alphabet = list(config['alphabet'])
    
    # 1. Load Online Model
    online_model = OnlineHTRModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=0.0
    )
    if online_ckpt_path.exists():
        checkpoint = torch.load(online_ckpt_path, map_location=device)
        online_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        st.error(f"Online checkpoint missing at: {online_ckpt_path}")
        
    online_model.to(device).eval()
    
    # 2. Load Offline Model (Aligned precisely to pretrained.pth architecture keys)
    offline_model = CRNN(num_classes=71, hidden_size=256)
    
    # Corrected CNN channels matching the checkpoint dimensions
    from models.offline.model import ConvBlock
    offline_model.cnn = nn.Sequential(
        ConvBlock(1, 64, (2, 2)),    # cnn.0
        ConvBlock(64, 128, (2, 2)),  # cnn.1
        ConvBlock(128, 256, (2, 1)), # cnn.2
        ConvBlock(256, 256, (2, 1)), # cnn.3
        ConvBlock(256, 512, (2, 1)), # cnn.4
    )
    
    # Define sequence_projection as a Sequential block projecting 512 down to 256
    offline_model.sequence_projection = nn.Sequential(
        nn.Linear(512, 256)
    )
    
    # Re-initialize the LSTM with input_size=256 and 2 layers to match checkpoint depth
    offline_model.rnn = nn.LSTM(
        input_size=256,
        hidden_size=256,
        num_layers=2, 
        bidirectional=True,
        batch_first=True,
        dropout=0.35,
    )
    
    # Re-initialize classifier to match 71 class output target mapping
    offline_model.classifier = nn.Linear(256 * 2, 71)

    # Using your precise path: PROJ_ROOT/models/checkpoints/offline/pretrained.pth
    

    if offline_ckpt_path.exists():
        offline_checkpoint = torch.load(offline_ckpt_path, map_location=device)
        if isinstance(offline_checkpoint, dict) and 'model_state_dict' in offline_checkpoint:
            offline_model.load_state_dict(offline_checkpoint['model_state_dict'])
        else:
            offline_model.load_state_dict(offline_checkpoint)
    else:
        st.error(f"Offline checkpoint missing at: {offline_ckpt_path}")
        
    offline_model.to(device).eval()
    
    # Set the matching alphabet for the offline prediction decoder (71 classes)
    offline_alphabet = list(config['alphabet'])[:71]
    
    # 3. Features stats for Online normalization
    if stats_path.exists():
        stats = torch.load(stats_path, map_location='cpu')
        global_mean = stats['mean']
        global_std = stats['std']
    else:
        global_mean, global_std = None, None
        
    return online_model, offline_model, online_alphabet, offline_alphabet, global_mean, global_std

# Unpack the configurations
online_model, offline_model, online_alph, offline_alph, global_mean, global_std = load_models_and_configs()

# --- Helper Utilities ---
def logsumexp(values):
    finite_values = [v for v in values if v > -np.inf]
    if not finite_values: return -np.inf
    max_val = max(finite_values)
    return float(max_val + np.log(sum(np.exp(v - max_val) for v in finite_values)))

# --- Preprocessing Functions ---
def preprocess_online_strokes(json_data):
    """Transforms raw canvas trace paths into the online 6-feature vector format."""
    strokes = []
    if not json_data or "objects" not in json_data:
        return None
        
    for obj in json_data["objects"]:
        if obj["type"] == "path":
            path_data = obj["path"]
            stroke_points = []
            for cmd in path_data:
                if cmd[0] in ['M', 'L'] and len(cmd) >= 3:
                    stroke_points.append([cmd[1], cmd[2], 0.0])
                elif cmd[0] == 'Q' and len(cmd) >= 5:
                    stroke_points.append([cmd[3], cmd[4], 0.0])
            if stroke_points:
                stroke_points[-1][2] = 1.0
                strokes.append(torch.tensor(stroke_points, dtype=torch.float32))
                
    if not strokes:
        return None
        
    sequence = torch.cat(strokes, dim=0)
    x, y, pen = sequence[:, 0], sequence[:, 1], sequence[:, 2]
    
    dx = torch.diff(x, prepend=x[0:1])
    dy = torch.diff(y, prepend=y[0:1])
    r = torch.sqrt(dx**2 + dy**2) + 1e-8
    log_r = torch.log(r)
    angle = torch.atan2(dy, dx)
    pen_up = (pen == 1).float()
    pen_down = (pen == 0).float()
    
    feat_seq = torch.stack([dx, dy, log_r, angle, pen_up, pen_down], dim=1)
    
    if global_mean is not None and global_std is not None:
        feat_seq[:, :4] = (feat_seq[:, :4] - global_mean) / global_std
        
    return feat_seq.unsqueeze(0).to(device)

def preprocess_offline_image(image_rgba):
    """Converts RGBA canvas array into inverted gray normalized (1, 1, 128, 512) tensor."""
    if image_rgba is None: return None
    alpha = image_rgba[:, :, 3]
    if np.sum(alpha) == 0: return None
    
    y_indices, x_indices = np.where(alpha > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    cropped = alpha[y_min:y_max+1, x_min:x_max+1]
    h, w = cropped.shape
    desired_ratio = 512 / 128
    current_ratio = w / h
    
    if current_ratio > desired_ratio:
        new_h = int(w / desired_ratio)
        pad_ver = max(0, (new_h - h) // 2)
        padded = np.pad(cropped, ((pad_ver, pad_ver), (0, 0)), mode='constant', constant_values=0)
    else:
        new_w = int(h * desired_ratio)
        pad_hor = max(0, (new_w - w) // 2)
        padded = np.pad(cropped, ((0, 0), (pad_hor, pad_hor)), mode='constant', constant_values=0)

    img_resized = cv2.resize(padded, (512, 128), interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    return img_tensor.unsqueeze(0).unsqueeze(0).to(device)

# --- Late Fusion Beam Search Decoding Logic ---
def perform_beam_fusion(online_logits, offline_logits, alpha_weight=0.5, beam_width=15):
    """Fuses output paths from online and offline models dynamically."""
    
    # 1. Safely normalize logits to log-probabilities to prevent exponential branch expansion freezes
    online_log_probs = torch.log_softmax(online_logits, dim=-1)
    offline_log_probs = torch.log_softmax(offline_logits, dim=-1)
    
    # 2. Extract standard 2D (Time, Classes) matrices
    on_log_probs_2d = online_log_probs.squeeze(1) if online_log_probs.ndim == 3 else online_log_probs
    
    if offline_log_probs.ndim == 3:
        if offline_log_probs.shape[1] == 1:
            off_log_probs_2d = offline_log_probs.squeeze(1) # Shape: (T, C)
        elif offline_log_probs.shape[0] == 1:
            off_log_probs_2d = offline_log_probs.squeeze(0) # Shape: (T, C)
        else:
            off_log_probs_2d = offline_log_probs[:, 0, :]
    else:
        off_log_probs_2d = offline_log_probs

    # 3. Add back clean fake batch dimensions (T, 1, C) expected by your CTCDecoder framework
    on_log_probs_clean = on_log_probs_2d.unsqueeze(1)
    off_log_probs_clean = off_log_probs_2d.unsqueeze(1)

    # 4. Run the beam search decoders with proper log-probabilities
    decoder_online = CTCDecoder(online_alph, blank_idx=0)
    online_beams = decoder_online.beam_search(on_log_probs_clean, beam_width=beam_width, top_k=beam_width)[0]
    
    decoder_offline = CTCDecoder(offline_alph, blank_idx=0)
    offline_beams = decoder_offline.beam_search(off_log_probs_clean, beam_width=beam_width, top_k=beam_width)[0]
    
    fused_scores = {}
    
    # Process Online candidates
    for text, score in online_beams:
        fused_scores[text] = fused_scores.get(text, 0.0) + alpha_weight * score
        
    # Process Offline candidates safely 
    for text, score in offline_beams:
        fused_scores[text] = fused_scores.get(text, 0.0) + (1.0 - alpha_weight) * score
        
    sorted_predictions = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_predictions, online_beams[:3], offline_beams[:3]


# --- Optimized Fast Prefix Beam Search Decoder ---
def fast_ctc_beam_search(log_probs_2d, alphabet, beam_width=15, blank_idx=0, top_k_chars=10):
    """
    A highly optimized, tensor-overhead-free CTC beam search decoder.
    Prunes characters using top_k_chars to execute instantaneously.
    """
    # log_probs_2d: (T, C) array or tensor
    if torch.is_tensor(log_probs_2d):
        log_probs_2d = log_probs_2d.detach().cpu().numpy()
        
    T, C = log_probs_2d.shape
    # beams map: prefix_tuple -> (p_blank, p_nonblank)
    beams = {(): (0.0, -float('inf'))}
    
    for t in range(T):
        timestep = log_probs_2d[t]
        next_beams = {}
        
        # Prune character set per timestep to avoid scanning all classes
        top_indices = np.argsort(timestep)[::-1][:top_k_chars]
        
        def update_beam(prefix, p_b=None, p_nb=None):
            exist_b, exist_nb = next_beams.get(prefix, (-float('inf'), -float('inf')))
            if p_b is not None:
                # fast logsumexp inline
                m = max(exist_b, p_b)
                exist_b = m + math.log(math.exp(exist_b - m) + math.exp(p_b - m)) if m > -float('inf') else -float('inf')
            if p_nb is not None:
                m = max(exist_nb, p_nb)
                exist_nb = m + math.log(math.exp(exist_nb - m) + math.exp(p_nb - m)) if m > -float('inf') else -float('inf')
            next_beams[prefix] = (exist_b, exist_nb)

        for prefix, (p_blank, p_nonblank) in beams.items():
            # 1. Handle Blank Token transition
            total_prob = max(p_blank, p_nonblank)
            if total_prob > -float('inf'):
                total_prob = total_prob + math.log(math.exp(p_blank - total_prob) + math.exp(p_nonblank - total_prob))
            
            update_beam(prefix, p_b=total_prob + float(timestep[blank_idx]))
            
            # 2. Handle Non-blank Character transitions
            for c in top_indices:
                if c == blank_idx:
                    continue
                char_score = float(timestep[c])
                if prefix and prefix[-1] == c:
                    update_beam(prefix, p_nb=p_nonblank + char_score)
                    update_beam(prefix + (c,), p_nb=p_blank + char_score)
                else:
                    update_beam(prefix + (c,), p_nb=total_prob + char_score)
                    
        # Prune the beams to beam_width
        sorted_beams = sorted(
            next_beams.items(),
            key=lambda item: max(item[1][0], item[1][1]) + math.log(math.exp(item[1][0] - max(item[1][0], item[1][1])) + math.exp(item[1][1] - max(item[1][0], item[1][1]))),
            reverse=True
        )
        beams = dict(sorted_beams[:beam_width])

    # Package output candidates with sequence length normalization
    candidates = []
    for prefix, (p_b, p_nb) in beams.items():
        m = max(p_b, p_nb)
        total_score = m + math.log(math.exp(p_b - m) + math.exp(p_nb - m)) if m > -float('inf') else -float('inf')
        norm_score = total_score / max(1, len(prefix))
        
        text = ''.join([alphabet[idx] for idx in prefix if idx < len(alphabet)])
        candidates.append((text, norm_score))
        
    return sorted(candidates, key=lambda x: x[1], reverse=True)


# --- Late Fusion Combination Routing ---
def perform_beam_fusion(online_logits, offline_logits, alpha_weight=0.5, beam_width=15):
    """Fuses output paths from online and offline models dynamically using fast search blocks."""
    
    # Standardize predictions to log-probabilities
    online_log_probs = torch.log_softmax(online_logits, dim=-1)
    offline_log_probs = torch.log_softmax(offline_logits, dim=-1)
    
    # Isolate matrices down to clean 2D shapes (Time, Classes)
    on_probs_2d = online_log_probs.squeeze(1) if online_log_probs.ndim == 3 else online_log_probs
    if offline_log_probs.ndim == 3:
        off_probs_2d = offline_log_probs.squeeze(1) if offline_log_probs.shape[1] == 1 else offline_log_probs.squeeze(0)
    else:
        off_probs_2d = offline_log_probs

    # Run the ultra-fast decoder instances
    online_beams = fast_ctc_beam_search(on_probs_2d, online_alph, beam_width=beam_width, blank_idx=0)
    offline_beams = fast_ctc_beam_search(off_probs_2d, offline_alph, beam_width=beam_width, blank_idx=0)
    
    # Compute score interpolation
    fused_scores = {}
    for text, score in online_beams:
        fused_scores[text] = fused_scores.get(text, 0.0) + alpha_weight * score
        
    for text, score in offline_beams:
        fused_scores[text] = fused_scores.get(text, 0.0) + (1.0 - alpha_weight) * score
        
    sorted_predictions = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_predictions, online_beams, offline_beams

# --- Streamlit Layout View ---
st.set_page_config(page_title="Multimodal HTR Fusion (Online + Offline)", layout="wide")
st.title("✒️ Multimodal Handwriting Recognition")
st.subheader("Late Fusion Beam Search Integration Engine")

# Put the model configuration sliders and action button inside a Form
with st.form(key="htr_prediction_form"):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("Draw words or text lines inside the digital canvas frame below:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=4,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=250,
            width=800,
            drawing_mode="freedraw",
            key="htr_canvas",
        )

    with col2:
        st.markdown("### Model Controls")
        fusion_weight = st.slider("Weight Weighting Factor (α: Online dominance)", 0.0, 1.0, 0.5, step=0.05)
        beam_width_sel = st.slider("Beam Width Size", 5, 30, 15)
        
        # This button triggers the execution manually and stops infinite re-runs!
        submit_button = st.form_submit_button(label="🔮 Recognize Writing")

# Only execute inference when the user explicitly clicks the Form Submit Button
if submit_button and canvas_result.json_data is not None:
    with st.spinner("Processing stroke paths and running beam search fusion..."):
        online_input = preprocess_online_strokes(canvas_result.json_data)
        offline_input = preprocess_offline_image(canvas_result.image_data)
        
        if online_input is not None and offline_input is not None:
            with torch.no_grad():
                # Get raw model outputs
                online_logits = online_model(online_input)   
                offline_logits = offline_model(offline_input) 
                
                # Apply log_softmax to dimension -1 immediately
                online_log_probs = torch.log_softmax(online_logits, dim=-1)
                offline_log_probs = torch.log_softmax(offline_logits, dim=-1)
                
                # Squeeze down to (T, C) shape to cleanly eliminate batch formatting variables
                on_probs_fixed = online_log_probs.squeeze(1) if online_log_probs.ndim == 3 else online_log_probs
                if offline_log_probs.ndim == 3:
                    off_probs_fixed = offline_log_probs.squeeze(1) if offline_log_probs.shape[1] == 1 else offline_log_probs.squeeze(0)
                else:
                    off_probs_fixed = offline_log_probs

                # Run your CTCDecoders directly using standard (T, C) array matrices
                decoder_online = CTCDecoder(online_alph, blank_idx=0)
                online_beams = decoder_online.beam_search(on_probs_fixed, beam_width=beam_width_sel, top_k=beam_width_sel)[0]
                
                decoder_offline = CTCDecoder(offline_alph, blank_idx=0)
                offline_beams = decoder_offline.beam_search(off_probs_fixed, beam_width=beam_width_sel, top_k=beam_width_sel)[0]
                
                # Perform Late Fusion combination logic mapping
                fused_scores = {}
                for text, score in online_beams:
                    fused_scores[text] = fused_scores.get(text, 0.0) + fusion_weight * score
                    
                for text, score in offline_beams:
                    fused_scores[text] = fused_scores.get(text, 0.0) + (1.0 - fusion_weight) * score
                    
                fused_preds = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            
            # --- Render Output Metric Panels ---
            st.markdown("---")
            if fused_preds:
                st.success(f"### Final Fused Prediction Output: **`{fused_preds[0][0]}`**")
            
            c_on, c_off = st.columns(2)
            with c_on:
                st.markdown("#### 📱 Online Branch Hypotheses")
                for text, score in online_beams[:3]:
                    st.write(f"- **`{text}`** (Log Prob Score: `{score:.4f}`)")
                    
            with c_off:
                st.markdown("#### 🖼️ Offline Branch Hypotheses")
                for text, score in offline_beams[:3]:
                    st.write(f"- **`{text}`** (Log Prob Score: `{score:.4f}`)")
        else:
            st.warning("Could not extract valid strokes or image paths. Please write clearly on the canvas panel.")
else:
    if not submit_button:
        st.info("✍️ Draw your word on the canvas above and click **Recognize Writing** to process predictions.")