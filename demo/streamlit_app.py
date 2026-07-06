import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pathlib import Path
import sys
import json


sys.path.append(str(Path(__file__).parent.parent / "models" / "online"))

from models.online.model import OnlineHTRModel, CTCDecoder
from models.online.dataset import CTCLabelEncoder
from models.online.train_isgl import load_config, compute_output_lengths

# -------------------- CONFIG --------------------
CONFIG_PATH = Path(__file__).parent.parent / "models" / "online" / "config_isgl.yaml"
ONLINE_CHECKPOINT = Path(__file__).parent.parent / "models" / "online" / "checkpoints_final" / "isgl" / "best_isgl_final.pth"
# If you have an offline model checkpoint, set its path here:
OFFLINE_CHECKPOINT = None  # Path to offline model .pth, or None if not available

# Beam search parameters
BEAM_WIDTH = 20
TOP_K = 5

# Global stats for normalization (same as used in training)
STATS_PATH = Path(__file__).parent.parent / "models" / "online" / "feature_stats.pt"

# -------------------- LOAD CONFIG AND STATS --------------------
config = load_config(CONFIG_PATH)['iam']
alphabet = list(config['alphabet'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load global stats
if STATS_PATH.exists():
    stats = torch.load(STATS_PATH, weights_only=False)
    global_mean = stats['mean'].to(device)
    global_std = stats['std'].to(device)
else:
    st.error("feature_stats.pt not found. Please run compute_stats.py first.")
    st.stop()

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_online_model():
    model = OnlineHTRModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=0.0  # for inference
    ).to(device)
    checkpoint = torch.load(ONLINE_CHECKPOINT, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

online_model = load_online_model()
decoder = CTCDecoder(alphabet, blank_idx=0)

# Optionally load offline model if available
offline_model = None
if OFFLINE_CHECKPOINT is not None and OFFLINE_CHECKPOINT.exists():
    # Here you would define your offline model class and load it.
    # Since we don't have that, we'll skip for now.
    st.warning("Offline model not implemented in this app yet.")
else:
    st.info("Using only online model for prediction.")

# -------------------- PREPROCESSING --------------------
def strokes_to_features(strokes, max_len=500):
    """
    Convert strokes (list of (x, y, pen_down) tuples) into a tensor of shape (seq_len, 6).
    strokes: list of dicts with 'x', 'y', and 'pen_down' (or 'stroke' with points).
    """
    # If strokes come as a list of stroke paths, each path is a list of points.
    # We need to concatenate all points with pen_up/down flags.
    # The canvas returns a list of stroke dictionaries with 'path' (list of points).
    # Each point: [x, y] (normalized 0-1) or pixel coordinates.
    # We'll convert to a flat array of (x, y, pen) where pen=0 for drawing, 1 for pen-up between strokes.
    all_points = []
    for stroke in strokes:
        # Each stroke is a dict with 'path': list of [x, y] (normalized)
        path = stroke.get('path', [])
        if len(path) == 0:
            continue
        # Add points
        for i, (px, py) in enumerate(path):
            all_points.append([px, py, 0])  # pen down during stroke
        # Add a pen-up marker between strokes (except after the last)
        if len(all_points) > 0:
            all_points.append([all_points[-1][0], all_points[-1][1], 1])  # pen up at last point

    if len(all_points) == 0:
        return None

    # Convert to numpy array
    points = np.array(all_points, dtype=np.float32)
    # If the last point has pen_up, remove it (we don't need trailing up)
    if len(points) > 0 and points[-1, 2] == 1:
        points = points[:-1]

    # Truncate if too long
    if len(points) > max_len:
        points = points[:max_len]

    # Convert to torch tensor
    seq = torch.from_numpy(points).to(device)  # (seq_len, 3)

    # Compute derivatives
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

    # Stack 6 features
    features = torch.stack([dx, dy, log_r, angle, pen_up, pen_down], dim=1)  # (seq_len, 6)

    # Normalize with global stats
    features[:, :4] = (features[:, :4] - global_mean) / (global_std + 1e-8)

    # Add batch dimension
    features = features.unsqueeze(0)  # (1, seq_len, 6)
    lengths = torch.tensor([features.size(1)], dtype=torch.long).to(device)

    return features, lengths

# -------------------- PREDICTION --------------------
def predict(strokes):
    result = strokes_to_features(strokes)
    if result is None:
        return ""

    features, lengths = result

    with torch.no_grad():
        logits = online_model(features, lengths)
        output_lengths = compute_output_lengths(lengths, downsample_factor=1)  # factor=1 for LSTM
        max_out = logits.size(0)
        output_lengths = torch.clamp(output_lengths, max=max_out)

        # Greedy (for quick display)
        greedy_text = decoder.greedy_decode(logits, output_lengths)[0]

        # Beam search (top 5)
        beams = decoder.beam_search(logits, output_lengths, beam_width=BEAM_WIDTH, top_k=TOP_K)[0]
        # beams is list of (hypothesis, score)

    # For now, just return the best beam (top score)
    best_beam = beams[0][0] if beams else greedy_text
    return best_beam

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Online Handwriting Recognition", layout="centered")
st.title("✍️ Online Handwriting Recognition (Fused Model Demo)")

st.markdown("Draw a word or short phrase in the canvas below. The model will predict the text after you finish drawing.")

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=3,
    stroke_color="#000000",
    background_color="#ffffff",
    height=300,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # The canvas returns stroke data in canvas_result.json_data
    if canvas_result.json_data is not None:
        strokes = canvas_result.json_data["objects"]
        # Each object has 'path' (list of points) and 'type' == 'path'
        # We need to extract paths
        stroke_list = []
        for obj in strokes:
            if obj["type"] == "path":
                path = obj["path"]
                # path is a list of lists [[x, y], [x, y], ...]
                # Convert to (x, y) normalized? The canvas coordinates are in pixels.
                # We'll keep them as pixel coordinates – the model expects relative offsets,
                # but the derivatives (dx, dy) are scale-invariant? Not exactly.
                # For consistency, we should normalize x,y to a fixed range (e.g., 0-1) or
                # we could use raw pixel coordinates; the model was trained on normalized data.
                # Since the training data is in pixel coordinates (after preprocessing),
                # we can just use the pixel coordinates directly.
                stroke_list.append({"path": path})
        # Predict
        if stroke_list:
            prediction = predict(stroke_list)
            st.success(f"Prediction: **{prediction}**")
        else:
            st.info("Draw something to see the prediction.")
    else:
        st.info("Draw something to see the prediction.")
else:
    st.info("Draw something to see the prediction.")

# Optional: show the top 5 beam hypotheses
if st.checkbox("Show top beam hypotheses"):
    if canvas_result.json_data is not None:
        strokes = canvas_result.json_data["objects"]
        stroke_list = []
        for obj in strokes:
            if obj["type"] == "path":
                path = obj["path"]
                stroke_list.append({"path": path})
        if stroke_list:
            result = strokes_to_features(stroke_list)
            if result is not None:
                features, lengths = result
                with torch.no_grad():
                    logits = online_model(features, lengths)
                    output_lengths = compute_output_lengths(lengths, downsample_factor=1)
                    max_out = logits.size(0)
                    output_lengths = torch.clamp(output_lengths, max=max_out)
                    beams = decoder.beam_search(logits, output_lengths, beam_width=BEAM_WIDTH, top_k=5)[0]
                st.write("Top 5 hypotheses:")
                for i, (hyp, score) in enumerate(beams):
                    st.write(f"{i+1}: {hyp} (score: {score:.4f})")