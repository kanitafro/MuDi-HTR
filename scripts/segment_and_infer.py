"""Segment a page image into crops and run offline OCR on each crop.

Usage:
  python scripts/segment_and_infer.py --image data/TEST_0001.jpg --out debug_crops

Outputs:
 - crops in out_dir (created by segmentation)
 - results.json with per-crop best text and confidence
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

import torch
from PIL import Image

# ensure project root on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.offline.model import CRNN
from preprocessing.inference_preprocess import try_preprocess_variants, select_best_candidate


def load_offline_model(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Offline checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    vocab = ckpt.get('encoder_vocab', []) or ckpt.get('config', {}).get('vocab', [])
    # infer sizes
    classifier_weight = state.get('classifier.weight')
    if classifier_weight is not None:
        num_classes = int(classifier_weight.shape[0])
    else:
        num_classes = len(vocab) if vocab else 80
    # infer hidden size
    hidden = 256
    for k in state:
        if 'rnn.weight_ih_l0' in k:
            hidden = int(state[k].shape[0] // 4)
            break
    model = CRNN(num_classes=num_classes, hidden_size=hidden)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, vocab


def run_segmentation(image_path: Path, out_dir: Path, use_easyocr: bool = True, gpu: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(project_root / 'scripts' / 'segment_with_craft.py'), '--image', str(image_path), '--out-dir', str(out_dir)]
    if not use_easyocr:
        cmd.append('--min-area')
        cmd.append('1')
    if gpu:
        cmd.append('--gpu')
    print('Running segmentation:', ' '.join(cmd))
    subprocess.check_call(cmd)
    meta_file = out_dir / 'crops_meta.json'
    if not meta_file.exists():
        raise FileNotFoundError('Segmentation did not produce crops_meta.json')
    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    return meta


def infer_on_crops(out_dir: Path, offline_model, offline_vocab, device: torch.device, beam_width: int = 10):
    meta_file = out_dir / 'crops_meta.json'
    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    crops = meta.get('meta', [])
    results = []
    for entry in crops:
        path = Path(entry.get('path'))
        if not path.exists():
            print('Skipping missing crop', path)
            continue
        pil = Image.open(path).convert('L')
        candidates = try_preprocess_variants(pil)
        best = select_best_candidate(offline_model, offline_vocab, candidates, device=device, beam_width=beam_width)
        results.append({
            'id': entry.get('id'),
            'path': str(path),
            'text': best.get('text', ''),
            'conf': float(best.get('conf', -1.0)),
            'method': best.get('method'),
            'invert': best.get('invert'),
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, default=Path('debug_crops'))
    # prefer finetuned checkpoint when available
    default_ckpt = project_root / 'models' / 'checkpoints' / 'offline' / 'finetuned.pth'
    if not default_ckpt.exists():
        default_ckpt = project_root / 'models' / 'checkpoints' / 'offline' / 'pretrained.pth'
    p.add_argument('--offline-ckpt', type=Path, default=default_ckpt)
    p.add_argument('--beam-width', type=int, default=15)
    p.add_argument('--no-easyocr', action='store_true')
    p.add_argument('--gpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    # 1. segment
    meta = run_segmentation(args.image, args.out_dir, use_easyocr=not args.no_easyocr, gpu=args.gpu)

    # 2. load offline model
    offline_model, offline_vocab = load_offline_model(args.offline_ckpt, device)

    # 3. infer on crops
    results = infer_on_crops(args.out_dir, offline_model, offline_vocab, device=device, beam_width=args.beam_width)

    # 4. save results
    (args.out_dir / 'results.json').write_text(json.dumps({'results': results, 'meta_method': meta.get('method')}, indent=2), encoding='utf-8')
    print('Saved results for', len(results), 'crops to', args.out_dir / 'results.json')


if __name__ == '__main__':
    main()
