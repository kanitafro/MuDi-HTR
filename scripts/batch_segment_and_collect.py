"""Batch-segment images in a folder and collect low-confidence crops.

Usage:
  python scripts/batch_segment_and_collect.py --input-dir images/ --out debug_batch --threshold 0.5

This imports helper functions from `scripts/segment_and_infer.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.segment_and_infer import run_segmentation, load_offline_model, infer_on_crops


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, default=Path('debug_batch'))
    p.add_argument('--offline-ckpt', type=Path, default=project_root / 'models' / 'checkpoints' / 'offline' / 'finetuned.pth')
    p.add_argument('--threshold', type=float, default=0.6)
    p.add_argument('--no-easyocr', action='store_true')
    args = p.parse_args()

    device = 'cpu'
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.offline_ckpt.exists():
        print('Offline checkpoint not found:', args.offline_ckpt)
        return

    offline_model, offline_vocab = load_offline_model(args.offline_ckpt, device='cpu')

    aggregated = []
    for img_path in sorted(args.input_dir.glob('*')):
        if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
            continue
        print('Processing', img_path)
        out_dir = args.out_dir / img_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            meta = run_segmentation(img_path, out_dir, use_easyocr=not args.no_easyocr, gpu=False)
            results = infer_on_crops(out_dir, offline_model, offline_vocab, device='cpu', beam_width=12)
            low = [r for r in results if r.get('conf', -1.0) < args.threshold]
            aggregated.append({'image': str(img_path), 'meta_method': meta.get('method'), 'num_crops': len(results), 'low_conf': low})
            (out_dir / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        except Exception as e:
            print('Error processing', img_path, e)

    (args.out_dir / 'aggregated.json').write_text(json.dumps(aggregated, indent=2), encoding='utf-8')
    print('Saved aggregated results to', args.out_dir / 'aggregated.json')


if __name__ == '__main__':
    main()
