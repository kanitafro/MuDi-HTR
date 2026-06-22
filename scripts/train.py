"""Offline dataloader smoke test for MuDi-HTR."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.offline import OfflineHandwritingDataset, create_offline_dataloader


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Quick smoke test for offline data pipeline.")
	parser.add_argument("--dataset", default="to-be/OpenHand-Synth", help="Dataset name used in preprocessing.")
	parser.add_argument("--split", default="train", help="Processed split name.")
	parser.add_argument("--root", default="data/processed/offline", help="Processed offline root directory.")
	parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the smoke test.")
	parser.add_argument("--max-samples", type=int, default=128, help="Limit sample count for quick testing.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	root = Path(args.root)

	encoder = OfflineHandwritingDataset.build_text_encoder(
		root_dir=root,
		split=args.split,
		dataset_name=args.dataset,
		max_samples=args.max_samples,
	)

	dataloader = create_offline_dataloader(
		root_dir=root,
		split=args.split,
		dataset_name=args.dataset,
		text_encoder=encoder,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=0,
		max_samples=args.max_samples,
	)

	batch = next(iter(dataloader))
	print(f"Loaded split: {args.split}")
	print(f"Encoder classes (CTC): {encoder.num_classes}")
	print(f"Image batch shape: {tuple(batch['images'].shape)}")
	print(f"Targets shape: {tuple(batch['targets'].shape)}")
	print(f"Target lengths: {batch['target_lengths'].tolist()}")


if __name__ == "__main__":
	main()
