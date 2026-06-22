"""Main preprocessing orchestrator for online handwriting datasets."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Preprocess online handwriting datasets")
    parser.add_argument('--dataset', choices=['didi', 'iam_ondb', 'all'], 
                       default='didi', help='Dataset to preprocess')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocessing even if data exists')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Online Handwriting Data Preprocessing")
    print("="*60)
    
    if args.dataset in ['didi', 'all']:
        print("\n📁 Preprocessing DIDI dataset...")
        from preprocessing.didi_preprocess import main as didi_main
        didi_main()
    
    if args.dataset in ['iam_ondb', 'all']:
        print("\n📁 Preprocessing IAM-OnDB dataset...")
        from preprocessing.iam_ondb_preprocess import main as iam_main
        iam_main()
    
    print("\n" + "="*60)
    print("✅ All preprocessing complete!")
    print("="*60)

if __name__ == "__main__":
    main()