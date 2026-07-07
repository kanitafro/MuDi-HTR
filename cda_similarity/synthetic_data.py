"""Generate synthetic journal entries for testing MinHash + LSH."""

import argparse
import random
import json
from typing import List, Dict
from pathlib import Path

# Common words for generating realistic-sounding sentences
COMMON_WORDS = [
    "today", "felt", "happy", "sad", "anxious", "tired", "energetic", "thought",
    "about", "work", "school", "friend", "family", "weather", "rain", "sun",
    "cloud", "meeting", "important", "really", "very", "quite", "somewhat",
    "I", "you", "we", "they", "he", "she", "it", "that", "this", "those",
    "was", "were", "am", "is", "are", "have", "had", "will", "would", "could",
    "should", "might", "must", "maybe", "probably", "definitely", "certainly",
    "good", "bad", "nice", "great", "terrible", "wonderful", "awful",
    "journey", "journal", "entry", "day", "week", "month", "year",
    "morning", "afternoon", "evening", "night", "yesterday", "tomorrow"
]

# Arguments
argparser = argparse.ArgumentParser(description="Generate synthetic journal entries for testing.")
argparser.add_argument('--num_entries', type=int, default=1000, help="Number of entries to generate.")
argparser.add_argument('--duplicate_fraction', type=float, default=0.2, help="Fraction of entries that are near-duplicates.")
argparser.add_argument('--min_words', type=int, default=5, help="Minimum number of words in each sentence.")
argparser.add_argument('--max_words', type=int, default=15, help="Maximum number of words in each sentence.")
argparser.add_argument('--output', type=str, default="synthetic_entries.json", help="Output JSON file path.")
args = argparser.parse_args()

def generate_sentence(min_words: int = args.min_words, max_words: int = args.max_words) -> str:
    """Generate a random sentence."""
    length = random.randint(min_words, max_words)
    words = random.sample(COMMON_WORDS, k=length)
    return ' '.join(words).capitalize() + '.'


def generate_journal_entries(num_entries: int, duplicate_fraction: float = 0.2) -> List[Dict]:
    """
    Generate synthetic journal entries.
    
    Args:
        num_entries: Number of entries to generate.
        duplicate_fraction: Fraction of entries that are near-duplicates (with slight variations).
    
    Returns:
        List of dicts with 'text' and 'timestamp' keys.
    """
    entries = []
    base_texts = [generate_sentence() for _ in range(int(num_entries * (1 - duplicate_fraction)))]
    
    for i in range(num_entries):
        # Decide if this entry is a duplicate or new
        if len(base_texts) > 0 and random.random() < duplicate_fraction:
            # Pick a random base text and modify slightly
            base = random.choice(base_texts)
            # Variation: change a few words
            words = base.split()
            if len(words) > 3:
                # Replace 1-2 random words
                for _ in range(random.randint(0, 2)):
                    if words:
                        idx = random.randint(0, len(words)-1)
                        words[idx] = random.choice(COMMON_WORDS)
                text = ' '.join(words)
            else:
                text = base
        else:
            text = generate_sentence()
            # Store base texts for future duplicates
            if len(base_texts) < num_entries:
                base_texts.append(text)
        
        entries.append({
            'text': text,
            'timestamp': random.randint(1600000000, 1700000000)  # random timestamp
        })
    
    return entries


def save_synthetic_data(entries: List[Dict], filepath: Path):
    """Save synthetic entries to JSON."""
    with open(filepath, 'w') as f:
        json.dump(entries, f, indent=2)


def load_synthetic_data(filepath: Path) -> List[Dict]:
    """Load synthetic entries from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":  
    

    # Generate and save entries for initial testing
    entries = generate_journal_entries(args.num_entries, duplicate_fraction=args.duplicate_fraction)
    save_synthetic_data(entries, Path(args.output))
    print(f"Generated {len(entries)} synthetic entries.")