"""Test scalability of MinHash + LSH on synthetic data."""

import argparse
import sys
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from cda_similarity.minhash_similarity import TextSimilarityIndex
from cda_similarity.synthetic_data import generate_journal_entries

def run_scalability_test(sizes: List[int], num_queries: int = 100):
    """
    Test query time and memory for different dataset sizes.
    """
    results = []
    
    for size in sizes:
        print(f"Testing size {size}...")
        # Generate synthetic data with more duplicates for better recall testing
        entries = generate_journal_entries(size, duplicate_fraction=0.3)
        
        # Build index with lower threshold
        index = TextSimilarityIndex(threshold=0.3, num_perm=128, shingle_size=3)
        start = time.time()
        for entry in entries:
            index.add_document(entry['text'], {'timestamp': entry['timestamp']})
        build_time = time.time() - start
        
        # Query time
        query_times = []
        for _ in range(num_queries):
            query_text = random.choice(entries)['text']
            start_q = time.time()
            _ = index.query(query_text, top_k=3)
            query_times.append(time.time() - start_q)
        
        avg_query_time_ms = np.mean(query_times) * 1000
        
        # Estimate memory (rough)
        memory_mb = (size * 128 * 4) / (1024 * 1024)
        
        results.append({
            'size': size,
            'avg_query_time_ms': avg_query_time_ms,
            'build_time_sec': build_time,
            'memory_mb': memory_mb
        })
        
        print(f"  Avg query time: {avg_query_time_ms:.2f} ms")
        print(f"  Build time: {build_time:.2f} s")
    
    return results


def plot_scalability(results: List[Dict], save_path: Path = None):
    """Plot query time vs dataset size."""
    sizes = [r['size'] for r in results]
    times = [r['avg_query_time_ms'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Query time
    axes[0].plot(sizes, times, marker='o', linestyle='-', linewidth=2, color='steelblue')
    axes[0].set_xlabel('Number of Documents')
    axes[0].set_ylabel('Average Query Time (ms)')
    axes[0].set_title('LSH Query Time Scalability')
    axes[0].grid(True, alpha=0.3)
    
    # Build time
    build_times = [r['build_time_sec'] for r in results]
    axes[1].plot(sizes, build_times, marker='s', linestyle='-', linewidth=2, color='seagreen')
    axes[1].set_xlabel('Number of Documents')
    axes[1].set_ylabel('Build Time (seconds)')
    axes[1].set_title('Index Build Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compare_recall(num_docs: int = 1000, num_queries: int = 100):
    """
    Compare LSH recall against exact Jaccard with improved methodology.
    """
    print("\n" + "="*60)
    print("RECALL EVALUATION")
    print("="*60)
    
    # Generate data with duplicates
    entries = generate_journal_entries(num_docs, duplicate_fraction=0.3)
    index = TextSimilarityIndex(threshold=0.3, num_perm=128, shingle_size=3)
    
    # Store doc ids
    doc_ids = []
    for entry in entries:
        doc_id = index.add_document(entry['text'], {'timestamp': entry['timestamp']})
        doc_ids.append(doc_id)
    
    print(f"Index built with {len(doc_ids)} documents")
    
    # For each query, compute exact Jaccard and check if LSH returns at least one similar doc
    hits = 0
    total = 0
    exact_similar_total = 0
    
    query_indices = random.sample(range(len(entries)), min(num_queries, len(entries)))
    
    for idx in tqdm(query_indices, desc="Evaluating recall"):
        query_text = entries[idx]['text']
        query_mh = index.compute_minhash(query_text)
        
        # Exact Jaccard with all docs (exclude self)
        exact_sims = []
        for doc_id in doc_ids:
            if doc_id != doc_ids[idx]:
                sim = query_mh.jaccard(index.minhashes[doc_id])
                exact_sims.append((doc_id, sim))
        
        # Find docs with similarity > 0.3 (ground truth similar)
        similar_docs = [doc_id for doc_id, sim in exact_sims if sim > 0.3]
        
        if similar_docs:
            exact_similar_total += 1
            # LSH query
            lsh_results = index.query(query_text, top_k=5)
            lsh_ids = {r[0] for r in lsh_results}
            # Check if LSH found at least one similar doc
            if set(similar_docs) & lsh_ids:
                hits += 1
        else:
            # No similar docs: LSH should return nothing (or only false positives)
            lsh_results = index.query(query_text, top_k=1)
            if not lsh_results:
                hits += 1
            # else: false positive, but we'll still count as correct if no similar docs exist
            # (LSH might return some candidate with low similarity)
        
        total += 1
    
    recall = hits / total if total > 0 else 0.0
    print(f"\nRecall (LSH vs exact Jaccard): {recall:.3f} (based on {total} queries)")
    print(f"  Documents with at least one similar doc: {exact_similar_total}/{total}")
    
    return recall


def test_different_thresholds(num_docs: int = 1000, num_queries: int = 100):
    """
    Test how LSH threshold affects recall.
    """
    print("\n" + "="*60)
    print("THRESHOLD TUNING")
    print("="*60)
    
    thresholds = [0.2, 0.3, 0.4, 0.5]
    results = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        entries = generate_journal_entries(num_docs, duplicate_fraction=0.3)
        index = TextSimilarityIndex(threshold=threshold, num_perm=128, shingle_size=3)
        
        doc_ids = []
        for entry in entries:
            doc_id = index.add_document(entry['text'], {'timestamp': entry['timestamp']})
            doc_ids.append(doc_id)
        
        # Query time
        query_times = []
        for _ in range(50):
            query_text = random.choice(entries)['text']
            start_q = time.time()
            _ = index.query(query_text, top_k=3)
            query_times.append(time.time() - start_q)
        avg_time = np.mean(query_times) * 1000
        
        # Recall
        hits = 0
        total = 0
        query_indices = random.sample(range(len(entries)), min(num_queries, len(entries)))
        for idx in query_indices:
            query_text = entries[idx]['text']
            query_mh = index.compute_minhash(query_text)
            exact_sims = []
            for doc_id in doc_ids:
                if doc_id != doc_ids[idx]:
                    sim = query_mh.jaccard(index.minhashes[doc_id])
                    exact_sims.append((doc_id, sim))
            similar_docs = [doc_id for doc_id, sim in exact_sims if sim > threshold]
            if similar_docs:
                lsh_results = index.query(query_text, top_k=5)
                lsh_ids = {r[0] for r in lsh_results}
                if set(similar_docs) & lsh_ids:
                    hits += 1
            else:
                lsh_results = index.query(query_text, top_k=1)
                if not lsh_results:
                    hits += 1
            total += 1
        
        recall = hits / total if total > 0 else 0.0
        results.append({
            'threshold': threshold,
            'recall': recall,
            'avg_query_time_ms': avg_time
        })
        print(f"  Recall: {recall:.3f}, Query time: {avg_time:.2f} ms")
    
    # Plot threshold vs recall
    plt.figure(figsize=(10, 6))
    plt.plot([r['threshold'] for r in results], [r['recall'] for r in results], 
             marker='o', linestyle='-', linewidth=2, color='purple')
    plt.xlabel('LSH Threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs LSH Threshold')
    plt.grid(True, alpha=0.3)
    plt.savefig('../experiments/figures/minhash_recall_vs_threshold1.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    # Run scalability tests
    sizes = [1000, 5000, 10000, 50000]
    results = run_scalability_test(sizes, num_queries=50)
    
    # Plot scalability
    plot_scalability(results, save_path=Path("../experiments/figures/minhash_scalability_plot1.png"))
    
    # Test recall with improved methodology
    recall = compare_recall(num_docs=1000, num_queries=100)
    
    # Test different thresholds
    threshold_results = test_different_thresholds(num_docs=1000, num_queries=100)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Best threshold: {max(threshold_results, key=lambda x: x['recall'])['threshold']}")
    print(f"Best recall: {max(threshold_results, key=lambda x: x['recall'])['recall']:.3f}")