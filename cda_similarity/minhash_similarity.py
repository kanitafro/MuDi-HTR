# cda_similarity/minhash_similarity.py
"""MinHash + LSH for document similarity."""

import json
import time
from typing import List, Dict, Tuple, Optional
from datasketch import MinHash, MinHashLSH
import numpy as np
from pathlib import Path


class TextSimilarityIndex:
    """
    Builds an index of text documents using MinHash + LSH.
    Supports insertion, query, and serialization.
    """
    
    def __init__(self, threshold: float = 0.5, num_perm: int = 128, shingle_size: int = 3):
        """
        Args:
            threshold: Jaccard similarity threshold for LSH (documents with similarity > threshold are candidates)
            num_perm: Number of permutations for MinHash
            shingle_size: Character n-gram size (default 3)
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        
        # LSH index
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes: Dict[str, MinHash] = {}  # id -> MinHash
        self.metadata: Dict[str, Dict] = {}      # id -> {text, timestamp, ...}
        self.next_id = 0
    
    @staticmethod
    def shingle_text(text: str, k: int = 3) -> set:
        """Convert text into a set of character k-shingles."""
        if len(text) < k:
            return {text}
        return {text[i:i+k] for i in range(len(text) - k + 1)}
    
    def compute_minhash(self, text: str) -> MinHash:
        """Compute MinHash signature for a text."""
        shingles = self.shingle_text(text, self.shingle_size)
        m = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))
        return m
    
    def add_document(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Insert a new document into the index.
        
        Args:
            text: Document text
            metadata: Additional info (timestamp, user, etc.)
        
        Returns:
            doc_id: Unique identifier for the document
        """
        doc_id = f"doc_{self.next_id}"
        self.next_id += 1
        
        mh = self.compute_minhash(text)
        self.minhashes[doc_id] = mh
        self.lsh.insert(doc_id, mh)
        
        self.metadata[doc_id] = {
            'text': text,
            'timestamp': metadata.get('timestamp', time.time()) if metadata else time.time(),
            **(metadata or {})
        }
        return doc_id
    
    def query(self, text: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Find similar documents for a query text.
        
        Returns:
            List of (doc_id, jaccard_similarity, metadata) sorted by similarity descending.
        """
        if not self.minhashes:
            return []
        
        mh = self.compute_minhash(text)
        # Get candidate IDs from LSH
        candidate_ids = self.lsh.query(mh)
        
        if not candidate_ids:
            return []
        
        # Compute exact Jaccard similarities
        similarities = []
        for doc_id in candidate_ids:
            # Compute exact Jaccard between query and stored MinHash
            sim = mh.jaccard(self.minhashes[doc_id])
            # Only keep those above threshold (optional)
            if sim >= self.threshold:
                similarities.append((doc_id, sim, self.metadata[doc_id]))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, filepath: Path):
        """Save index to disk."""
        data = {
            'threshold': self.threshold,
            'num_perm': self.num_perm,
            'shingle_size': self.shingle_size,
            'next_id': self.next_id,
            'metadata': self.metadata,
            'minhashes': {k: v.digest().hex() for k, v in self.minhashes.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: Path):
        """Load index from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.threshold = data['threshold']
        self.num_perm = data['num_perm']
        self.shingle_size = data['shingle_size']
        self.next_id = data['next_id']
        self.metadata = data['metadata']
        
        # Rebuild MinHash objects from hex digests
        self.minhashes = {}
        for doc_id, digest_hex in data['minhashes'].items():
            m = MinHash(num_perm=self.num_perm)
            m.deserialize(bytes.fromhex(digest_hex))
            self.minhashes[doc_id] = m
            self.lsh.insert(doc_id, m)


def evaluate_recall(index: TextSimilarityIndex, queries: List[Tuple[str, str]], exact_threshold: float = 0.5):
    """
    Evaluate recall of LSH compared to exact Jaccard.
    
    Args:
        index: TextSimilarityIndex instance
        queries: List of (query_text, expected_similar_doc_id)
        exact_threshold: Threshold for considering two docs similar
    
    Returns:
        recall: Fraction of queries where LSH returned the expected doc
    """
    recall_count = 0
    for query_text, expected_id in queries:
        results = index.query(query_text, top_k=1)
        if results and results[0][0] == expected_id:
            recall_count += 1
    return recall_count / len(queries) if queries else 0.0