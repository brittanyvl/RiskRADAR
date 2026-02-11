"""BM25 keyword search index built from chunks.jsonl."""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .config import SEARCH_CONFIG
from .preprocessing import tokenize_for_bm25

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.chunk_ids: list[str] = []
        self.report_ids: list[str] = []
        self.section_names: list[str] = []
        self.corpus_size: int = 0

    def build(self, chunks_jsonl_path: Path = None) -> None:
        """Build BM25 index from chunks.jsonl."""
        path = chunks_jsonl_path or SEARCH_CONFIG.chunks_jsonl_path
        logger.info(f"Building BM25 index from {path}")
        start = time.perf_counter()

        chunk_ids = []
        report_ids = []
        section_names = []
        tokenized_corpus = []

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                chunk = json.loads(line)
                chunk_ids.append(chunk["chunk_id"])
                report_ids.append(chunk["report_id"])
                section_names.append(chunk.get("section_name", ""))
                tokens = tokenize_for_bm25(chunk["chunk_text"])
                tokenized_corpus.append(tokens)

                if line_num % 5000 == 0:
                    logger.info(f"  Tokenized {line_num:,} chunks...")

        self.chunk_ids = chunk_ids
        self.report_ids = report_ids
        self.section_names = section_names
        self.corpus_size = len(chunk_ids)

        logger.info(f"Building BM25Okapi index over {self.corpus_size:,} chunks...")
        self.index = BM25Okapi(tokenized_corpus)

        elapsed = time.perf_counter() - start
        avg_tokens = np.mean([len(t) for t in tokenized_corpus])
        logger.info(
            f"BM25 index built in {elapsed:.1f}s "
            f"({self.corpus_size:,} chunks, avg {avg_tokens:.0f} tokens/chunk)"
        )

    def save(self, path: Path = None) -> None:
        """Save index to pickle file."""
        path = path or SEARCH_CONFIG.bm25_index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"BM25 index saved to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path = None) -> "BM25Index":
        """Load index from pickle file."""
        path = path or SEARCH_CONFIG.bm25_index_path
        logger.info(f"Loading BM25 index from {path}")
        start = time.perf_counter()
        with open(path, "rb") as f:
            obj = pickle.load(f)
        elapsed = time.perf_counter() - start
        logger.info(f"BM25 index loaded in {elapsed:.1f}s ({obj.corpus_size:,} chunks)")
        return obj

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """
        Search the index.

        Returns:
            List of dicts with keys: chunk_id, report_id, section_name, score, rank
        """
        if self.index is None:
            raise RuntimeError("BM25 index not built or loaded. Call build() or load() first.")

        tokens = tokenize_for_bm25(query)
        scores = self.index.get_scores(tokens)

        # Get top-k indices by score (descending)
        top_indices = np.argsort(scores)[::-1][:limit]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            if score <= 0:
                break
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "report_id": self.report_ids[idx],
                "section_name": self.section_names[idx],
                "score": score,
                "rank": rank,
            })

        return results
