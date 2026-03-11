"""Hybrid search combining BM25 and semantic search with RRF fusion."""

import logging

from .bm25 import BM25Index
from .semantic import SemanticSearcher
from .config import SearchConfig, SEARCH_CONFIG

logger = logging.getLogger(__name__)


class HybridSearcher:
    def __init__(
        self,
        bm25_index: BM25Index = None,
        semantic_searcher: SemanticSearcher = None,
        config: SearchConfig = None,
    ):
        self.config = config or SEARCH_CONFIG
        self._bm25 = bm25_index
        self._semantic = semantic_searcher

    def _ensure_bm25(self):
        if self._bm25 is None:
            self._bm25 = BM25Index.load()

    def _ensure_semantic(self):
        if self._semantic is None:
            self._semantic = SemanticSearcher()

    def search(
        self,
        query: str,
        limit: int = 20,
        mode: str = "hybrid",
        filters: dict = None,
    ) -> list[dict]:
        """
        Search with specified mode.

        Args:
            query: Search query text.
            limit: Max results to return.
            mode: "bm25", "semantic", or "hybrid".
            filters: Optional filter dict (semantic/hybrid modes only).

        Returns:
            List of dicts with keys:
                chunk_id, report_id, section_name, score, rank, source
        """
        if mode == "bm25":
            self._ensure_bm25()
            results = self._bm25.search(query, limit=limit)
            for r in results:
                r["source"] = "bm25"
            return results

        elif mode == "semantic":
            self._ensure_semantic()
            results = self._semantic.search(query, limit=limit, filters=filters)
            for r in results:
                r["source"] = "semantic"
            return results

        else:  # hybrid
            self._ensure_bm25()
            self._ensure_semantic()
            # Fetch more from each for better fusion coverage
            fetch_limit = limit * 2
            bm25_results = self._bm25.search(query, limit=fetch_limit)
            sem_results = self._semantic.search(
                query, limit=fetch_limit, filters=filters
            )
            return self._rrf_fuse(bm25_results, sem_results, limit)

    def _rrf_fuse(
        self, bm25_results: list, semantic_results: list, limit: int
    ) -> list[dict]:
        """Weighted Reciprocal Rank Fusion."""
        k = self.config.rrf_k
        w_bm25 = self.config.bm25_weight
        w_sem = self.config.semantic_weight

        scores = {}  # chunk_id -> {rrf_score, metadata}

        def _new_entry(r):
            return {
                "chunk_id": r["chunk_id"],
                "report_id": r["report_id"],
                "section_name": r["section_name"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "sem_rank": None,
                "title": r.get("title", ""),
                "location": r.get("location", ""),
                "pdf_url": r.get("pdf_url", ""),
                "accident_date": r.get("accident_date", ""),
                "l1_categories": r.get("l1_categories", []),
                "l2_subcategories": r.get("l2_subcategories", []),
            }

        for r in bm25_results:
            cid = r["chunk_id"]
            if cid not in scores:
                scores[cid] = _new_entry(r)
            scores[cid]["rrf_score"] += w_bm25 / (k + r["rank"])
            scores[cid]["bm25_rank"] = r["rank"]

        for r in semantic_results:
            cid = r["chunk_id"]
            if cid not in scores:
                scores[cid] = _new_entry(r)
            else:
                # Merge payload from semantic side into existing BM25 entry
                entry = scores[cid]
                if not entry.get("title"):
                    for field in ("title", "location", "pdf_url", "accident_date",
                                  "l1_categories", "l2_subcategories"):
                        entry[field] = r.get(field, entry[field])
            scores[cid]["rrf_score"] += w_sem / (k + r["rank"])
            scores[cid]["sem_rank"] = r["rank"]

        # Sort by RRF score descending
        fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)[
            :limit
        ]

        # Add final rank and source info
        for i, r in enumerate(fused):
            r["rank"] = i + 1
            r["score"] = r.pop("rrf_score")
            if r["bm25_rank"] is not None and r["sem_rank"] is not None:
                r["source"] = "both"
            elif r["bm25_rank"] is not None:
                r["source"] = "bm25"
            else:
                r["source"] = "semantic"

        return fused
