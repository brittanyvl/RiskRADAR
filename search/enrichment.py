"""Result enrichment — combines Qdrant results with chunk text and SQLite metadata."""

import json
import logging
from pathlib import Path

from .result_types import SearchResult
from .config import SEARCH_CONFIG

logger = logging.getLogger(__name__)


class ChunkIndex:
    """In-memory index mapping chunk_id -> chunk_text from JSONL."""

    def __init__(self, chunks_jsonl_path: Path = None):
        self._index: dict[str, str] = {}
        path = chunks_jsonl_path or SEARCH_CONFIG.chunks_jsonl_path
        self._build(path)

    def _build(self, path: Path) -> None:
        logger.info(f"Building chunk text index from {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                self._index[chunk["chunk_id"]] = chunk.get("chunk_text", "")
        logger.info(f"Chunk index built: {len(self._index):,} entries")

    def get(self, chunk_id: str) -> str:
        return self._index.get(chunk_id, "")

    @property
    def size(self) -> int:
        return len(self._index)


class ResultEnricher:
    """Converts raw search result dicts into rich SearchResult objects."""

    def __init__(self, chunk_index: ChunkIndex):
        self._chunks = chunk_index

    def enrich(self, raw_results: list[dict]) -> list[SearchResult]:
        """Bulk-enrich a list of raw search dicts with chunk text and metadata."""
        if not raw_results:
            return []

        # Fetch report metadata for any results missing payload fields
        report_ids = list({r["report_id"] for r in raw_results})
        metadata_map = self._fetch_report_metadata(report_ids)

        enriched = []
        for raw in raw_results:
            rid = raw["report_id"]
            meta = metadata_map.get(rid, {})

            enriched.append(SearchResult(
                chunk_id=raw["chunk_id"],
                report_id=rid,
                score=raw.get("score", 0.0),
                rank=raw.get("rank", 0),
                source=raw.get("source", ""),
                chunk_text=self._chunks.get(raw["chunk_id"]),
                section_name=raw.get("section_name", ""),
                # Prefer Qdrant payload, fall back to SQLite
                title=raw.get("title") or meta.get("title", rid),
                accident_date=raw.get("accident_date") or meta.get("accident_date", ""),
                location=raw.get("location") or meta.get("location", ""),
                l1_categories=raw.get("l1_categories") or meta.get("l1_categories", []),
                l2_subcategories=raw.get("l2_subcategories", []),
                pdf_url=raw.get("pdf_url", ""),
                aircraft_category=meta.get("aircraft_category", ""),
            ))
        return enriched

    def _fetch_report_metadata(self, report_ids: list[str]) -> dict[str, dict]:
        """Bulk-fetch report metadata from SQLite for enrichment."""
        if not report_ids:
            return {}

        from analytics.queries.shared import get_connection

        conn = get_connection()
        try:
            placeholders = ",".join("?" * len(report_ids))

            rows = conn.execute(f"""
                SELECT
                    r.filename AS report_id,
                    r.title,
                    r.accident_date,
                    r.location,
                    f.aircraft_category
                FROM reports r
                LEFT JOIN report_features f ON r.filename = f.report_id
                WHERE r.filename IN ({placeholders})
            """, report_ids).fetchall()

            # Fetch L1 categories for BM25-only results (no Qdrant payload)
            tax_rows = conn.execute(f"""
                SELECT report_id, GROUP_CONCAT(DISTINCT category_code) AS cats
                FROM report_taxonomy
                WHERE level = 'L1' AND report_id IN ({placeholders})
                GROUP BY report_id
            """, report_ids).fetchall()
        finally:
            conn.close()

        tax_map = {r[0]: r[1].split(",") if r[1] else [] for r in tax_rows}

        result = {}
        for row in rows:
            rid = row[0]
            result[rid] = {
                "title": row[1] or "",
                "accident_date": row[2] or "",
                "location": row[3] or "",
                "aircraft_category": row[4] or "",
                "l1_categories": tax_map.get(rid, []),
            }
        return result
