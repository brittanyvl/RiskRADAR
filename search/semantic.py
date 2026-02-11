"""Semantic search using Qdrant vector database."""

import logging
import time
from typing import Optional

from .config import SEARCH_CONFIG

logger = logging.getLogger(__name__)


class SemanticSearcher:
    def __init__(self, model_name: str = None, collection_name: str = None):
        self.model_name = model_name or SEARCH_CONFIG.default_model
        self.collection_name = collection_name or SEARCH_CONFIG.default_collection
        self._model = None
        self._client = None

    def _ensure_loaded(self):
        """Lazy-load model and Qdrant client on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from qdrant_client import QdrantClient
            from embeddings.config import get_model_config
            from riskradar.config import get_qdrant_config

            config = get_model_config(self.model_name)
            qdrant_config = get_qdrant_config()

            logger.info(f"Loading embedding model: {config.model_id}")
            self._model = SentenceTransformer(config.model_id)

            self._client = QdrantClient(
                url=qdrant_config["url"],
                api_key=qdrant_config["api_key"],
            )
            logger.info(f"Connected to Qdrant collection: {self.collection_name}")

    def search(self, query: str, limit: int = 20, filters: dict = None) -> list[dict]:
        """
        Search Qdrant.

        Args:
            query: Search query text.
            limit: Max results to return.
            filters: Optional dict with keys like 'l1_categories', 'report_id',
                     'date_from', 'date_to'.

        Returns:
            List of dicts with keys: chunk_id, report_id, section_name, score, rank
        """
        self._ensure_loaded()

        start = time.perf_counter()
        query_vector = self._model.encode(query).tolist()
        embed_time = (time.perf_counter() - start) * 1000

        qdrant_filter = self._build_filter(filters) if filters else None

        start = time.perf_counter()
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=qdrant_filter,
        )
        search_time = (time.perf_counter() - start) * 1000

        results = []
        for rank, hit in enumerate(response.points, 1):
            payload = hit.payload or {}
            results.append({
                "chunk_id": payload.get("chunk_id", ""),
                "report_id": payload.get("report_id", ""),
                "section_name": payload.get("section_name", ""),
                "score": hit.score,
                "rank": rank,
            })

        logger.debug(
            f"Semantic search: {len(results)} results "
            f"(embed={embed_time:.0f}ms, search={search_time:.0f}ms)"
        )
        return results

    def _build_filter(self, filters: dict):
        """Convert filter dict to Qdrant Filter objects."""
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, Range

        must_conditions = []

        if "l1_categories" in filters:
            cats = filters["l1_categories"]
            if isinstance(cats, str):
                cats = [cats]
            must_conditions.append(
                FieldCondition(key="l1_categories", match=MatchAny(any=cats))
            )

        if "report_id" in filters:
            must_conditions.append(
                FieldCondition(key="report_id", match=MatchValue(value=filters["report_id"]))
            )

        if "date_from" in filters or "date_to" in filters:
            range_kwargs = {}
            if "date_from" in filters:
                range_kwargs["gte"] = filters["date_from"]
            if "date_to" in filters:
                range_kwargs["lte"] = filters["date_to"]
            must_conditions.append(
                FieldCondition(key="accident_date", range=Range(**range_kwargs))
            )

        if not must_conditions:
            return None

        return Filter(must=must_conditions)
