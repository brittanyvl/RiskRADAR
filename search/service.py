"""SearchService facade — single entry point for the Streamlit search page."""

import logging
import time
from dataclasses import dataclass, field

from .hybrid import HybridSearcher
from .enrichment import ResultEnricher, ChunkIndex
from .result_types import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """Encapsulates all search parameters from the UI."""

    query: str = ""
    mode: str = "hybrid"  # "hybrid" | "semantic" | "bm25"
    l1_categories: list[str] = field(default_factory=list)
    l2_subcategories: list[str] = field(default_factory=list)
    aircraft_categories: list[str] = field(default_factory=list)
    date_from: str | None = None  # ISO date "YYYY-MM-DD"
    date_to: str | None = None


class SearchService:
    """
    Facade combining HybridSearcher + ResultEnricher + post-search filters.

    The view calls search() and receives fully enriched SearchResult objects.
    """

    def __init__(
        self,
        searcher: HybridSearcher,
        enricher: ResultEnricher,
    ):
        self._searcher = searcher
        self._enricher = enricher

    def search(
        self, query: SearchQuery, max_results: int = 30
    ) -> tuple[list[SearchResult], float]:
        """
        Execute search and return (all_results, elapsed_ms).

        Returns all post-filtered results (up to max_results). The view
        handles pagination by slicing the returned list.
        """
        start = time.perf_counter()

        # Build Qdrant-native filters (L1, L2, date)
        qdrant_filters = self._build_qdrant_filters(query)

        # Over-fetch when aircraft filter active to compensate for post-filter attrition
        needs_ac_filter = bool(query.aircraft_categories)
        fetch_limit = min(max_results * 3, 100) if needs_ac_filter else max_results

        raw = self._searcher.search(
            query.query,
            limit=fetch_limit,
            mode=query.mode,
            filters=qdrant_filters if qdrant_filters else None,
        )

        # Post-filter by aircraft category (SQLite-only field)
        if needs_ac_filter:
            raw = self._apply_aircraft_filter(raw, query.aircraft_categories)

        # Trim to max_results
        raw = raw[:max_results]

        # Enrich with chunk text and metadata
        enriched = self._enricher.enrich(raw)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Search complete: {len(enriched)} results in {elapsed_ms:.0f}ms "
            f"(mode={query.mode})"
        )
        return enriched, elapsed_ms

    def _build_qdrant_filters(self, query: SearchQuery) -> dict:
        filters = {}
        if query.l1_categories:
            filters["l1_categories"] = query.l1_categories
        if query.l2_subcategories:
            filters["l2_subcategories"] = query.l2_subcategories
        if query.date_from:
            filters["date_from"] = query.date_from
        if query.date_to:
            filters["date_to"] = query.date_to
        return filters

    def _apply_aircraft_filter(
        self, raw: list[dict], aircraft_categories: list[str]
    ) -> list[dict]:
        from analytics.queries.search_filters import filter_reports_by_aircraft

        report_ids = [r["report_id"] for r in raw]
        valid_ids = filter_reports_by_aircraft(report_ids, aircraft_categories)
        return [r for r in raw if r["report_id"] in valid_ids]
