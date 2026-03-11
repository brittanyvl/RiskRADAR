"""Canonical search result type used by all search layers."""

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Rich search result combining Qdrant payload, chunk text, and SQLite metadata."""

    # Identity
    chunk_id: str = ""
    report_id: str = ""

    # Scoring
    score: float = 0.0
    rank: int = 0
    source: str = ""  # "bm25" | "semantic" | "both"

    # Chunk content (from JSONL)
    chunk_text: str = ""
    section_name: str = ""

    # Report metadata (from Qdrant payload or SQLite fallback)
    title: str = ""
    accident_date: str = ""
    location: str = ""

    # Taxonomy
    l1_categories: list[str] = field(default_factory=list)
    l2_subcategories: list[str] = field(default_factory=list)

    # Direct access
    pdf_url: str = ""

    # Aircraft (from SQLite report_features)
    aircraft_category: str = ""
