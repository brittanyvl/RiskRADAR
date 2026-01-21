"""
taxonomy/config.py
------------------
Configuration for CICTT-anchored causal taxonomy.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Paths - data is within taxonomy module
PROJECT_ROOT = Path(__file__).parent.parent
TAXONOMY_DATA_DIR = Path(__file__).parent / "data"
TAXONOMY_REVIEW_DIR = TAXONOMY_DATA_DIR / "review"
CHUNKS_JSONL_PATH = PROJECT_ROOT / "extraction" / "json_data" / "chunks.jsonl"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings_data"

# Ensure directories exist
TAXONOMY_DATA_DIR.mkdir(exist_ok=True)
TAXONOMY_REVIEW_DIR.mkdir(exist_ok=True)


@dataclass
class TaxonomyConfig:
    """Configuration for CICTT taxonomy mapping."""

    # Version tracking
    version: str = "2.0.0"

    # Embedding model for similarity matching
    embedding_model: str = "NASA-AIML/MIKA_Custom_IR"

    # Sections to analyze for causal factors (strict filtering)
    # These sections contain the actual cause determination
    causal_sections: list = field(default_factory=lambda: [
        "PROBABLE CAUSE",
        "ANALYSIS",
        "CONCLUSIONS",
        "FINDINGS",
    ])

    # Sections to EXCLUDE (appendices, recommendations, factual only)
    excluded_sections: list = field(default_factory=lambda: [
        "APPENDIX",
        "APPENDICES",
        "RECOMMENDATIONS",
        "FACTUAL INFORMATION",
        "AIRCRAFT INFORMATION",
        "METEOROLOGICAL INFORMATION",
        "WRECKAGE",
        "TESTS AND RESEARCH",
        "SURVIVAL ASPECTS",
    ])

    # Minimum similarity threshold for category assignment
    min_similarity_threshold: float = 0.40

    # Top-k categories to consider per chunk
    top_k_categories: int = 3

    # Minimum token count for meaningful chunks
    min_chunk_tokens: int = 100

    # Multi-cause: maximum categories per report
    max_categories_per_report: int = 5


@dataclass
class ReviewConfig:
    """Configuration for human review process."""

    # Number of sample chunks per category for review
    samples_per_category: int = 5

    # Include full chunk text (not truncated)
    full_chunk_text: bool = True

    # Include report metadata
    include_report_metadata: bool = True

    # Output format for review
    output_format: str = "html"  # html, csv, or json


# Default configurations
TAXONOMY_CONFIG = TaxonomyConfig()
REVIEW_CONFIG = ReviewConfig()
