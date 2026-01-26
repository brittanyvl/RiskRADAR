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

    # =========================================================================
    # Level 1 (CICTT) Configuration
    # =========================================================================

    # Minimum similarity threshold for L1 category assignment
    min_similarity_threshold: float = 0.40

    # Top-k categories to consider per chunk at L1
    top_k_categories: int = 3

    # Minimum token count for meaningful chunks
    min_chunk_tokens: int = 100

    # Multi-cause: maximum L1 categories per report
    max_categories_per_report: int = 5

    # =========================================================================
    # Level 2 (Subcategory) Configuration
    # =========================================================================

    # Enable hierarchical (two-pass) classification
    enable_l2_classification: bool = True

    # Minimum L2 similarity threshold (can be lower since we're within parent)
    min_l2_similarity_threshold: float = 0.35

    # Top-k subcategories per L1 parent per chunk
    top_k_subcategories: int = 2

    # Maximum subcategories per L1 category per report
    max_subcategories_per_parent: int = 3

    # Minimum combined confidence (L1_sim * L2_sim) for assignment
    min_combined_confidence: float = 0.20

    # L1 categories that have defined subcategories (hierarchical mapping)
    # Only these will have L2 classification applied
    l2_enabled_categories: list = field(default_factory=lambda: [
        "LOC-I",    # Loss of Control - Inflight (6 subcategories)
        "CFIT",     # Controlled Flight Into Terrain (5 subcategories)
        "SCF-PP",   # System/Component Failure - Powerplant (4 subcategories)
        "SCF-NP",   # System/Component Failure - Non-Powerplant (5 subcategories)
        "ICE",      # Icing (3 subcategories)
        "FUEL",     # Fuel Related (3 subcategories)
        "WSTRW",    # Windshear or Thunderstorm (2 subcategories)
    ])

    # HFACS-applicable L1 categories (human factors can be co-assigned)
    hfacs_applicable_categories: list = field(default_factory=lambda: [
        "LOC-I",
        "CFIT",
        "FUEL",
        "UIMC",
        "LALT",
    ])


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
