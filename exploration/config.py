"""
exploration/config.py
---------------------
Configuration for freeform topic exploration using BERTopic.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
EXPLORATION_DATA_DIR = Path(__file__).parent / "data"
EXPLORATION_VIZ_DIR = EXPLORATION_DATA_DIR / "visualizations"
CHUNKS_JSONL_PATH = PROJECT_ROOT / "extraction" / "json_data" / "chunks.jsonl"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings_data"

# Ensure directories exist
EXPLORATION_DATA_DIR.mkdir(exist_ok=True)
EXPLORATION_VIZ_DIR.mkdir(exist_ok=True)


@dataclass
class BERTopicConfig:
    """Configuration for BERTopic topic discovery."""

    # Version tracking
    version: str = "1.0.0"

    # Embedding model (use pre-generated MIKA embeddings)
    embedding_model: str = "NASA-AIML/MIKA_Custom_IR"
    use_precomputed_embeddings: bool = True

    # Dimensionality reduction (UMAP)
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    umap_random_state: int = 42

    # Clustering (HDBSCAN)
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 5
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"

    # Topic representation (c-TF-IDF)
    top_n_words: int = 10
    n_gram_range: tuple = (1, 2)
    min_topic_size: int = 20

    # Hierarchical topics
    hierarchical_reduction: bool = True
    nr_topics: int | None = None

    # Filtering - sections to include (broader for exploration)
    filter_sections: list = field(default_factory=lambda: [
        "PROBABLE CAUSE",
        "ANALYSIS",
        "CONCLUSIONS",
        "FINDINGS",
        "ANALYSIS AND CONCLUSIONS",
    ])

    # Minimum token count for chunks
    min_chunk_tokens: int = 100


# Default configuration
BERTOPIC_CONFIG = BERTopicConfig()
