"""
embeddings/config.py
--------------------
Configuration for embedding models and pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDINGS_DATA_DIR = PROJECT_ROOT / "embeddings_data"
CHUNKS_JSONL_PATH = PROJECT_ROOT / "extraction" / "json_data" / "chunks.jsonl"


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    name: str                    # Short name: 'minilm' or 'mika'
    model_id: str               # HuggingFace model ID
    collection_name: str        # Qdrant collection name
    expected_dimension: int     # Expected embedding dimension
    batch_size: int             # Batch size for CPU encoding


# Model registry
MODELS = {
    "minilm": ModelConfig(
        name="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="riskradar_minilm",
        expected_dimension=384,
        batch_size=32,
    ),
    "mika": ModelConfig(
        name="mika",
        model_id="NASA-AIML/MIKA_Custom_IR",
        collection_name="riskradar_mika",
        expected_dimension=768,
        batch_size=16,  # Larger model, smaller batches
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get configuration for a model by name.

    Args:
        model_name: 'minilm' or 'mika'

    Returns:
        ModelConfig instance

    Raises:
        ValueError: If model name not recognized
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODELS.keys())}"
        )
    return MODELS[model_name]


def get_parquet_path(model_name: str) -> Path:
    """Get the Parquet output path for a model."""
    return EMBEDDINGS_DATA_DIR / f"{model_name}_embeddings.parquet"


# Pipeline configuration
PIPELINE_CONFIG = {
    "version": "5.0.0",
    "distance_metric": "cosine",
    "upload_batch_size": 100,
    "retry_attempts": 3,
    "retry_delay_sec": 2.0,
}
