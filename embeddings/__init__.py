"""
embeddings/
-----------
Phase 5: Text embedding pipeline for RiskRADAR.

Generates embeddings using two models:
- MiniLM (384 dims): General purpose baseline
- MIKA (768 dims): Aerospace/aviation domain-specific

Stores embeddings locally in Parquet, then uploads to Qdrant Cloud.
"""

from .config import MODELS, get_model_config
from .models import EmbeddingModel, load_model

__all__ = [
    "MODELS",
    "get_model_config",
    "EmbeddingModel",
    "load_model",
]
