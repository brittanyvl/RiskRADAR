"""
embeddings/models.py
--------------------
Model wrapper for sentence-transformers embedding models.

Provides a consistent interface for loading models and generating embeddings.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import ModelConfig, get_model_config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence-transformers models with CPU optimization.

    Provides consistent interface for:
    - Loading models
    - Detecting embedding dimensions
    - Batch encoding with progress
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize and load a sentence-transformers model.

        Args:
            config: ModelConfig with model_id and settings

        Raises:
            RuntimeError: If model fails to load
        """
        self.config = config
        self.model_id = config.model_id
        self.batch_size = config.batch_size

        logger.info(f"Loading model: {self.model_id}")

        try:
            self._model = SentenceTransformer(self.model_id)
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise RuntimeError(
                f"Failed to load model '{self.model_id}'. "
                f"Check your internet connection and that the model exists on HuggingFace. "
                f"Error: {e}"
            ) from e

        # Detect and validate dimension
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self._dimension}")

        if config.expected_dimension and self._dimension != config.expected_dimension:
            raise ValueError(
                f"Dimension mismatch for {config.name}: "
                f"expected {config.expected_dimension}, got {self._dimension}"
            )

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def name(self) -> str:
        """Get model short name."""
        return self.config.name

    def encode(
        self,
        texts: List[str],
        show_progress: bool = True,
        batch_size: int | None = None
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings to encode
            show_progress: Show progress bar
            batch_size: Override default batch size

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self._dimension)

        bs = batch_size or self.batch_size

        embeddings = self._model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text (for testing/benchmarking).

        Args:
            text: Single text string

        Returns:
            1D numpy array of shape (dimension,)
        """
        embeddings = self.encode([text], show_progress=False)
        return embeddings[0]

    def __repr__(self) -> str:
        return f"EmbeddingModel(name={self.name}, dim={self._dimension})"


def load_model(model_name: str) -> EmbeddingModel:
    """
    Load an embedding model by name.

    Args:
        model_name: 'minilm' or 'mika'

    Returns:
        Loaded EmbeddingModel

    Raises:
        ValueError: If model name not recognized
        RuntimeError: If model fails to load
    """
    config = get_model_config(model_name)
    return EmbeddingModel(config)
