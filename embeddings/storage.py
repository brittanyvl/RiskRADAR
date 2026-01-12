"""
embeddings/storage.py
---------------------
Parquet-based storage for embeddings.

Handles reading chunks from JSONL and writing/reading embeddings to/from Parquet.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import CHUNKS_JSONL_PATH, EMBEDDINGS_DATA_DIR, get_parquet_path

logger = logging.getLogger(__name__)


def load_chunks(
    chunks_path: Path | None = None,
    limit: int | None = None
) -> List[Dict]:
    """
    Load chunks from chunks.jsonl.

    Args:
        chunks_path: Path to chunks.jsonl (default: config path)
        limit: Optional limit on number of chunks

    Returns:
        List of chunk dicts with chunk_id, chunk_text, and metadata
    """
    path = chunks_path or CHUNKS_JSONL_PATH

    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                chunk = json.loads(line)
                chunks.append(chunk)

    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def save_embeddings(
    model_name: str,
    chunk_ids: List[str],
    embeddings: np.ndarray,
    chunks: List[Dict],
    output_path: Path | None = None
) -> Dict:
    """
    Save embeddings to Parquet file.

    Schema includes chunk metadata for Qdrant payload construction.

    Args:
        model_name: Model name ('minilm' or 'mika')
        chunk_ids: List of chunk IDs
        embeddings: numpy array (N, D)
        chunks: List of chunk dicts with metadata
        output_path: Optional output path (default: config path)

    Returns:
        Dict with file stats (path, size_mb, count)
    """
    path = output_path or get_parquet_path(model_name)

    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build lookup for chunk metadata
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    # Prepare data for Parquet
    records = []
    for i, chunk_id in enumerate(chunk_ids):
        chunk = chunk_lookup.get(chunk_id, {})
        records.append({
            "chunk_id": chunk_id,
            "embedding": embeddings[i].tolist(),
            "report_id": chunk.get("report_id", ""),
            "chunk_sequence": chunk.get("chunk_sequence", 0),
            "page_start": chunk.get("page_start", 0),
            "page_end": chunk.get("page_end", 0),
            "section_name": chunk.get("section_name", ""),
            "token_count": chunk.get("token_count", 0),
            "text_source": chunk.get("text_source", ""),
        })

    # Create PyArrow table
    table = pa.Table.from_pylist(records)

    # Add metadata
    metadata = {
        b"model_name": model_name.encode(),
        b"embedding_dimension": str(embeddings.shape[1]).encode(),
        b"count": str(len(chunk_ids)).encode(),
        b"created_at": datetime.now(timezone.utc).isoformat().encode(),
    }
    table = table.replace_schema_metadata(metadata)

    # Write to Parquet
    pq.write_table(table, path, compression="snappy")

    # Get file size
    size_mb = path.stat().st_size / (1024 * 1024)

    logger.info(f"Saved {len(chunk_ids)} embeddings to {path} ({size_mb:.1f} MB)")

    return {
        "path": str(path),
        "size_mb": size_mb,
        "count": len(chunk_ids),
        "dimension": embeddings.shape[1],
    }


def load_embeddings(
    model_name: str,
    input_path: Path | None = None
) -> tuple[List[str], np.ndarray, List[Dict], Dict]:
    """
    Load embeddings from Parquet file.

    Args:
        model_name: Model name ('minilm' or 'mika')
        input_path: Optional input path (default: config path)

    Returns:
        Tuple of (chunk_ids, embeddings, metadata_list, file_metadata)
    """
    path = input_path or get_parquet_path(model_name)

    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    # Read Parquet
    table = pq.read_table(path)

    # Extract file metadata
    file_metadata = {}
    if table.schema.metadata:
        for key, value in table.schema.metadata.items():
            file_metadata[key.decode()] = value.decode()

    # Convert to records
    df = table.to_pandas()

    chunk_ids = df["chunk_id"].tolist()
    embeddings = np.array(df["embedding"].tolist())

    # Build metadata list
    metadata_list = []
    for _, row in df.iterrows():
        metadata_list.append({
            "chunk_id": row["chunk_id"],
            "report_id": row["report_id"],
            "chunk_sequence": row["chunk_sequence"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "section_name": row["section_name"],
            "token_count": row["token_count"],
            "text_source": row["text_source"],
        })

    logger.info(f"Loaded {len(chunk_ids)} embeddings from {path}")

    return chunk_ids, embeddings, metadata_list, file_metadata


def get_parquet_stats(model_name: str) -> Dict:
    """
    Get statistics for a Parquet embeddings file.

    Args:
        model_name: Model name ('minilm' or 'mika')

    Returns:
        Dict with file statistics or {'exists': False}
    """
    path = get_parquet_path(model_name)

    if not path.exists():
        return {"exists": False, "model_name": model_name}

    # Read just metadata
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.schema_arrow.metadata or {}

    stats = {
        "exists": True,
        "model_name": model_name,
        "path": str(path),
        "size_mb": path.stat().st_size / (1024 * 1024),
        "num_rows": parquet_file.metadata.num_rows,
    }

    # Add embedded metadata
    for key, value in metadata.items():
        stats[key.decode()] = value.decode()

    return stats


def iter_embeddings_batched(
    model_name: str,
    batch_size: int = 100,
    input_path: Path | None = None
) -> Iterator[tuple[List[str], np.ndarray, List[Dict]]]:
    """
    Iterate over embeddings in batches (memory efficient for upload).

    Args:
        model_name: Model name
        batch_size: Number of embeddings per batch
        input_path: Optional input path

    Yields:
        Tuples of (chunk_ids, embeddings, metadata_list) for each batch
    """
    path = input_path or get_parquet_path(model_name)

    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    # Read in batches using PyArrow
    parquet_file = pq.ParquetFile(path)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()

        chunk_ids = df["chunk_id"].tolist()
        embeddings = np.array(df["embedding"].tolist())

        metadata_list = []
        for _, row in df.iterrows():
            metadata_list.append({
                "chunk_id": row["chunk_id"],
                "report_id": row["report_id"],
                "chunk_sequence": row["chunk_sequence"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "section_name": row["section_name"],
                "token_count": row["token_count"],
                "text_source": row["text_source"],
            })

        yield chunk_ids, embeddings, metadata_list
