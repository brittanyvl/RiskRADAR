"""
embeddings/embed.py
-------------------
Embedding generation pipeline.

Loads chunks, generates embeddings, and saves to Parquet.
"""

import gc
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict

from riskradar.config import DB_PATH
from sqlite.connection import init_db
from sqlite import queries

from .config import (
    CHUNKS_JSONL_PATH,
    PIPELINE_CONFIG,
    get_model_config,
    get_parquet_path,
)
from .models import load_model
from .storage import load_chunks, save_embeddings

logger = logging.getLogger(__name__)


def embed_chunks(
    model_name: str,
    chunks_path: Path | None = None,
    output_path: Path | None = None,
    limit: int | None = None,
    conn=None
) -> Dict:
    """
    Generate embeddings for all chunks using specified model.

    Pipeline:
    1. Load chunks from chunks.jsonl
    2. Load embedding model
    3. Generate embeddings (with progress bar)
    4. Save to Parquet
    5. Update SQLite run tracking

    Args:
        model_name: 'minilm' or 'mika'
        chunks_path: Path to chunks.jsonl (default: config path)
        output_path: Path for Parquet output (default: config path)
        limit: Optional limit on chunks (for testing)
        conn: Optional SQLite connection for run tracking

    Returns:
        Dict with statistics
    """
    close_conn = False
    if conn is None:
        conn = init_db(DB_PATH)
        close_conn = True

    # Get model config
    config = get_model_config(model_name)
    chunks_file = chunks_path or CHUNKS_JSONL_PATH
    output_file = output_path or get_parquet_path(model_name)

    # Create run record
    run_id = None
    try:
        run_id = queries.create_embedding_run(
            conn,
            model_name=model_name,
            model_id=config.model_id,
            run_type="full",
            config_json=json.dumps({
                "pipeline_version": PIPELINE_CONFIG["version"],
                "batch_size": config.batch_size,
                "expected_dimension": config.expected_dimension,
                "limit": limit,
            })
        )
        logger.info(f"Created embedding run {run_id}")
    except Exception as e:
        logger.warning(f"Failed to create run record: {e}")

    stats = {
        "model_name": model_name,
        "model_id": config.model_id,
        "started_at": time.time(),
    }

    try:
        # Step 1: Load chunks
        logger.info(f"Loading chunks from {chunks_file}")
        chunks = load_chunks(chunks_file, limit=limit)
        stats["total_chunks"] = len(chunks)

        if run_id:
            queries.update_embedding_run(conn, run_id, total_chunks=len(chunks))

        # Step 2: Load model
        logger.info(f"Loading model: {config.model_id}")
        model = load_model(model_name)
        stats["embedding_dimension"] = model.dimension

        if run_id:
            queries.update_embedding_run(conn, run_id, embedding_dimension=model.dimension)

        # Step 3: Extract texts
        logger.info("Extracting texts from chunks")
        chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [c["chunk_text"] for c in chunks]

        # Step 4: Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embed_start = time.time()
        embeddings = model.encode(texts, show_progress=True)
        embed_time = time.time() - embed_start

        stats["embeddings_generated"] = len(embeddings)
        stats["embed_time_sec"] = embed_time
        stats["embeddings_per_sec"] = len(embeddings) / embed_time if embed_time > 0 else 0

        logger.info(
            f"Generated {len(embeddings)} embeddings in {embed_time:.1f}s "
            f"({stats['embeddings_per_sec']:.1f} emb/sec)"
        )

        # Step 5: Save to Parquet
        logger.info(f"Saving embeddings to {output_file}")
        save_stats = save_embeddings(
            model_name=model_name,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            chunks=chunks,
            output_path=output_file,
        )
        stats["parquet_path"] = save_stats["path"]
        stats["parquet_size_mb"] = save_stats["size_mb"]

        # Calculate total time
        stats["total_time_sec"] = time.time() - stats["started_at"]

        # Update run record
        if run_id:
            queries.update_embedding_run(
                conn,
                run_id,
                status="completed",
                embeddings_generated=stats["embeddings_generated"],
                embedding_dimension=stats["embedding_dimension"],
                total_time_sec=stats["total_time_sec"],
                embeddings_per_sec=stats["embeddings_per_sec"],
                parquet_path=stats["parquet_path"],
                parquet_size_mb=stats["parquet_size_mb"],
            )

        logger.info(f"Embedding complete for {model_name}")
        logger.info(f"  Chunks: {stats['total_chunks']}")
        logger.info(f"  Dimension: {stats['embedding_dimension']}")
        logger.info(f"  Time: {stats['total_time_sec']:.1f}s")
        logger.info(f"  Speed: {stats['embeddings_per_sec']:.1f} emb/sec")
        logger.info(f"  Output: {stats['parquet_path']} ({stats['parquet_size_mb']:.1f} MB)")

        # Clean up model to free memory
        del model
        del embeddings
        gc.collect()

        stats["status"] = "completed"

    except Exception as e:
        stats["status"] = "failed"
        stats["error"] = str(e)
        stats["total_time_sec"] = time.time() - stats["started_at"]

        logger.error(f"Embedding failed: {e}")

        if run_id:
            queries.update_embedding_run(conn, run_id, status="failed", error_count=1)
            queries.log_embedding_error(
                conn,
                run_id=run_id,
                run_type="embed",
                error_type="embedding_failed",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
            )

        raise

    finally:
        if close_conn:
            conn.close()

    return stats


def embed_all_models(
    chunks_path: Path | None = None,
    limit: int | None = None,
    conn=None
) -> Dict:
    """
    Generate embeddings for all configured models.

    Processes models sequentially to manage memory.

    Args:
        chunks_path: Path to chunks.jsonl
        limit: Optional limit on chunks
        conn: Optional SQLite connection

    Returns:
        Dict with per-model statistics
    """
    from .config import MODELS

    close_conn = False
    if conn is None:
        conn = init_db(DB_PATH)
        close_conn = True

    all_stats = {}

    try:
        for model_name in MODELS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing model: {model_name}")
            logger.info(f"{'='*60}")

            try:
                stats = embed_chunks(
                    model_name=model_name,
                    chunks_path=chunks_path,
                    limit=limit,
                    conn=conn,
                )
                all_stats[model_name] = stats
            except Exception as e:
                logger.error(f"Failed to embed with {model_name}: {e}")
                all_stats[model_name] = {"status": "failed", "error": str(e)}

            # Force garbage collection between models
            gc.collect()

    finally:
        if close_conn:
            conn.close()

    return all_stats
