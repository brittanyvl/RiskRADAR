"""
embeddings/upload.py
--------------------
Upload embeddings from Parquet to Qdrant Cloud.
"""

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from riskradar.config import DB_PATH, get_qdrant_config
from sqlite.connection import init_db
from sqlite import queries

from .config import PIPELINE_CONFIG, get_model_config, get_parquet_path
from .storage import get_parquet_stats, iter_embeddings_batched

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """
    Create Qdrant client from configuration.

    Returns:
        Configured QdrantClient

    Raises:
        ValueError: If Qdrant config not found
    """
    config = get_qdrant_config()
    return QdrantClient(
        url=config["url"],
        api_key=config["api_key"],
    )


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_dim: int,
    recreate: bool = False
) -> bool:
    """
    Create Qdrant collection if it doesn't exist.

    Args:
        client: QdrantClient instance
        collection_name: Name for collection
        vector_dim: Embedding dimension
        recreate: If True, delete and recreate existing collection

    Returns:
        True if collection was created, False if already existed
    """
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if exists and recreate:
        logger.warning(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
        exists = False

    if exists:
        logger.info(f"Collection already exists: {collection_name}")
        return False

    # Create collection
    logger.info(f"Creating collection: {collection_name} (dim={vector_dim})")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
        ),
    )
    return True


def build_payload(metadata: Dict, report_metadata: Dict | None = None) -> Dict:
    """
    Build Qdrant payload from chunk metadata.

    Args:
        metadata: Chunk metadata from Parquet
        report_metadata: Optional report metadata from SQLite

    Returns:
        Payload dict for Qdrant
    """
    payload = {
        "chunk_id": metadata.get("chunk_id", ""),
        "report_id": metadata.get("report_id", ""),
        "chunk_sequence": metadata.get("chunk_sequence", 0),
        "page_start": metadata.get("page_start", 0),
        "page_end": metadata.get("page_end", 0),
        "section_name": metadata.get("section_name", ""),
        "token_count": metadata.get("token_count", 0),
        "text_source": metadata.get("text_source", ""),
    }

    # Add report metadata if available
    if report_metadata:
        payload["accident_date"] = report_metadata.get("accident_date", "")
        payload["report_date"] = report_metadata.get("report_date", "")
        payload["location"] = report_metadata.get("location", "")
        payload["title"] = report_metadata.get("title", "")

    return payload


def upload_embeddings(
    model_name: str,
    input_path: Path | None = None,
    recreate_collection: bool = False,
    conn=None
) -> Dict:
    """
    Upload embeddings from Parquet to Qdrant Cloud.

    Args:
        model_name: 'minilm' or 'mika'
        input_path: Path to Parquet file (default: config path)
        recreate_collection: If True, delete and recreate collection
        conn: Optional SQLite connection

    Returns:
        Dict with upload statistics
    """
    close_conn = False
    if conn is None:
        conn = init_db(DB_PATH)
        close_conn = True

    # Get model config
    config = get_model_config(model_name)
    parquet_path = input_path or get_parquet_path(model_name)
    batch_size = PIPELINE_CONFIG["upload_batch_size"]

    # Get Qdrant config
    qdrant_config = get_qdrant_config()

    # Get Parquet stats
    parquet_stats = get_parquet_stats(model_name)
    if not parquet_stats.get("exists"):
        raise FileNotFoundError(
            f"Embeddings file not found for {model_name}. "
            f"Run 'embed {model_name}' first."
        )

    total_vectors = parquet_stats.get("num_rows", 0)
    dimension = int(parquet_stats.get("embedding_dimension", config.expected_dimension))

    # Create run record
    run_id = None
    try:
        run_id = queries.create_qdrant_upload_run(
            conn,
            model_name=model_name,
            collection_name=config.collection_name,
            qdrant_url=qdrant_config["url"],
            config_json=json.dumps({
                "pipeline_version": PIPELINE_CONFIG["version"],
                "batch_size": batch_size,
                "recreate_collection": recreate_collection,
            })
        )
        logger.info(f"Created upload run {run_id}")
    except Exception as e:
        logger.warning(f"Failed to create run record: {e}")

    stats = {
        "model_name": model_name,
        "collection_name": config.collection_name,
        "total_vectors": total_vectors,
        "dimension": dimension,
        "started_at": time.time(),
        "uploaded_vectors": 0,
        "failed_vectors": 0,
        "batches_uploaded": 0,
    }

    try:
        # Load report metadata for payloads
        logger.info("Loading report metadata from SQLite")
        cursor = conn.execute("SELECT * FROM reports")
        reports = {row["filename"]: dict(row) for row in cursor.fetchall()}
        logger.info(f"Loaded {len(reports)} reports")

        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant: {qdrant_config['url']}")
        client = get_qdrant_client()

        # Create collection
        create_collection(
            client,
            config.collection_name,
            dimension,
            recreate=recreate_collection,
        )

        if run_id:
            queries.update_qdrant_upload_run(
                conn, run_id, total_vectors=total_vectors
            )

        # Upload in batches
        logger.info(f"Uploading {total_vectors} vectors in batches of {batch_size}")
        upload_start = time.time()

        for chunk_ids, embeddings, metadata_list in iter_embeddings_batched(
            model_name, batch_size=batch_size, input_path=parquet_path
        ):
            # Build points
            points = []
            for i, chunk_id in enumerate(chunk_ids):
                meta = metadata_list[i]
                report_meta = reports.get(meta.get("report_id", ""))
                payload = build_payload(meta, report_meta)

                points.append(PointStruct(
                    id=hash(chunk_id) & 0x7FFFFFFFFFFFFFFF,  # Positive int64
                    vector=embeddings[i].tolist(),
                    payload=payload,
                ))

            # Upload batch with retry
            success = False
            for attempt in range(PIPELINE_CONFIG["retry_attempts"]):
                try:
                    client.upsert(
                        collection_name=config.collection_name,
                        points=points,
                    )
                    success = True
                    break
                except Exception as e:
                    if attempt < PIPELINE_CONFIG["retry_attempts"] - 1:
                        delay = PIPELINE_CONFIG["retry_delay_sec"] * (2 ** attempt)
                        logger.warning(f"Batch upload failed, retrying in {delay}s: {e}")
                        time.sleep(delay)
                    else:
                        raise

            if success:
                stats["uploaded_vectors"] += len(points)
                stats["batches_uploaded"] += 1
            else:
                stats["failed_vectors"] += len(points)

            # Log progress
            if stats["batches_uploaded"] % 10 == 0:
                logger.info(
                    f"Uploaded {stats['uploaded_vectors']}/{total_vectors} vectors"
                )

        # Calculate final stats
        upload_time = time.time() - upload_start
        stats["upload_time_sec"] = upload_time
        stats["vectors_per_sec"] = stats["uploaded_vectors"] / upload_time if upload_time > 0 else 0
        stats["total_time_sec"] = time.time() - stats["started_at"]

        # Verify upload
        collection_info = client.get_collection(config.collection_name)
        stats["collection_count"] = collection_info.points_count

        # Update run record
        if run_id:
            queries.update_qdrant_upload_run(
                conn,
                run_id,
                status="completed",
                uploaded_vectors=stats["uploaded_vectors"],
                failed_vectors=stats["failed_vectors"],
                batches_uploaded=stats["batches_uploaded"],
                total_time_sec=stats["total_time_sec"],
                vectors_per_sec=stats["vectors_per_sec"],
            )

        logger.info(f"Upload complete for {model_name}")
        logger.info(f"  Collection: {config.collection_name}")
        logger.info(f"  Uploaded: {stats['uploaded_vectors']}")
        logger.info(f"  Failed: {stats['failed_vectors']}")
        logger.info(f"  Time: {stats['total_time_sec']:.1f}s")
        logger.info(f"  Speed: {stats['vectors_per_sec']:.1f} vec/sec")
        logger.info(f"  Collection count: {stats['collection_count']}")

        stats["status"] = "completed"

    except Exception as e:
        stats["status"] = "failed"
        stats["error"] = str(e)
        stats["total_time_sec"] = time.time() - stats["started_at"]

        logger.error(f"Upload failed: {e}")

        if run_id:
            queries.update_qdrant_upload_run(conn, run_id, status="failed", error_count=1)
            queries.log_embedding_error(
                conn,
                run_id=run_id,
                run_type="upload",
                error_type="upload_failed",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
            )

        raise

    finally:
        if close_conn:
            conn.close()

    return stats


def verify_collection(model_name: str) -> Dict:
    """
    Verify Qdrant collection exists and has expected count.

    Args:
        model_name: 'minilm' or 'mika'

    Returns:
        Dict with verification results
    """
    config = get_model_config(model_name)
    parquet_stats = get_parquet_stats(model_name)

    expected_count = parquet_stats.get("num_rows", 0) if parquet_stats.get("exists") else 0

    try:
        client = get_qdrant_client()
        collection_info = client.get_collection(config.collection_name)

        actual_count = collection_info.points_count
        match = actual_count == expected_count

        return {
            "model_name": model_name,
            "collection_name": config.collection_name,
            "exists": True,
            "expected_count": expected_count,
            "actual_count": actual_count,
            "match": match,
            "status": "ok" if match else "mismatch",
        }

    except Exception as e:
        return {
            "model_name": model_name,
            "collection_name": config.collection_name,
            "exists": False,
            "error": str(e),
            "status": "error",
        }


def upload_all_models(
    recreate_collections: bool = False,
    conn=None
) -> Dict:
    """
    Upload embeddings for all models to Qdrant.

    Args:
        recreate_collections: If True, delete and recreate collections
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
            logger.info(f"Uploading model: {model_name}")
            logger.info(f"{'='*60}")

            try:
                stats = upload_embeddings(
                    model_name=model_name,
                    recreate_collection=recreate_collections,
                    conn=conn,
                )
                all_stats[model_name] = stats
            except Exception as e:
                logger.error(f"Failed to upload {model_name}: {e}")
                all_stats[model_name] = {"status": "failed", "error": str(e)}

    finally:
        if close_conn:
            conn.close()

    return all_stats
