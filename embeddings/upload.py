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


def enrich_payloads_with_taxonomy(
    model_name: str,
    l1_run_id: int = 1,
    l2_run_id: int = 1,
    conn=None
) -> Dict:
    """
    Enrich existing Qdrant payloads with taxonomy data and PDF URLs.

    Updates existing vectors in-place without re-uploading embeddings.
    Adds:
      - l1_categories: List of L1 category codes for the report
      - l2_subcategories: List of L2 subcategory codes for the report
      - pdf_url: Direct link to NTSB original PDF

    Args:
        model_name: 'minilm' or 'mika'
        l1_run_id: L1 classification run ID
        l2_run_id: L2 classification run ID
        conn: Optional SQLite connection

    Returns:
        Dict with enrichment statistics
    """
    import pandas as pd
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SetPayloadOperation

    from taxonomy.config import TAXONOMY_DATA_DIR

    close_conn = False
    if conn is None:
        conn = init_db(DB_PATH)
        close_conn = True

    config = get_model_config(model_name)

    stats = {
        "model_name": model_name,
        "collection_name": config.collection_name,
        "l1_run_id": l1_run_id,
        "l2_run_id": l2_run_id,
        "started_at": time.time(),
        "reports_with_taxonomy": 0,
        "reports_without_taxonomy": 0,
        "points_updated": 0,
        "batches_processed": 0,
    }

    try:
        # Load L1 report-level taxonomy
        l1_path = TAXONOMY_DATA_DIR / f"report_categories_run{l1_run_id}.parquet"
        if not l1_path.exists():
            raise FileNotFoundError(f"L1 results not found: {l1_path}")

        l1_df = pd.read_parquet(l1_path)
        logger.info(f"Loaded L1 results: {len(l1_df)} report-category assignments")

        # Build report_id -> [l1_categories] mapping
        l1_by_report = l1_df.groupby("report_id")["category_code"].apply(list).to_dict()

        # Load L2 report-level taxonomy
        l2_path = TAXONOMY_DATA_DIR / f"report_l2_run{l2_run_id}.parquet"
        if l2_path.exists():
            l2_df = pd.read_parquet(l2_path)
            logger.info(f"Loaded L2 results: {len(l2_df)} report-subcategory assignments")
            l2_by_report = l2_df.groupby("report_id")["subcategory_code"].apply(list).to_dict()
        else:
            logger.warning(f"L2 results not found: {l2_path}")
            l2_by_report = {}

        # Load report metadata for PDF URLs
        cursor = conn.execute("SELECT filename, pdf_url FROM reports")
        pdf_urls = {row["filename"]: row["pdf_url"] for row in cursor.fetchall()}
        logger.info(f"Loaded {len(pdf_urls)} PDF URLs")

        # Build combined enrichment data per report
        all_reports = set(l1_by_report.keys()) | set(l2_by_report.keys()) | set(pdf_urls.keys())

        enrichment_data = {}
        for report_id in all_reports:
            enrichment_data[report_id] = {
                "l1_categories": l1_by_report.get(report_id, []),
                "l2_subcategories": l2_by_report.get(report_id, []),
                "pdf_url": pdf_urls.get(report_id, ""),
            }
            if l1_by_report.get(report_id):
                stats["reports_with_taxonomy"] += 1
            else:
                stats["reports_without_taxonomy"] += 1

        logger.info(f"Reports with taxonomy: {stats['reports_with_taxonomy']}")
        logger.info(f"Reports without taxonomy: {stats['reports_without_taxonomy']}")

        # Connect to Qdrant
        client = get_qdrant_client()

        # Verify collection exists
        collection_info = client.get_collection(config.collection_name)
        total_points = collection_info.points_count
        logger.info(f"Collection {config.collection_name}: {total_points} points")

        # Update payloads in batches by scrolling through collection
        # Group by report_id for efficient batch updates
        batch_size = 100
        offset = None

        while True:
            # Scroll through points
            scroll_result = client.scroll(
                collection_name=config.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points, next_offset = scroll_result

            if not points:
                break

            # Group points by report_id for batch updates
            points_by_report = {}
            for point in points:
                report_id = point.payload.get("report_id", "")
                if report_id in enrichment_data:
                    if report_id not in points_by_report:
                        points_by_report[report_id] = []
                    points_by_report[report_id].append(point.id)

            # Batch update per report
            for report_id, point_ids in points_by_report.items():
                new_payload = enrichment_data[report_id]
                client.set_payload(
                    collection_name=config.collection_name,
                    payload=new_payload,
                    points=point_ids,
                )
                stats["points_updated"] += len(point_ids)

            stats["batches_processed"] += 1

            if stats["batches_processed"] % 50 == 0:
                logger.info(f"Updated {stats['points_updated']} points...")

            offset = next_offset
            if offset is None:
                break

        stats["total_time_sec"] = time.time() - stats["started_at"]
        stats["status"] = "completed"

        logger.info(f"\n{'='*60}")
        logger.info("Payload Enrichment Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Collection: {config.collection_name}")
        logger.info(f"Points updated: {stats['points_updated']}")
        logger.info(f"Reports with taxonomy: {stats['reports_with_taxonomy']}")
        logger.info(f"Reports without taxonomy: {stats['reports_without_taxonomy']}")
        logger.info(f"Time: {stats['total_time_sec']:.1f}s")

    except Exception as e:
        stats["status"] = "failed"
        stats["error"] = str(e)
        stats["total_time_sec"] = time.time() - stats["started_at"]
        logger.error(f"Enrichment failed: {e}")
        raise

    finally:
        if close_conn:
            conn.close()

    return stats


def enrich_all_models(
    l1_run_id: int = 1,
    l2_run_id: int = 1,
    conn=None
) -> Dict:
    """
    Enrich payloads for all models with taxonomy data.

    Args:
        l1_run_id: L1 classification run ID
        l2_run_id: L2 classification run ID
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
            logger.info(f"Enriching model: {model_name}")
            logger.info(f"{'='*60}")

            try:
                stats = enrich_payloads_with_taxonomy(
                    model_name=model_name,
                    l1_run_id=l1_run_id,
                    l2_run_id=l2_run_id,
                    conn=conn,
                )
                all_stats[model_name] = stats
            except Exception as e:
                logger.error(f"Failed to enrich {model_name}: {e}")
                all_stats[model_name] = {"status": "failed", "error": str(e)}

    finally:
        if close_conn:
            conn.close()

    return all_stats
