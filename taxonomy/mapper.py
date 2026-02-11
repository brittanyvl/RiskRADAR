"""
taxonomy/mapper.py
------------------
Map NTSB report chunks to CICTT occurrence categories using embeddings.

Uses pre-computed MIKA embeddings to find semantic similarity between
report content and CICTT category definitions.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from .cictt import CICTT_CATEGORIES, CICTTCategory, get_primary_categories
from .config import (
    CHUNKS_JSONL_PATH,
    EMBEDDINGS_DIR,
    TAXONOMY_DATA_DIR,
    TAXONOMY_CONFIG,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


def load_chunks_with_metadata() -> pd.DataFrame:
    """
    Load chunks with report metadata for context.

    Returns DataFrame with chunk data and report metadata joined.
    """
    logger.info(f"Loading chunks from {CHUNKS_JSONL_PATH}")

    chunks = []
    with open(CHUNKS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append({
                "chunk_id": chunk["chunk_id"],
                "report_id": chunk["report_id"],
                "chunk_text": chunk["chunk_text"],
                "section_name": chunk.get("section_name", ""),
                "token_count": chunk["token_count"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
            })

    chunks_df = pd.DataFrame(chunks)
    logger.info(f"Loaded {len(chunks_df):,} chunks")

    # Load report metadata from SQLite
    from riskradar.config import DB_PATH
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    reports_df = pd.read_sql_query("""
        SELECT filename as report_id, title, location, accident_date, report_date
        FROM reports
    """, conn)
    conn.close()

    # Merge
    merged = chunks_df.merge(reports_df, on="report_id", how="left")
    logger.info(f"Merged with report metadata: {len(merged):,} chunks")

    return merged


def filter_causal_sections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only causal-related sections, excluding appendices etc.
    """
    config = TAXONOMY_CONFIG

    # Build inclusion pattern
    include_pattern = "|".join(config.causal_sections)

    # Build exclusion pattern
    exclude_pattern = "|".join(config.excluded_sections)

    # Filter: must match causal AND not match excluded
    include_mask = df["section_name"].str.upper().str.contains(
        include_pattern, na=False, regex=True, case=False
    )
    exclude_mask = df["section_name"].str.upper().str.contains(
        exclude_pattern, na=False, regex=True, case=False
    )
    token_mask = df["token_count"] >= config.min_chunk_tokens

    filtered = df[include_mask & ~exclude_mask & token_mask].copy()

    logger.info(
        f"Filtered to causal sections: {len(df):,} -> {len(filtered):,} chunks "
        f"({len(filtered)/len(df)*100:.1f}%)"
    )

    return filtered


def load_mika_embeddings(chunk_ids: list[str]) -> pd.DataFrame:
    """Load pre-computed MIKA embeddings for specified chunks."""
    embeddings_path = EMBEDDINGS_DIR / "mika_embeddings.parquet"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"MIKA embeddings not found at {embeddings_path}. "
            "Run `python -m embeddings.cli embed mika` first."
        )

    logger.info(f"Loading MIKA embeddings from {embeddings_path}")

    table = pq.read_table(embeddings_path)
    df = table.to_pandas()

    chunk_id_set = set(chunk_ids)
    df_filtered = df[df["chunk_id"].isin(chunk_id_set)].copy()

    logger.info(f"Loaded {len(df_filtered):,} embeddings for {len(chunk_ids):,} chunks")

    return df_filtered[["chunk_id", "embedding"]]


def compute_cictt_embeddings(model: SentenceTransformer) -> dict[str, np.ndarray]:
    """
    Compute embeddings for CICTT category seed phrases.

    Returns dict mapping category code to average embedding of its seed phrases.
    """
    logger.info("Computing CICTT category embeddings...")

    category_embeddings = {}

    for cat in get_primary_categories():
        # Combine description and seed phrases for richer representation
        texts = [cat.description] + cat.seed_phrases
        embeddings = model.encode(texts, convert_to_numpy=True)

        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        category_embeddings[cat.code] = avg_embedding

    logger.info(f"Computed embeddings for {len(category_embeddings)} categories")
    return category_embeddings


def compute_similarities(
    chunk_embeddings: np.ndarray,
    category_embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute cosine similarity between chunks and CICTT categories.

    Returns DataFrame with columns: chunk_idx, category_code, similarity
    """
    # Stack category embeddings into matrix
    codes = list(category_embeddings.keys())
    cat_matrix = np.vstack([category_embeddings[code] for code in codes])

    # Normalize for cosine similarity
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    cat_norms = np.linalg.norm(cat_matrix, axis=1, keepdims=True)

    chunk_normalized = chunk_embeddings / (chunk_norms + 1e-8)
    cat_normalized = cat_matrix / (cat_norms + 1e-8)

    # Compute similarities: (n_chunks, n_categories)
    similarities = chunk_normalized @ cat_normalized.T

    logger.info(f"Computed similarity matrix: {similarities.shape}")

    return similarities, codes


def assign_categories(
    chunks_df: pd.DataFrame,
    similarities: np.ndarray,
    codes: list[str],
    threshold: float = None,
    top_k: int = None,
) -> pd.DataFrame:
    """
    Assign CICTT categories to chunks based on similarity scores.
    """
    if threshold is None:
        threshold = TAXONOMY_CONFIG.min_similarity_threshold
    if top_k is None:
        top_k = TAXONOMY_CONFIG.top_k_categories

    assignments = []

    for idx in range(len(chunks_df)):
        chunk_sims = similarities[idx]

        # Get top-k categories above threshold
        top_indices = np.argsort(chunk_sims)[::-1][:top_k]

        for rank, cat_idx in enumerate(top_indices):
            sim = chunk_sims[cat_idx]
            if sim >= threshold:
                assignments.append({
                    "chunk_id": chunks_df.iloc[idx]["chunk_id"],
                    "report_id": chunks_df.iloc[idx]["report_id"],
                    "category_code": codes[cat_idx],
                    "similarity": float(sim),
                    "rank": rank + 1,
                })

    assignments_df = pd.DataFrame(assignments)
    logger.info(f"Generated {len(assignments_df):,} category assignments")

    return assignments_df


def aggregate_report_categories(assignments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate chunk-level assignments to report-level categories.

    For each report, compute weighted category scores and assign top categories.
    """
    # Group by report and category, average similarity
    report_cats = assignments_df.groupby(["report_id", "category_code"]).agg({
        "similarity": ["mean", "max", "count"],
        "chunk_id": "count"
    }).reset_index()

    report_cats.columns = [
        "report_id", "category_code",
        "avg_similarity", "max_similarity", "n_assignments", "n_chunks"
    ]

    # Score = weighted combination
    report_cats["score"] = (
        0.4 * report_cats["avg_similarity"] +
        0.4 * report_cats["max_similarity"] +
        0.2 * np.minimum(report_cats["n_chunks"] / 5, 1.0)  # Bonus for multiple chunks
    )

    # Rank within each report
    report_cats["rank"] = report_cats.groupby("report_id")["score"].rank(
        ascending=False, method="first"
    ).astype(int)

    # Keep top categories per report
    max_cats = TAXONOMY_CONFIG.max_categories_per_report
    report_cats = report_cats[report_cats["rank"] <= max_cats].copy()

    # Normalize scores to percentages within each report
    report_totals = report_cats.groupby("report_id")["score"].sum()
    report_cats["pct_contribution"] = report_cats.apply(
        lambda row: 100 * row["score"] / report_totals[row["report_id"]], axis=1
    ).round(1)

    logger.info(f"Aggregated to {len(report_cats):,} report-category assignments")

    return report_cats


def map_reports_to_cictt(run_id: int = 1) -> dict:
    """
    Full pipeline: Map all reports to CICTT categories.

    Returns dict with mapping results and statistics.
    """
    logger.info("=" * 60)
    logger.info("Starting CICTT Taxonomy Mapping")
    logger.info("=" * 60)

    # Step 1: Load and filter chunks
    logger.info("Step 1: Loading chunks with metadata...")
    all_chunks = load_chunks_with_metadata()

    logger.info("Step 2: Filtering to causal sections...")
    causal_chunks = filter_causal_sections(all_chunks)

    # Step 3: Load embeddings
    logger.info("Step 3: Loading chunk embeddings...")
    chunk_ids = causal_chunks["chunk_id"].tolist()
    embeddings_df = load_mika_embeddings(chunk_ids)

    # Merge to ensure alignment
    merged = causal_chunks.merge(embeddings_df, on="chunk_id", how="inner")
    chunk_embeddings = np.vstack(merged["embedding"].values)

    logger.info(f"Processing {len(merged):,} chunks with embeddings")

    # Step 4: Compute CICTT category embeddings
    logger.info("Step 4: Computing CICTT category embeddings...")
    model = SentenceTransformer(TAXONOMY_CONFIG.embedding_model)
    category_embeddings = compute_cictt_embeddings(model)

    # Step 5: Compute similarities
    logger.info("Step 5: Computing chunk-category similarities...")
    similarities, codes = compute_similarities(chunk_embeddings, category_embeddings)

    # Step 6: Assign categories to chunks
    logger.info("Step 6: Assigning categories to chunks...")
    chunk_assignments = assign_categories(
        merged.drop(columns=["embedding"]),
        similarities,
        codes
    )

    # Step 7: Aggregate to report level
    logger.info("Step 7: Aggregating to report level...")
    report_categories = aggregate_report_categories(chunk_assignments)

    # Add category names
    from .cictt import CICTT_BY_CODE
    report_categories["category_name"] = report_categories["category_code"].map(
        lambda c: CICTT_BY_CODE[c].name if c in CICTT_BY_CODE else c
    )

    # Save results
    logger.info("Step 8: Saving results...")

    # Chunk-level assignments
    chunk_path = TAXONOMY_DATA_DIR / f"chunk_assignments_run{run_id}.parquet"
    chunk_assignments.to_parquet(chunk_path, index=False)

    # Report-level categories
    report_path = TAXONOMY_DATA_DIR / f"report_categories_run{run_id}.parquet"
    report_categories.to_parquet(report_path, index=False)

    # Summary statistics
    stats = {
        "run_id": run_id,
        "n_chunks_total": len(all_chunks),
        "n_chunks_causal": len(causal_chunks),
        "n_chunks_processed": len(merged),
        "n_reports": merged["report_id"].nunique(),
        "n_chunk_assignments": len(chunk_assignments),
        "n_report_assignments": len(report_categories),
        "categories_used": report_categories["category_code"].nunique(),
    }

    # Category distribution
    cat_dist = report_categories.groupby("category_code").agg({
        "report_id": "count"
    }).rename(columns={"report_id": "n_reports"}).sort_values("n_reports", ascending=False)

    stats_path = TAXONOMY_DATA_DIR / f"mapping_stats_run{run_id}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("CICTT Mapping Complete!")
    logger.info(f"  Chunks processed: {len(merged):,}")
    logger.info(f"  Reports mapped: {stats['n_reports']}")
    logger.info(f"  Categories used: {stats['categories_used']}")
    logger.info("=" * 60)

    return {
        "stats": stats,
        "chunk_assignments": chunk_assignments,
        "report_categories": report_categories,
        "category_distribution": cat_dist,
        "paths": {
            "chunk_assignments": chunk_path,
            "report_categories": report_path,
            "stats": stats_path,
        }
    }


def retry_unclassified(run_id: int = 2, config=None) -> dict:
    """
    Retry classification for accident reports that were not classified in the
    original run.

    Uses RetryConfig with relaxed thresholds (wider section matching,
    lower similarity threshold, smaller min tokens).

    Args:
        run_id: Run identifier for this retry pass (default 2)
        config: RetryConfig instance (uses default if None)

    Returns:
        Dict with retry results and statistics
    """
    from .config import RETRY_CONFIG
    from .cictt import CICTT_BY_CODE
    from riskradar.config import DB_PATH
    import sqlite3
    from datetime import datetime

    config = config or RETRY_CONFIG

    logger.info("=" * 60)
    logger.info("RETRY: Classifying unclassified accident reports")
    logger.info("=" * 60)

    # Find accident reports missing taxonomy
    conn = sqlite3.connect(DB_PATH)
    missing_df = pd.read_sql_query("""
        SELECT rt.report_id
        FROM report_types rt
        WHERE rt.report_type = 'accident'
          AND rt.report_id NOT IN (
              SELECT DISTINCT report_id FROM report_taxonomy WHERE level = 'L1'
          )
    """, conn)
    conn.close()

    missing_ids = set(missing_df["report_id"].tolist())
    logger.info(f"Found {len(missing_ids)} accident reports without taxonomy")

    if not missing_ids:
        logger.info("No unclassified accident reports - nothing to retry")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    # Load all chunks, then filter to missing reports
    all_chunks = load_chunks_with_metadata()
    target_chunks = all_chunks[all_chunks["report_id"].isin(missing_ids)].copy()
    logger.info(f"Found {len(target_chunks):,} chunks for {len(missing_ids)} missing reports")

    if target_chunks.empty:
        logger.info("No chunks found for missing reports")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    # Filter causal sections with relaxed config
    include_pattern = "|".join(config.causal_sections)
    exclude_pattern = "|".join(config.excluded_sections)

    include_mask = target_chunks["section_name"].str.upper().str.contains(
        include_pattern, na=False, regex=True, case=False
    )
    exclude_mask = target_chunks["section_name"].str.upper().str.contains(
        exclude_pattern, na=False, regex=True, case=False
    )
    token_mask = target_chunks["token_count"] >= config.min_chunk_tokens

    causal_chunks = target_chunks[include_mask & ~exclude_mask & token_mask].copy()
    logger.info(f"Filtered to {len(causal_chunks):,} causal chunks (relaxed config)")

    if causal_chunks.empty:
        logger.info("No causal chunks found with relaxed config")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    # Load embeddings
    chunk_ids = causal_chunks["chunk_id"].tolist()
    embeddings_df = load_mika_embeddings(chunk_ids)

    merged = causal_chunks.merge(embeddings_df, on="chunk_id", how="inner")
    if merged.empty:
        logger.info("No embeddings found for target chunks")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    chunk_embeddings = np.vstack(merged["embedding"].values)
    logger.info(f"Processing {len(merged):,} chunks with embeddings")

    # Compute CICTT embeddings
    model = SentenceTransformer(config.embedding_model)
    category_embeddings = compute_cictt_embeddings(model)

    # Compute similarities and assign
    similarities, codes = compute_similarities(chunk_embeddings, category_embeddings)
    chunk_assignments = assign_categories(
        merged.drop(columns=["embedding"]),
        similarities,
        codes,
        threshold=config.min_similarity_threshold,
        top_k=config.top_k_categories,
    )

    if chunk_assignments.empty:
        logger.info("No assignments above threshold")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    # Aggregate to report level
    report_categories = aggregate_report_categories(chunk_assignments)
    report_categories["category_name"] = report_categories["category_code"].map(
        lambda c: CICTT_BY_CODE[c].name if c in CICTT_BY_CODE else c
    )

    # Save to parquet
    chunk_path = TAXONOMY_DATA_DIR / f"chunk_assignments_run{run_id}.parquet"
    chunk_assignments.to_parquet(chunk_path, index=False)

    report_path = TAXONOMY_DATA_DIR / f"report_categories_run{run_id}.parquet"
    report_categories.to_parquet(report_path, index=False)

    # Save to SQLite (report_taxonomy table - same schema as original pipeline)
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)

    for _, row in report_categories.iterrows():
        conn.execute(
            """INSERT OR IGNORE INTO report_taxonomy
               (report_id, level, category_code, category_name, parent_code,
                confidence, rank, source_run_id)
               VALUES (?, 'L1', ?, ?, NULL, ?, ?, ?)""",
            (row["report_id"], row["category_code"], row["category_name"],
             float(row["score"]), int(row["rank"]), run_id),
        )

    conn.commit()
    conn.close()

    stats = {
        "run_id": run_id,
        "retry_count": len(missing_ids),
        "chunks_processed": len(merged),
        "chunk_assignments": len(chunk_assignments),
        "report_assignments": len(report_categories),
        "reports_classified": report_categories["report_id"].nunique(),
        "categories_used": report_categories["category_code"].nunique(),
    }

    # Save stats
    stats_path = TAXONOMY_DATA_DIR / f"mapping_stats_run{run_id}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("Retry Classification Complete!")
    logger.info(f"  Missing reports targeted: {len(missing_ids)}")
    logger.info(f"  Reports classified: {stats['reports_classified']}")
    logger.info(f"  Categories used: {stats['categories_used']}")
    logger.info("=" * 60)

    return {
        "stats": stats,
        "chunk_assignments": chunk_assignments,
        "report_categories": report_categories,
    }


def retry_final_pass(run_id: int = 3, config=None) -> dict:
    """
    Final retry pass using ALL sections except Tier 4 exclusions.

    This is the last-resort pass for edge-case reports (mostly 1970s-era)
    that have non-standard section names. Instead of include-listing
    causal sections, we include everything EXCEPT known low-signal sections
    (recommendations, numbered paragraphs, narrow factual).

    Args:
        run_id: Run identifier for this final pass (default 3)
        config: FinalRetryConfig instance (uses default if None)

    Returns:
        Dict with retry results and statistics
    """
    from .config import FINAL_RETRY_CONFIG
    from .cictt import CICTT_BY_CODE
    from riskradar.config import DB_PATH
    import sqlite3
    from datetime import datetime

    config = config or FINAL_RETRY_CONFIG

    logger.info("=" * 60)
    logger.info("FINAL PASS: Classifying remaining edge-case reports")
    logger.info("Using ALL sections except Tier 4 exclusions")
    logger.info("=" * 60)

    # Find accident reports still missing taxonomy
    conn = sqlite3.connect(DB_PATH)
    missing_df = pd.read_sql_query("""
        SELECT rt.report_id
        FROM report_types rt
        WHERE rt.report_type = 'accident'
          AND rt.report_id NOT IN (
              SELECT DISTINCT report_id FROM report_taxonomy WHERE level = 'L1'
          )
    """, conn)
    conn.close()

    missing_ids = set(missing_df["report_id"].tolist())
    logger.info(f"Found {len(missing_ids)} accident reports still without taxonomy")

    if not missing_ids:
        logger.info("No unclassified accident reports remain - nothing to do")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    # Load all chunks, then filter to missing reports
    all_chunks = load_chunks_with_metadata()
    target_chunks = all_chunks[all_chunks["report_id"].isin(missing_ids)].copy()
    logger.info(f"Found {len(target_chunks):,} chunks for {len(missing_ids)} missing reports")

    if target_chunks.empty:
        logger.info("No chunks found for missing reports")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    # Exclusion-based filtering: include everything EXCEPT Tier 4
    exclude_pattern = "|".join(config.excluded_sections)
    exclude_mask = target_chunks["section_name"].str.upper().str.contains(
        exclude_pattern, na=False, regex=True, case=False
    )
    token_mask = target_chunks["token_count"] >= config.min_chunk_tokens

    eligible_chunks = target_chunks[~exclude_mask & token_mask].copy()
    logger.info(f"After excluding Tier 4 sections: {len(eligible_chunks):,} chunks "
                f"(from {len(target_chunks):,})")

    # Log section distribution for transparency
    section_dist = eligible_chunks["section_name"].value_counts()
    logger.info("Section distribution of eligible chunks:")
    for sec, cnt in section_dist.head(15).items():
        logger.info(f"  {sec}: {cnt}")

    if eligible_chunks.empty:
        logger.info("No eligible chunks after filtering")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    # Load embeddings
    chunk_ids = eligible_chunks["chunk_id"].tolist()
    embeddings_df = load_mika_embeddings(chunk_ids)

    merged = eligible_chunks.merge(embeddings_df, on="chunk_id", how="inner")
    if merged.empty:
        logger.info("No embeddings found for target chunks")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    chunk_embeddings = np.vstack(merged["embedding"].values)
    logger.info(f"Processing {len(merged):,} chunks with embeddings")

    # Compute CICTT embeddings
    model = SentenceTransformer(config.embedding_model)
    category_embeddings = compute_cictt_embeddings(model)

    # Compute similarities and assign
    similarities, codes = compute_similarities(chunk_embeddings, category_embeddings)
    chunk_assignments = assign_categories(
        merged.drop(columns=["embedding"]),
        similarities,
        codes,
        threshold=config.min_similarity_threshold,
        top_k=config.top_k_categories,
    )

    if chunk_assignments.empty:
        logger.info("No assignments above threshold")
        return {"stats": {"retry_count": 0, "run_id": run_id}}

    # Aggregate to report level
    report_categories = aggregate_report_categories(chunk_assignments)
    report_categories["category_name"] = report_categories["category_code"].map(
        lambda c: CICTT_BY_CODE[c].name if c in CICTT_BY_CODE else c
    )

    # Save to parquet
    chunk_path = TAXONOMY_DATA_DIR / f"chunk_assignments_run{run_id}.parquet"
    chunk_assignments.to_parquet(chunk_path, index=False)

    report_path = TAXONOMY_DATA_DIR / f"report_categories_run{run_id}.parquet"
    report_categories.to_parquet(report_path, index=False)

    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)

    for _, row in report_categories.iterrows():
        conn.execute(
            """INSERT OR IGNORE INTO report_taxonomy
               (report_id, level, category_code, category_name, parent_code,
                confidence, rank, source_run_id)
               VALUES (?, 'L1', ?, ?, NULL, ?, ?, ?)""",
            (row["report_id"], row["category_code"], row["category_name"],
             float(row["score"]), int(row["rank"]), run_id),
        )

    conn.commit()
    conn.close()

    stats = {
        "run_id": run_id,
        "retry_count": len(missing_ids),
        "chunks_processed": len(merged),
        "chunk_assignments": len(chunk_assignments),
        "report_assignments": len(report_categories),
        "reports_classified": report_categories["report_id"].nunique(),
        "categories_used": report_categories["category_code"].nunique(),
        "sections_used": eligible_chunks["section_name"].nunique(),
    }

    # Save stats
    stats_path = TAXONOMY_DATA_DIR / f"mapping_stats_run{run_id}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("Final Pass Classification Complete!")
    logger.info(f"  Missing reports targeted: {len(missing_ids)}")
    logger.info(f"  Reports classified: {stats['reports_classified']}")
    logger.info(f"  Categories used: {stats['categories_used']}")
    logger.info(f"  Sections used: {stats['sections_used']}")
    logger.info("=" * 60)

    return {
        "stats": stats,
        "chunk_assignments": chunk_assignments,
        "report_categories": report_categories,
    }


def load_mapping_results(run_id: int = 1) -> dict:
    """Load previous mapping results."""
    chunk_path = TAXONOMY_DATA_DIR / f"chunk_assignments_run{run_id}.parquet"
    report_path = TAXONOMY_DATA_DIR / f"report_categories_run{run_id}.parquet"
    stats_path = TAXONOMY_DATA_DIR / f"mapping_stats_run{run_id}.json"

    results = {}

    if chunk_path.exists():
        results["chunk_assignments"] = pd.read_parquet(chunk_path)

    if report_path.exists():
        results["report_categories"] = pd.read_parquet(report_path)

    if stats_path.exists():
        with open(stats_path) as f:
            results["stats"] = json.load(f)

    return results
