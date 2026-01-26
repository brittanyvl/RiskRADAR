"""
taxonomy/pipeline.py
--------------------
Pipeline for L2 subcategory classification, building on existing L1 results.

This pipeline does NOT re-run L1 classification. It:
1. Loads existing L1 results from mapper.py run
2. Identifies chunks with L2-enabled L1 categories
3. Loads MIKA embeddings for those chunks
4. Classifies to L2 subcategories
5. Aggregates to report level
6. Saves combined L1+L2 results

Usage:
    python -m taxonomy.cli classify-l2              # Add L2 to existing L1 run
    python -m taxonomy.cli classify-l2 --l1-run 1   # Specify L1 run to extend
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    CHUNKS_JSONL_PATH,
    TAXONOMY_DATA_DIR,
    TAXONOMY_CONFIG,
)
from .hierarchical_mapper import HierarchicalMapper, load_hierarchical_results
from .mapper import load_chunks_with_metadata, filter_causal_sections

logger = logging.getLogger(__name__)


def run_l2_classification(
    l1_run_id: int = 1,
    l2_run_id: int = 1,
    config=None,
) -> dict:
    """
    Run L2 subcategory classification on existing L1 results.

    This does NOT re-run L1 - it builds on your existing L1 classification.

    Args:
        l1_run_id: The run ID of existing L1 classification to extend
        l2_run_id: Unique identifier for this L2 run
        config: TaxonomyConfig instance (uses default if None)

    Returns:
        Dict with L2 classification results and statistics
    """
    config = config or TAXONOMY_CONFIG
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Starting L2 Subcategory Classification")
    logger.info("=" * 60)
    logger.info(f"Building on L1 run: {l1_run_id}")
    logger.info(f"L2 run ID: {l2_run_id}")
    logger.info(f"Config version: {config.version}")

    # Initialize mapper
    mapper = HierarchicalMapper(config)

    # Step 1: Load existing L1 results
    logger.info("\nStep 1: Loading existing L1 results...")
    l1_chunk_assignments, l1_report_categories = mapper.load_existing_l1_results(l1_run_id)

    # Step 2: Filter to L2-enabled categories only
    logger.info("\nStep 2: Filtering to L2-enabled categories...")
    l2_enabled = set(config.l2_enabled_categories)
    eligible_chunks = l1_chunk_assignments[
        l1_chunk_assignments["category_code"].isin(l2_enabled)
    ]
    logger.info(f"Found {len(eligible_chunks):,} chunk assignments with L2-enabled categories")
    logger.info(f"From {eligible_chunks['chunk_id'].nunique():,} unique chunks")

    # Step 3: Load embeddings for eligible chunks
    logger.info("\nStep 3: Loading MIKA embeddings for eligible chunks...")
    eligible_chunk_ids = eligible_chunks["chunk_id"].unique().tolist()
    embeddings_df = mapper.load_chunk_embeddings(eligible_chunk_ids)

    # Build embedding lookup
    embedding_lookup = dict(zip(embeddings_df["chunk_id"], embeddings_df["embedding"]))

    # Prepare aligned arrays
    chunk_ids_with_emb = [cid for cid in eligible_chunk_ids if cid in embedding_lookup]
    chunk_embeddings = np.vstack([embedding_lookup[cid] for cid in chunk_ids_with_emb])

    # Create chunk_id to report_id mapping
    chunk_to_report = dict(zip(
        l1_chunk_assignments["chunk_id"],
        l1_chunk_assignments["report_id"]
    ))
    report_ids = [chunk_to_report[cid] for cid in chunk_ids_with_emb]

    logger.info(f"Processing {len(chunk_ids_with_emb):,} chunks with embeddings")

    # Step 4: Run L2 classification
    logger.info("\nStep 4: Running L2 (subcategory) classification...")
    l2_start = time.time()

    # Filter L1 assignments to only those with embeddings
    l1_for_l2 = eligible_chunks[eligible_chunks["chunk_id"].isin(chunk_ids_with_emb)]

    l2_chunk_assignments = mapper.classify_l2(
        chunk_embeddings, chunk_ids_with_emb, report_ids, l1_for_l2
    )
    l2_time = time.time() - l2_start
    logger.info(f"L2 classification completed in {l2_time:.1f}s")

    # Step 5: Aggregate L2 to report level
    logger.info("\nStep 5: Aggregating L2 to report level...")
    l2_report_assignments = mapper.aggregate_l2_to_report(l2_chunk_assignments)

    # Step 6: Save L2 results
    logger.info("\nStep 6: Saving L2 results...")

    # Save chunk-level L2
    if not l2_chunk_assignments.empty:
        chunk_path = TAXONOMY_DATA_DIR / f"chunk_l2_run{l2_run_id}.parquet"
        l2_chunk_assignments.to_parquet(chunk_path, index=False)

    # Save report-level L2
    if not l2_report_assignments.empty:
        report_path = TAXONOMY_DATA_DIR / f"report_l2_run{l2_run_id}.parquet"
        l2_report_assignments.to_parquet(report_path, index=False)

    total_time = time.time() - start_time

    # Summary statistics
    stats = {
        "l1_run_id": l1_run_id,
        "l2_run_id": l2_run_id,
        "timestamp": datetime.now().isoformat(),
        "config_version": config.version,
        "l1_source": {
            "chunk_assignments": len(l1_chunk_assignments),
            "report_categories": len(l1_report_categories),
            "unique_reports": l1_chunk_assignments["report_id"].nunique(),
        },
        "l2": {
            "eligible_chunks": len(eligible_chunks),
            "processed_chunks": len(chunk_ids_with_emb),
            "chunk_assignments": len(l2_chunk_assignments),
            "report_assignments": len(l2_report_assignments),
            "subcategories_used": l2_report_assignments["subcategory_code"].nunique() if not l2_report_assignments.empty else 0,
            "reports_with_l2": l2_report_assignments["report_id"].nunique() if not l2_report_assignments.empty else 0,
            "time_sec": round(l2_time, 2),
        },
        "performance": {
            "total_time_sec": round(total_time, 2),
            "chunks_per_sec": round(len(chunk_ids_with_emb) / l2_time, 1) if l2_time > 0 else 0,
        },
    }

    # Save stats
    stats_path = TAXONOMY_DATA_DIR / f"l2_stats_run{l2_run_id}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("L2 Subcategory Classification Complete!")
    logger.info("=" * 60)
    logger.info(f"L1 source run: {l1_run_id}")
    logger.info(f"Chunks processed: {len(chunk_ids_with_emb):,}")
    logger.info(f"L2 chunk assignments: {len(l2_chunk_assignments):,}")
    logger.info(f"L2 report assignments: {len(l2_report_assignments):,}")
    logger.info(f"Subcategories used: {stats['l2']['subcategories_used']}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info("=" * 60)

    return {
        "stats": stats,
        "l1_chunks": l1_chunk_assignments,
        "l1_reports": l1_report_categories,
        "l2_chunks": l2_chunk_assignments,
        "l2_reports": l2_report_assignments,
    }


def get_classification_summary(run_id: int = 1) -> dict:
    """
    Get summary of a classification run.

    Args:
        run_id: Run ID to summarize

    Returns:
        Dict with summary statistics
    """
    results = load_hierarchical_results(run_id)

    if not results:
        return {"error": f"No results found for run {run_id}"}

    summary = {
        "run_id": run_id,
        "stats": results.get("stats", {}),
    }

    # L1 category distribution
    if "l1_reports" in results:
        l1_df = results["l1_reports"]
        l1_dist = l1_df.groupby("category_code").agg({
            "report_id": "count",
            "pct_contribution": "mean",
        }).rename(columns={"report_id": "n_reports", "pct_contribution": "avg_pct"})
        l1_dist = l1_dist.sort_values("n_reports", ascending=False)
        summary["l1_distribution"] = l1_dist.to_dict("index")

    # L2 subcategory distribution
    if "l2_reports" in results:
        l2_df = results["l2_reports"]
        l2_dist = l2_df.groupby(["parent_code", "subcategory_code"]).agg({
            "report_id": "count",
            "pct_of_parent": "mean",
        }).rename(columns={"report_id": "n_reports", "pct_of_parent": "avg_pct"})
        l2_dist = l2_dist.sort_values("n_reports", ascending=False)
        summary["l2_distribution"] = {
            f"{idx[0]}|{idx[1]}": v for idx, v in l2_dist.to_dict("index").items()
        }

    return summary


def compare_runs(run_id_1: int, run_id_2: int) -> dict:
    """
    Compare two classification runs.

    Args:
        run_id_1: First run ID
        run_id_2: Second run ID

    Returns:
        Dict with comparison statistics
    """
    results_1 = load_hierarchical_results(run_id_1)
    results_2 = load_hierarchical_results(run_id_2)

    if not results_1:
        return {"error": f"No results found for run {run_id_1}"}
    if not results_2:
        return {"error": f"No results found for run {run_id_2}"}

    comparison = {
        "run_1": run_id_1,
        "run_2": run_id_2,
    }

    # Compare stats
    stats_1 = results_1.get("stats", {})
    stats_2 = results_2.get("stats", {})

    comparison["stats_comparison"] = {
        "l1_chunk_assignments": {
            "run_1": stats_1.get("l1_chunk_assignments", 0),
            "run_2": stats_2.get("l1_chunk_assignments", 0),
        },
        "l1_categories_used": {
            "run_1": stats_1.get("l1_categories_used", 0),
            "run_2": stats_2.get("l1_categories_used", 0),
        },
        "l2_subcategories_used": {
            "run_1": stats_1.get("l2_subcategories_used", 0),
            "run_2": stats_2.get("l2_subcategories_used", 0),
        },
    }

    # Compare L1 report assignments
    if "l1_reports" in results_1 and "l1_reports" in results_2:
        l1_1 = results_1["l1_reports"][["report_id", "category_code", "rank"]]
        l1_2 = results_2["l1_reports"][["report_id", "category_code", "rank"]]

        # Merge to find differences
        merged = l1_1.merge(
            l1_2, on=["report_id", "category_code"], how="outer",
            suffixes=("_1", "_2")
        )

        comparison["l1_agreement"] = {
            "total_assignments_1": len(l1_1),
            "total_assignments_2": len(l1_2),
            "matching": len(merged[(merged["rank_1"].notna()) & (merged["rank_2"].notna())]),
            "only_in_1": len(merged[merged["rank_2"].isna()]),
            "only_in_2": len(merged[merged["rank_1"].isna()]),
        }

    return comparison


if __name__ == "__main__":
    # Basic CLI for testing - prefer using taxonomy.cli instead
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run L2 classification (use taxonomy.cli for full CLI)")
    parser.add_argument("command", choices=["classify-l2", "summary", "compare"])
    parser.add_argument("--l1-run", type=int, default=1, help="L1 run ID to build on")
    parser.add_argument("--l2-run", type=int, default=1, help="L2 run ID")
    parser.add_argument("--run-id-2", type=int, help="Second run ID for comparison")

    args = parser.parse_args()

    if args.command == "classify-l2":
        results = run_l2_classification(
            l1_run_id=args.l1_run,
            l2_run_id=args.l2_run,
        )
        print(json.dumps(results["stats"], indent=2))

    elif args.command == "summary":
        summary = get_classification_summary(args.l2_run)
        print(json.dumps(summary, indent=2, default=str))

    elif args.command == "compare":
        if not args.run_id_2:
            print("Error: --run-id-2 required for comparison")
        else:
            comparison = compare_runs(args.l2_run, args.run_id_2)
            print(json.dumps(comparison, indent=2))
