"""
eval/benchmark.py
-----------------
Professional benchmark runner for comparing embedding model retrieval quality.

Features:
- 50 stratified queries across 6 categories
- Latency tracking (embed + search time)
- Statistical significance tests (bootstrap CI, Wilcoxon)
- Parquet output for Streamlit visualization
- Comprehensive markdown report generation

Usage:
    python -m eval.benchmark run              # Run both models
    python -m eval.benchmark run -m minilm    # Single model
    python -m eval.benchmark report           # Generate comparison report
    python -m eval.benchmark validate         # Validate ground truth
"""

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import yaml
from qdrant_client import QdrantClient

from embeddings.config import MODELS, get_model_config
from embeddings.models import load_model
from riskradar.config import get_qdrant_config

logger = logging.getLogger(__name__)

# Paths
EVAL_DIR = Path(__file__).parent
GOLD_QUERIES_PATH = EVAL_DIR / "gold_queries.yaml"
RESULTS_DIR = EVAL_DIR / "results"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    # Query metadata
    query_id: str
    query_text: str
    category: str
    difficulty: str
    intent: str
    model: str

    # Retrieval results
    retrieved_report_ids: List[str] = field(default_factory=list)
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_sections: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)

    # Ground truth
    expected_report_ids: List[str] = field(default_factory=list)
    expected_sections: List[str] = field(default_factory=list)

    # Core metrics
    mrr: float = 0.0
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    hit_at_20: bool = False
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    ndcg_at_10: float = 0.0
    section_accuracy_at_10: float = 0.0

    # Latency (milliseconds)
    embed_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


@dataclass
class BenchmarkRun:
    """Complete benchmark run metadata and results."""
    run_id: str
    model_name: str
    timestamp: str
    duration_sec: float = 0.0

    # Query counts
    total_queries: int = 0
    queries_by_category: Dict[str, int] = field(default_factory=dict)
    queries_by_difficulty: Dict[str, int] = field(default_factory=dict)

    # Aggregate metrics
    mean_mrr: float = 0.0
    mean_hit_at_10: float = 0.0
    mean_precision_at_10: float = 0.0
    mean_recall_at_10: float = 0.0
    mean_ndcg_at_10: float = 0.0
    mean_section_accuracy: float = 0.0

    # Latency stats
    mean_embed_latency_ms: float = 0.0
    mean_search_latency_ms: float = 0.0
    mean_total_latency_ms: float = 0.0
    p95_total_latency_ms: float = 0.0

    # Stratified metrics
    metrics_by_category: Dict[str, Dict] = field(default_factory=dict)
    metrics_by_difficulty: Dict[str, Dict] = field(default_factory=dict)

    # Individual results
    query_results: List[QueryResult] = field(default_factory=list)


# =============================================================================
# METRIC CALCULATIONS
# =============================================================================

def compute_mrr(retrieved: List[str], relevant: set) -> float:
    """Mean Reciprocal Rank - position of first relevant result."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Precision@K - fraction of top K that are relevant."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / k


def compute_recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Recall@K - fraction of relevant found in top K."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & relevant)
    return hits / len(relevant)


def compute_ndcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if relevant, 0 otherwise).
    Only counts the FIRST occurrence of each relevant item to avoid
    inflated scores when multiple chunks from the same report are retrieved.
    """
    if not relevant or k == 0:
        return 0.0

    # DCG - only count first occurrence of each relevant item
    dcg = 0.0
    seen_relevant = set()
    for i, item in enumerate(retrieved[:k]):
        if item in relevant and item not in seen_relevant:
            dcg += 1.0 / math.log2(i + 2)  # +2 because positions are 1-indexed
            seen_relevant.add(item)

    # Ideal DCG (all relevant items at top positions)
    ideal_k = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


def compute_section_accuracy(retrieved_sections: List[str], expected_sections: set, k: int) -> float:
    """
    Fraction of top K results from expected sections.

    Uses case-insensitive substring matching to handle variations like:
    - "(b) PROBABLE CAUSE" matches "PROBABLE CAUSE"
    - "probable cause" matches "PROBABLE CAUSE"
    """
    if not expected_sections or k == 0:
        return 0.0

    # Normalize expected sections to uppercase for comparison
    expected_upper = {s.upper() for s in expected_sections}

    retrieved_k = retrieved_sections[:k]
    hits = 0
    for retrieved in retrieved_k:
        if not retrieved:
            continue
        retrieved_upper = retrieved.upper()
        # Check if retrieved section matches or contains any expected section
        for expected in expected_upper:
            if expected in retrieved_upper or retrieved_upper in expected:
                hits += 1
                break

    return hits / k


# =============================================================================
# QDRANT SEARCH
# =============================================================================

def get_qdrant_client() -> QdrantClient:
    """Create Qdrant client from configuration."""
    config = get_qdrant_config()
    return QdrantClient(url=config["url"], api_key=config["api_key"])


def check_chunk_relevance(chunk_text: str, relevance_signals: List[str], min_signals: int = 2) -> bool:
    """
    Check if a chunk is relevant based on relevance signals.

    A chunk is considered relevant if it contains at least min_signals
    from the relevance_signals list.
    """
    if not chunk_text or not relevance_signals:
        return False

    chunk_lower = chunk_text.lower()
    matches = sum(1 for signal in relevance_signals if signal.lower() in chunk_lower)
    return matches >= min_signals


def search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 20
) -> Tuple[List[Dict], float]:
    """
    Search Qdrant and return results with timing.

    Returns:
        (results, search_time_ms)
    """
    start = time.perf_counter()
    # Use query_points API (qdrant-client >= 1.7)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    search_time = (time.perf_counter() - start) * 1000

    parsed = [
        {
            "chunk_id": hit.payload.get("chunk_id", "") if hit.payload else "",
            "report_id": hit.payload.get("report_id", "") if hit.payload else "",
            "section_name": hit.payload.get("section_name", "") if hit.payload else "",
            "chunk_text": hit.payload.get("chunk_text", "") if hit.payload else "",
            "score": hit.score,
        }
        for hit in response.points
    ]

    return parsed, search_time


# =============================================================================
# QUERY EVALUATION
# =============================================================================

def evaluate_query(
    query_data: Dict,
    category: str,
    model_name: str,
    client: QdrantClient,
    model,
    universal_sections: List[str] = None,
    chunks_df: Optional[pd.DataFrame] = None,
) -> QueryResult:
    """Evaluate a single query against a model with full metrics."""
    config = get_model_config(model_name)
    query_text = query_data["query"]
    query_id = query_data["id"]
    difficulty = query_data.get("difficulty", "medium")
    intent = query_data.get("intent", "lookup")

    # Embed query with timing
    embed_start = time.perf_counter()
    query_vector = model.encode(query_text).tolist()
    embed_time = (time.perf_counter() - embed_start) * 1000

    # Search with timing
    results, search_time = search_qdrant(client, config.collection_name, query_vector, limit=20)

    # Extract retrieved items
    retrieved_report_ids = [r["report_id"] for r in results]
    retrieved_chunk_ids = [r["chunk_id"] for r in results]
    retrieved_sections = [r["section_name"] for r in results]
    retrieved_scores = [r["score"] for r in results]

    # Get chunk_text - first try from Qdrant payload, then fall back to chunks_df
    retrieved_texts = []
    for r in results:
        chunk_text = r.get("chunk_text", "")
        # If no text from Qdrant, try to look up in chunks_df
        if not chunk_text and chunks_df is not None:
            chunk_id = r.get("chunk_id", "")
            if chunk_id and chunk_id in chunks_df.index:
                chunk_text = str(chunks_df.loc[chunk_id].get("chunk_text", ""))
        retrieved_texts.append(chunk_text)

    # Get ground truth based on category
    ground_truth = query_data.get("ground_truth", {})
    relevance_signals = ground_truth.get("relevance_signals", [])

    # Determine relevance for each retrieved result
    # For categories with expected_report_ids, use those
    # For categories without, use relevance_signals to determine if chunks are relevant
    expected_report_ids = set(ground_truth.get("expected_report_ids", []))

    # If no expected_report_ids but we have relevance_signals, compute relevance dynamically
    use_signal_relevance = False
    relevant_indices = set()

    if category in ["incident_lookup", "conceptual_queries"]:
        # These should have expected_report_ids
        expected_sections = set()
    elif category == "section_queries":
        expected_report_ids = set()
        # Use primary_sections (new format) or fall back to target_sections (old format)
        primary = ground_truth.get("primary_sections", ground_truth.get("target_sections", []))
        expected_sections = set(primary)
        # Add universal sections (SYNOPSIS, SUMMARY, etc.) as valid for all section queries
        if universal_sections:
            expected_sections.update(universal_sections)
    elif category in ["comparative_queries", "aircraft_queries", "phase_queries"]:
        # These use relevance_signals for dynamic evaluation
        expected_sections = set()
        if relevance_signals and not expected_report_ids:
            use_signal_relevance = True
            # Check each retrieved chunk for relevance based on signals
            for i, text in enumerate(retrieved_texts):
                if check_chunk_relevance(text, relevance_signals, min_signals=2):
                    relevant_indices.add(i)
    else:
        expected_report_ids = set()
        expected_sections = set()

    # Compute metrics
    result = QueryResult(
        query_id=query_id,
        query_text=query_text,
        category=category,
        difficulty=difficulty,
        intent=intent,
        model=model_name,
        retrieved_report_ids=retrieved_report_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_sections=retrieved_sections,
        retrieved_scores=retrieved_scores,
        expected_report_ids=list(expected_report_ids),
        expected_sections=list(expected_sections),
        embed_latency_ms=embed_time,
        search_latency_ms=search_time,
        total_latency_ms=embed_time + search_time,
    )

    # Report-based metrics (when we have expected_report_ids)
    if expected_report_ids:
        result.mrr = compute_mrr(retrieved_report_ids, expected_report_ids)
        result.hit_at_1 = any(r in expected_report_ids for r in retrieved_report_ids[:1])
        result.hit_at_3 = any(r in expected_report_ids for r in retrieved_report_ids[:3])
        result.hit_at_5 = any(r in expected_report_ids for r in retrieved_report_ids[:5])
        result.hit_at_10 = any(r in expected_report_ids for r in retrieved_report_ids[:10])
        result.hit_at_20 = any(r in expected_report_ids for r in retrieved_report_ids[:20])
        result.precision_at_5 = compute_precision_at_k(retrieved_report_ids, expected_report_ids, 5)
        result.precision_at_10 = compute_precision_at_k(retrieved_report_ids, expected_report_ids, 10)
        result.recall_at_5 = compute_recall_at_k(retrieved_report_ids, expected_report_ids, 5)
        result.recall_at_10 = compute_recall_at_k(retrieved_report_ids, expected_report_ids, 10)
        result.recall_at_20 = compute_recall_at_k(retrieved_report_ids, expected_report_ids, 20)
        result.ndcg_at_10 = compute_ndcg_at_k(retrieved_report_ids, expected_report_ids, 10)

    # Signal-based metrics (when we use relevance_signals to evaluate)
    elif use_signal_relevance and relevant_indices:
        # MRR: reciprocal rank of first relevant result
        for i in range(len(retrieved_chunk_ids)):
            if i in relevant_indices:
                result.mrr = 1.0 / (i + 1)
                break

        # Hit@K: any relevant in top K
        result.hit_at_1 = 0 in relevant_indices
        result.hit_at_3 = any(i in relevant_indices for i in range(min(3, len(retrieved_chunk_ids))))
        result.hit_at_5 = any(i in relevant_indices for i in range(min(5, len(retrieved_chunk_ids))))
        result.hit_at_10 = any(i in relevant_indices for i in range(min(10, len(retrieved_chunk_ids))))
        result.hit_at_20 = any(i in relevant_indices for i in range(min(20, len(retrieved_chunk_ids))))

        # Precision@K: fraction of top K that are relevant
        result.precision_at_5 = sum(1 for i in range(min(5, len(retrieved_chunk_ids))) if i in relevant_indices) / 5
        result.precision_at_10 = sum(1 for i in range(min(10, len(retrieved_chunk_ids))) if i in relevant_indices) / 10

        # nDCG@10
        dcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(retrieved_chunk_ids))) if i in relevant_indices)
        ideal_k = min(len(relevant_indices), 10)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k)) if ideal_k > 0 else 0
        result.ndcg_at_10 = dcg / idcg if idcg > 0 else 0.0

    # Section-based metrics
    if expected_sections:
        result.section_accuracy_at_10 = compute_section_accuracy(retrieved_sections, expected_sections, 10)
        # For section queries, use section accuracy as a proxy for hit
        result.hit_at_10 = result.section_accuracy_at_10 > 0

    return result


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def load_gold_queries() -> Dict:
    """Load gold standard queries from YAML."""
    with open(GOLD_QUERIES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks_for_relevance() -> Optional[pd.DataFrame]:
    """Load chunks parquet for relevance checking."""
    chunks_path = Path(__file__).parent.parent / "analytics" / "data" / "chunks.parquet"
    if chunks_path.exists():
        try:
            df = pd.read_parquet(chunks_path)
            df = df.set_index("chunk_id")
            logger.info(f"Loaded {len(df)} chunks for relevance checking")
            return df
        except Exception as e:
            logger.warning(f"Could not load chunks parquet: {e}")
    return None


def run_benchmark(model_name: str) -> BenchmarkRun:
    """Run complete benchmark for a model."""
    run_start = time.time()
    run_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Starting benchmark run: {run_id}")

    # Load resources
    gold_queries = load_gold_queries()
    client = get_qdrant_client()

    # Load chunks for relevance checking (signal-based evaluation)
    chunks_df = load_chunks_for_relevance()

    # Load universal sections (valid for all section queries)
    universal_sections = gold_queries.get("universal_valid_sections", [])
    logger.info(f"Universal valid sections: {universal_sections}")

    logger.info(f"Loading model: {model_name}")
    model = load_model(model_name)

    run = BenchmarkRun(
        run_id=run_id,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )

    # Define categories to evaluate
    categories = [
        "incident_lookup",
        "conceptual_queries",
        "section_queries",
        "comparative_queries",
        "aircraft_queries",
        "phase_queries",
    ]

    # Evaluate each category
    for category in categories:
        queries = gold_queries.get(category, [])
        if not queries:
            continue

        logger.info(f"Evaluating {len(queries)} {category} queries")
        run.queries_by_category[category] = len(queries)

        for query_data in queries:
            result = evaluate_query(query_data, category, model_name, client, model, universal_sections, chunks_df)
            run.query_results.append(result)

            # Track difficulty
            difficulty = result.difficulty
            run.queries_by_difficulty[difficulty] = run.queries_by_difficulty.get(difficulty, 0) + 1

            logger.debug(f"  {result.query_id}: MRR={result.mrr:.3f}, latency={result.total_latency_ms:.1f}ms")

    run.total_queries = len(run.query_results)
    run.duration_sec = time.time() - run_start

    # Compute aggregate metrics
    _compute_aggregates(run)

    logger.info(f"Benchmark complete: {run.total_queries} queries in {run.duration_sec:.1f}s")

    return run


def _compute_aggregates(run: BenchmarkRun):
    """Compute aggregate metrics from individual query results."""
    if not run.query_results:
        return

    results = run.query_results

    # Categories that use different evaluation modes:
    # - Report-based: incident_lookup, conceptual_queries (have expected_report_ids)
    # - Section-based: section_queries (have expected_sections, MRR not applicable)
    # - Signal-based: comparative_queries, aircraft_queries, phase_queries (MRR via relevance_signals)
    SIGNAL_BASED_CATEGORIES = {"comparative_queries", "aircraft_queries", "phase_queries"}

    # Results that have MRR computed (either report-based or signal-based)
    # Exclude section_queries which are evaluated on section_accuracy only
    mrr_results = [r for r in results if r.expected_report_ids or r.category in SIGNAL_BASED_CATEGORIES]
    section_results = [r for r in results if r.expected_sections]

    if mrr_results:
        run.mean_mrr = np.mean([r.mrr for r in mrr_results])
        run.mean_hit_at_10 = np.mean([r.hit_at_10 for r in mrr_results])
        run.mean_precision_at_10 = np.mean([r.precision_at_10 for r in mrr_results])
        run.mean_recall_at_10 = np.mean([r.recall_at_10 for r in mrr_results])
        run.mean_ndcg_at_10 = np.mean([r.ndcg_at_10 for r in mrr_results])

    if section_results:
        run.mean_section_accuracy = np.mean([r.section_accuracy_at_10 for r in section_results])

    # Latency stats
    latencies = [r.total_latency_ms for r in results]
    run.mean_embed_latency_ms = np.mean([r.embed_latency_ms for r in results])
    run.mean_search_latency_ms = np.mean([r.search_latency_ms for r in results])
    run.mean_total_latency_ms = np.mean(latencies)
    run.p95_total_latency_ms = np.percentile(latencies, 95)

    # Metrics by category
    for category in run.queries_by_category:
        cat_results = [r for r in results if r.category == category]
        cat_section_results = [r for r in cat_results if r.expected_sections]

        # Determine which results have MRR computed for this category
        if category in SIGNAL_BASED_CATEGORIES:
            # Signal-based categories: all results have MRR computed
            cat_mrr_results = cat_results
        elif category == "section_queries":
            # Section queries: MRR not applicable
            cat_mrr_results = []
        else:
            # Report-based categories: only results with expected_report_ids
            cat_mrr_results = [r for r in cat_results if r.expected_report_ids]

        metrics = {"count": len(cat_results)}
        if cat_mrr_results:
            metrics["mrr"] = np.mean([r.mrr for r in cat_mrr_results])
            metrics["hit_at_10"] = np.mean([r.hit_at_10 for r in cat_mrr_results])
            metrics["precision_at_10"] = np.mean([r.precision_at_10 for r in cat_mrr_results])
            metrics["ndcg_at_10"] = np.mean([r.ndcg_at_10 for r in cat_mrr_results])
        if cat_section_results:
            metrics["section_accuracy"] = np.mean([r.section_accuracy_at_10 for r in cat_section_results])
        metrics["latency_ms"] = np.mean([r.total_latency_ms for r in cat_results])

        run.metrics_by_category[category] = metrics

    # Metrics by difficulty
    for difficulty in run.queries_by_difficulty:
        diff_results = [r for r in results if r.difficulty == difficulty]
        # Include both report-based and signal-based results
        diff_mrr_results = [r for r in diff_results
                           if r.expected_report_ids or r.category in SIGNAL_BASED_CATEGORIES]

        metrics = {"count": len(diff_results)}
        if diff_mrr_results:
            metrics["mrr"] = np.mean([r.mrr for r in diff_mrr_results])
            metrics["hit_at_10"] = np.mean([r.hit_at_10 for r in diff_mrr_results])
        metrics["latency_ms"] = np.mean([r.total_latency_ms for r in diff_results])

        run.metrics_by_difficulty[difficulty] = metrics


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_results(run: BenchmarkRun) -> Tuple[Path, Path]:
    """Save benchmark results to JSON and Parquet."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Save detailed JSON
    json_path = RESULTS_DIR / f"benchmark_{run.run_id}.json"

    run_dict = {
        "run_id": run.run_id,
        "model_name": run.model_name,
        "timestamp": run.timestamp,
        "duration_sec": run.duration_sec,
        "total_queries": run.total_queries,
        "queries_by_category": run.queries_by_category,
        "queries_by_difficulty": run.queries_by_difficulty,
        "aggregate_metrics": {
            "mean_mrr": run.mean_mrr,
            "mean_hit_at_10": run.mean_hit_at_10,
            "mean_precision_at_10": run.mean_precision_at_10,
            "mean_recall_at_10": run.mean_recall_at_10,
            "mean_ndcg_at_10": run.mean_ndcg_at_10,
            "mean_section_accuracy": run.mean_section_accuracy,
        },
        "latency_stats": {
            "mean_embed_ms": run.mean_embed_latency_ms,
            "mean_search_ms": run.mean_search_latency_ms,
            "mean_total_ms": run.mean_total_latency_ms,
            "p95_total_ms": run.p95_total_latency_ms,
        },
        "metrics_by_category": run.metrics_by_category,
        "metrics_by_difficulty": run.metrics_by_difficulty,
        "query_results": [asdict(r) for r in run.query_results],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_dict, f, indent=2, default=str)

    # Save Parquet for Streamlit (flattened query results)
    parquet_path = RESULTS_DIR / f"benchmark_{run.run_id}.parquet"

    df_rows = []
    for r in run.query_results:
        df_rows.append({
            "query_id": r.query_id,
            "query_text": r.query_text,
            "category": r.category,
            "difficulty": r.difficulty,
            "intent": r.intent,
            "model": r.model,
            "mrr": r.mrr,
            "hit_at_1": r.hit_at_1,
            "hit_at_3": r.hit_at_3,
            "hit_at_5": r.hit_at_5,
            "hit_at_10": r.hit_at_10,
            "hit_at_20": r.hit_at_20,
            "precision_at_5": r.precision_at_5,
            "precision_at_10": r.precision_at_10,
            "recall_at_5": r.recall_at_5,
            "recall_at_10": r.recall_at_10,
            "recall_at_20": r.recall_at_20,
            "ndcg_at_10": r.ndcg_at_10,
            "section_accuracy_at_10": r.section_accuracy_at_10,
            "embed_latency_ms": r.embed_latency_ms,
            "search_latency_ms": r.search_latency_ms,
            "total_latency_ms": r.total_latency_ms,
            "top_result_report": r.retrieved_report_ids[0] if r.retrieved_report_ids else "",
            "top_result_score": r.retrieved_scores[0] if r.retrieved_scores else 0.0,
        })

    df = pd.DataFrame(df_rows)
    df.to_parquet(parquet_path, index=False)

    logger.info(f"Results saved to {json_path} and {parquet_path}")

    return json_path, parquet_path


def compute_statistical_comparison(minilm_run: BenchmarkRun, mika_run: BenchmarkRun) -> Dict:
    """Compute statistical significance between two models."""
    from scipy import stats

    # Align queries by ID
    minilm_mrr = {r.query_id: r.mrr for r in minilm_run.query_results}
    mika_mrr = {r.query_id: r.mrr for r in mika_run.query_results}

    common_ids = set(minilm_mrr.keys()) & set(mika_mrr.keys())
    if len(common_ids) < 10:
        return {"error": "Not enough common queries for statistical tests"}

    minilm_scores = [minilm_mrr[qid] for qid in common_ids]
    mika_scores = [mika_mrr[qid] for qid in common_ids]

    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(mika_scores, minilm_scores)

    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pvalue = stats.wilcoxon(mika_scores, minilm_scores)
    except ValueError:
        w_stat, w_pvalue = None, None

    # Bootstrap confidence interval for difference
    n_bootstrap = 1000
    diffs = np.array(mika_scores) - np.array(minilm_scores)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(diffs, size=len(diffs), replace=True)
        bootstrap_means.append(np.mean(sample))

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    # Win/loss/tie analysis
    wins = sum(1 for m, n in zip(mika_scores, minilm_scores) if m > n + 0.01)
    losses = sum(1 for m, n in zip(mika_scores, minilm_scores) if n > m + 0.01)
    ties = len(common_ids) - wins - losses

    return {
        "n_queries": len(common_ids),
        "minilm_mean_mrr": np.mean(minilm_scores),
        "mika_mean_mrr": np.mean(mika_scores),
        "difference": np.mean(mika_scores) - np.mean(minilm_scores),
        "paired_ttest": {
            "t_statistic": t_stat,
            "p_value": t_pvalue,
            "significant_at_05": t_pvalue < 0.05 if t_pvalue else False,
        },
        "wilcoxon": {
            "statistic": w_stat,
            "p_value": w_pvalue,
            "significant_at_05": w_pvalue < 0.05 if w_pvalue else False,
        },
        "bootstrap_95_ci": {
            "lower": ci_lower,
            "upper": ci_upper,
            "significant": ci_lower > 0 or ci_upper < 0,
        },
        "win_loss_tie": {
            "mika_wins": wins,
            "minilm_wins": losses,
            "ties": ties,
        },
    }


def generate_report(minilm_path: Optional[Path] = None, mika_path: Optional[Path] = None) -> Path:
    """Generate comprehensive markdown report comparing models."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Find most recent results
    if minilm_path is None:
        minilm_files = sorted(RESULTS_DIR.glob("benchmark_minilm_*.json"), reverse=True)
        minilm_path = minilm_files[0] if minilm_files else None

    if mika_path is None:
        mika_files = sorted(RESULTS_DIR.glob("benchmark_mika_*.json"), reverse=True)
        mika_path = mika_files[0] if mika_files else None

    # Load results
    minilm_data = None
    mika_data = None

    if minilm_path and minilm_path.exists():
        with open(minilm_path, encoding='utf-8') as f:
            minilm_data = json.load(f)

    if mika_path and mika_path.exists():
        with open(mika_path, encoding='utf-8') as f:
            mika_data = json.load(f)

    if not minilm_data and not mika_data:
        raise FileNotFoundError("No benchmark results found. Run 'benchmark run' first.")

    # Generate report
    report_lines = _build_report_markdown(minilm_data, mika_data)

    report_path = EVAL_DIR / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Report saved to {report_path}")
    return report_path


def _build_report_markdown(minilm_data: Optional[Dict], mika_data: Optional[Dict]) -> List[str]:
    """Build markdown report content."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# RiskRADAR Embedding Model Benchmark Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Benchmark Version:** 2.0",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    # Determine winner
    if minilm_data and mika_data:
        minilm_mrr = minilm_data["aggregate_metrics"]["mean_mrr"]
        mika_mrr = mika_data["aggregate_metrics"]["mean_mrr"]
        diff = mika_mrr - minilm_mrr
        winner = "MIKA" if diff > 0.02 else ("MiniLM" if diff < -0.02 else "No clear winner")

        lines.extend([
            f"| Model | Mean MRR | Hit@10 | nDCG@10 | Latency (p95) |",
            f"|-------|----------|--------|---------|---------------|",
            f"| MiniLM | {minilm_mrr:.3f} | {minilm_data['aggregate_metrics']['mean_hit_at_10']:.1%} | {minilm_data['aggregate_metrics']['mean_ndcg_at_10']:.3f} | {minilm_data['latency_stats']['p95_total_ms']:.0f}ms |",
            f"| MIKA | {mika_mrr:.3f} | {mika_data['aggregate_metrics']['mean_hit_at_10']:.1%} | {mika_data['aggregate_metrics']['mean_ndcg_at_10']:.3f} | {mika_data['latency_stats']['p95_total_ms']:.0f}ms |",
            "",
            f"**Recommendation:** {winner} (MRR difference: {diff:+.3f})",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Methodology",
        "",
        "### Query Design",
        "",
        "| Category | Count | Difficulty Mix | Purpose |",
        "|----------|-------|----------------|---------|",
        "| Incident Lookup | 10 | Easy | Known accidents with specific report IDs |",
        "| Conceptual Queries | 12 | Medium-Hard | Technical concepts requiring semantic understanding |",
        "| Section Queries | 10 | Medium | Queries targeting specific report sections |",
        "| Comparative Queries | 8 | Hard | Analytical queries about patterns |",
        "| Aircraft Queries | 6 | Medium | Aircraft-type specific searches |",
        "| Phase Queries | 4 | Medium | Flight phase specific searches |",
        "",
        "### Ground Truth Verification",
        "",
        "All ground truth was established via SQL queries against `chunks.parquet`:",
        "- **Incident queries:** Report IDs verified from NTSB report metadata",
        "- **Conceptual queries:** Term co-occurrence verified via LIKE patterns",
        "- **Section queries:** Section names verified from chunk metadata",
        "",
        "### Metrics",
        "",
        "| Metric | Description |",
        "|--------|-------------|",
        "| MRR | Mean Reciprocal Rank - position of first relevant result |",
        "| Hit@K | Percentage with at least one relevant in top K |",
        "| Precision@K | Fraction of top K that are relevant |",
        "| Recall@K | Fraction of relevant found in top K |",
        "| nDCG@K | Normalized Discounted Cumulative Gain |",
        "| Section Accuracy | Fraction from expected sections (section queries) |",
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ])

    # Per-category breakdown
    for model_name, data in [("MiniLM", minilm_data), ("MIKA", mika_data)]:
        if not data:
            continue

        lines.extend([
            f"### {model_name}",
            "",
            "**Performance by Category:**",
            "",
            "| Category | MRR | Hit@10 | nDCG@10 | Latency |",
            "|----------|-----|--------|---------|---------|",
        ])

        for cat, metrics in data.get("metrics_by_category", {}).items():
            latency = metrics.get("latency_ms", 0)
            if cat == "section_queries":
                # Section queries use section_accuracy, not MRR-based metrics
                sec_acc = metrics.get("section_accuracy", 0)
                lines.append(f"| {cat} | *N/A* | *N/A* | *N/A* | {latency:.0f}ms |")
            else:
                mrr = metrics.get("mrr", 0)
                hit = metrics.get("hit_at_10", 0)
                ndcg = metrics.get("ndcg_at_10", 0)
                lines.append(f"| {cat} | {mrr:.3f} | {hit:.1%} | {ndcg:.3f} | {latency:.0f}ms |")

        # Add section_queries specific metrics in a separate note
        section_metrics = data.get("metrics_by_category", {}).get("section_queries", {})
        if section_metrics:
            sec_acc = section_metrics.get("section_accuracy", 0)
            lines.append("")
            lines.append(f"*Section queries evaluated on Section Accuracy: **{sec_acc:.1%}***")

        lines.extend([
            "",
            "**Performance by Difficulty:**",
            "",
            "| Difficulty | MRR | Hit@10 | Latency |",
            "|------------|-----|--------|---------|",
        ])

        for diff, metrics in data.get("metrics_by_difficulty", {}).items():
            mrr = metrics.get("mrr", 0)
            hit = metrics.get("hit_at_10", 0)
            latency = metrics.get("latency_ms", 0)
            lines.append(f"| {diff} | {mrr:.3f} | {hit:.1%} | {latency:.0f}ms |")

        lines.append("")

    # Statistical comparison
    if minilm_data and mika_data:
        lines.extend([
            "---",
            "",
            "## Statistical Analysis",
            "",
        ])

        try:
            # Compute stats from the loaded data
            minilm_mrr = {r["query_id"]: r["mrr"] for r in minilm_data.get("query_results", [])}
            mika_mrr = {r["query_id"]: r["mrr"] for r in mika_data.get("query_results", [])}
            common_ids = set(minilm_mrr.keys()) & set(mika_mrr.keys())

            if len(common_ids) >= 10:
                minilm_scores = [minilm_mrr[qid] for qid in common_ids]
                mika_scores = [mika_mrr[qid] for qid in common_ids]

                # Bootstrap CI
                diffs = np.array(mika_scores) - np.array(minilm_scores)
                bootstrap_means = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(1000)]
                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)

                # Win/loss/tie
                wins = sum(1 for m, n in zip(mika_scores, minilm_scores) if m > n + 0.01)
                losses = sum(1 for m, n in zip(mika_scores, minilm_scores) if n > m + 0.01)
                ties = len(common_ids) - wins - losses

                diff = np.mean(diffs)
                significant = ci_lower > 0 or ci_upper < 0

                lines.extend([
                    "### Bootstrap Confidence Interval (Primary)",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| MRR Difference (MIKA - MiniLM) | {diff:+.3f} |",
                    f"| 95% Confidence Interval | [{ci_lower:.3f}, {ci_upper:.3f}] |",
                    f"| Statistically Significant | {'Yes' if significant else 'No'} |",
                    "",
                    "**Interpretation:** " + (
                        f"MIKA is significantly better (CI excludes 0)." if ci_lower > 0 else
                        f"MiniLM is significantly better (CI excludes 0)." if ci_upper < 0 else
                        f"No significant difference detected (CI includes 0)."
                    ),
                    "",
                    "### Win/Loss/Tie Analysis",
                    "",
                    "| Model | Wins | Percentage |",
                    "|-------|------|------------|",
                    f"| MIKA | {wins} | {100*wins/len(common_ids):.1f}% |",
                    f"| MiniLM | {losses} | {100*losses/len(common_ids):.1f}% |",
                    f"| Tie | {ties} | {100*ties/len(common_ids):.1f}% |",
                    "",
                    f"*Note: A query is a 'tie' if MRR difference < 0.01*",
                    "",
                ])
        except Exception as e:
            lines.extend([
                f"*(Statistical analysis unavailable: {e})*",
                "",
            ])

    lines.extend([
        "---",
        "",
        "## Streamlit Visualization",
        "",
        "Results are saved in Parquet format for easy loading in Streamlit:",
        "",
        "```python",
        "import pandas as pd",
        "",
        "# Load benchmark results",
        "minilm_df = pd.read_parquet('eval/results/benchmark_minilm_*.parquet')",
        "mika_df = pd.read_parquet('eval/results/benchmark_mika_*.parquet')",
        "",
        "# Merge for comparison",
        "comparison = minilm_df.merge(",
        "    mika_df, on='query_id', suffixes=('_minilm', '_mika')",
        ")",
        "```",
        "",
        "---",
        "",
        "*Report generated by `eval/benchmark.py`*",
    ])

    return lines


# =============================================================================
# HUMAN REVIEW EXPORT/IMPORT
# =============================================================================

HUMAN_REVIEW_DIR = EVAL_DIR / "human_reviews"
KEYWORD_MATCH_DIR = HUMAN_REVIEW_DIR / "keyword_match"
MANUAL_REVIEW_DIR = HUMAN_REVIEW_DIR / "manual_review"
SEMANTIC_CATEGORIES = ["conceptual_queries", "comparative_queries"]

# Path to chunks data for loading text
CHUNKS_PARQUET = Path(__file__).parent.parent / "analytics" / "data" / "chunks.parquet"


def find_review_files(model_name: str = None, pattern: str = None) -> List[Path]:
    """
    Find all review files across HUMAN_REVIEW_DIR and its subdirectories.

    Args:
        model_name: If provided, filter to files matching this model (e.g., 'minilm', 'mika')
        pattern: Custom glob pattern. If not provided, uses default review_*.yaml

    Returns:
        List of Path objects for all matching review files
    """
    if pattern is None:
        if model_name:
            pattern = f"review_*_{model_name}.yaml"
        else:
            pattern = "review_*.yaml"

    all_files = []

    # Check root directory
    all_files.extend(HUMAN_REVIEW_DIR.glob(pattern))

    # Check subdirectories
    if KEYWORD_MATCH_DIR.exists():
        all_files.extend(KEYWORD_MATCH_DIR.glob(pattern))
    if MANUAL_REVIEW_DIR.exists():
        all_files.extend(MANUAL_REVIEW_DIR.glob(pattern))

    return list(all_files)


def _check_keyword_match(chunk_text: str, verification_sql: str, relevance_signals: List[str]) -> bool:
    """
    Check if a chunk matches the keyword-based ground truth.

    Returns True if the chunk would match the SQL ground truth pattern.
    """
    if not chunk_text:
        return False

    chunk_lower = chunk_text.lower()

    # Check relevance signals - if multiple signals present, likely a keyword match
    signals_found = sum(1 for sig in relevance_signals if sig.lower() in chunk_lower)
    if signals_found >= 2:
        return True

    # Try to parse simple SQL LIKE patterns
    if verification_sql:
        sql_lower = verification_sql.lower()
        # Extract LIKE patterns
        import re
        like_patterns = re.findall(r"like\s+['\"]%([^%'\"]+)%['\"]", sql_lower)
        if like_patterns:
            matches = sum(1 for pattern in like_patterns if pattern in chunk_lower)
            # If most patterns match, consider it a keyword match
            if matches >= len(like_patterns) * 0.5:
                return True

    return False


def export_for_human_review(model_name: str, fetch_text: bool = True) -> List[Path]:
    """
    Export benchmark results to YAML files for human review.

    Creates one file per query that needs human evaluation (conceptual + comparative).
    Auto-fills KEYWORD_MATCH judgments based on SQL ground truth.
    Human only needs to judge non-keyword results as SEMANTIC_MATCH or FALSE_POSITIVE.

    Files are organized into subdirectories:
    - keyword_match/: All results auto-filled, no human review needed
    - manual_review/: At least one result needs human judgment
    """
    HUMAN_REVIEW_DIR.mkdir(exist_ok=True)
    KEYWORD_MATCH_DIR.mkdir(exist_ok=True)
    MANUAL_REVIEW_DIR.mkdir(exist_ok=True)

    # Find most recent benchmark results
    result_files = sorted(RESULTS_DIR.glob(f"benchmark_{model_name}_*.json"), reverse=True)
    if not result_files:
        raise FileNotFoundError(f"No benchmark results found for {model_name}. Run 'benchmark run' first.")

    with open(result_files[0], encoding='utf-8') as f:
        benchmark_data = json.load(f)

    # Load gold queries for SQL ground truth reference
    gold_queries = load_gold_queries()
    gold_by_id = {}
    for cat in SEMANTIC_CATEGORIES:
        for q in gold_queries.get(cat, []):
            gold_by_id[q["id"]] = q

    # Load chunks parquet for text lookup
    chunks_df = None
    if fetch_text and CHUNKS_PARQUET.exists():
        try:
            chunks_df = pd.read_parquet(CHUNKS_PARQUET)
            chunks_df = chunks_df.set_index("chunk_id")
            logger.info(f"Loaded {len(chunks_df)} chunks for text lookup")
        except Exception as e:
            logger.warning(f"Could not load chunks parquet: {e}")

    exported_files = []
    total_auto_filled = 0

    for result in benchmark_data["query_results"]:
        if result["category"] not in SEMANTIC_CATEGORIES:
            continue

        query_id = result["query_id"]
        gold = gold_by_id.get(query_id, {})
        ground_truth = gold.get("ground_truth", {})

        verification_sql = ground_truth.get("verification_sql", "")
        relevance_signals = ground_truth.get("relevance_signals", [])
        expected_report_ids = set(ground_truth.get("expected_report_ids", []))

        # Build review document
        review_doc = {
            "metadata": {
                "query_id": query_id,
                "model": model_name,
                "category": result["category"],
                "difficulty": result["difficulty"],
                "exported_from": str(result_files[0].name),
                "reviewer": "",
                "review_date": "",
                "review_complete": False,
            },
            "query": {
                "text": result["query_text"],
                "sql_ground_truth": verification_sql,
                "expected_report_ids": list(expected_report_ids),
                "relevance_signals": relevance_signals,
            },
            "instructions": {
                "keyword_match": "Already filled in by automated check. Result matches SQL pattern or relevance signals.",
                "semantic_match": "YOU FILL THIS: Result is relevant but doesn't match keywords. This is what makes semantic search valuable!",
                "false_positive": "YOU FILL THIS: Result is NOT relevant to the query.",
                "note": "Only review results where judgment is empty. Pre-filled KEYWORD_MATCH can be changed if wrong.",
            },
            "automated_metrics": {
                "mrr": result["mrr"],
                "hit_at_10": result["hit_at_10"],
                "precision_at_10": result["precision_at_10"],
            },
            "results": [],
            "summary": {
                "auto_keyword_matches": 0,
                "needs_human_review": 0,
            },
        }

        # Add each result for review
        retrieved_chunks = result.get("retrieved_chunk_ids", [])
        retrieved_reports = result.get("retrieved_report_ids", [])
        retrieved_sections = result.get("retrieved_sections", [])
        retrieved_scores = result.get("retrieved_scores", [])

        auto_keyword_count = 0

        for i in range(min(10, len(retrieved_chunks))):
            chunk_id = retrieved_chunks[i] if i < len(retrieved_chunks) else ""
            report_id = retrieved_reports[i] if i < len(retrieved_reports) else ""

            # Get chunk text
            text_preview = ""
            chunk_text_full = ""
            if chunks_df is not None and chunk_id in chunks_df.index:
                chunk_text_full = str(chunks_df.loc[chunk_id].get("chunk_text", ""))
                text_preview = chunk_text_full[:500] + "..." if len(chunk_text_full) > 500 else chunk_text_full

            # Auto-fill keyword match check
            judgment = ""
            auto_note = ""

            # Check if result matches expected report IDs
            if report_id in expected_report_ids:
                judgment = "KEYWORD_MATCH"
                auto_note = "[AUTO] Report ID in expected list"
                auto_keyword_count += 1
            # Check if result matches SQL/signals
            elif _check_keyword_match(chunk_text_full, verification_sql, relevance_signals):
                judgment = "KEYWORD_MATCH"
                auto_note = "[AUTO] Matches relevance signals/SQL pattern"
                auto_keyword_count += 1

            review_doc["results"].append({
                "rank": i + 1,
                "chunk_id": chunk_id,
                "report_id": report_id,
                "section_name": retrieved_sections[i] if i < len(retrieved_sections) else "",
                "score": round(retrieved_scores[i], 4) if i < len(retrieved_scores) else 0.0,
                "text_preview": text_preview,
                "judgment": judgment,  # Pre-filled if KEYWORD_MATCH detected
                "notes": auto_note,  # Explains why auto-filled, human can override
            })

        review_doc["summary"]["auto_keyword_matches"] = auto_keyword_count
        review_doc["summary"]["needs_human_review"] = 10 - auto_keyword_count
        total_auto_filled += auto_keyword_count

        # Save to appropriate subdirectory based on review needs
        filename = f"review_{query_id}_{model_name}.yaml"
        needs_review = review_doc["summary"]["needs_human_review"]
        if needs_review > 0:
            filepath = MANUAL_REVIEW_DIR / filename
        else:
            filepath = KEYWORD_MATCH_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(review_doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

        exported_files.append(filepath)

    logger.info(f"Auto-filled {total_auto_filled} keyword matches across {len(exported_files)} queries")
    return exported_files


def import_human_reviews(model_name: str) -> Dict:
    """
    Import completed human review files and compute adjusted metrics.

    Returns aggregated human evaluation metrics.
    """
    review_files = find_review_files(model_name=model_name)

    if not review_files:
        raise FileNotFoundError(f"No human review files found for {model_name} in {HUMAN_REVIEW_DIR} or subdirectories")

    results = {
        "model": model_name,
        "queries_reviewed": 0,
        "queries_complete": 0,
        "total_results_reviewed": 0,
        "keyword_matches": 0,
        "semantic_matches": 0,
        "false_positives": 0,
        "per_query": [],
    }

    for filepath in review_files:
        with open(filepath, encoding="utf-8") as f:
            review = yaml.safe_load(f)

        results["queries_reviewed"] += 1

        if not review.get("metadata", {}).get("review_complete", False):
            logger.warning(f"Skipping incomplete review: {filepath.name}")
            continue

        results["queries_complete"] += 1

        # Count judgments
        query_keyword = 0
        query_semantic = 0
        query_false_pos = 0

        for r in review.get("results", []):
            judgment = r.get("judgment", "").upper()
            if judgment == "KEYWORD_MATCH":
                query_keyword += 1
                results["keyword_matches"] += 1
            elif judgment == "SEMANTIC_MATCH":
                query_semantic += 1
                results["semantic_matches"] += 1
            elif judgment == "FALSE_POSITIVE":
                query_false_pos += 1
                results["false_positives"] += 1
            results["total_results_reviewed"] += 1

        # Per-query metrics
        total_reviewed = query_keyword + query_semantic + query_false_pos
        if total_reviewed > 0:
            results["per_query"].append({
                "query_id": review["metadata"]["query_id"],
                "category": review["metadata"]["category"],
                "keyword_matches": query_keyword,
                "semantic_matches": query_semantic,
                "false_positives": query_false_pos,
                "semantic_precision": (query_keyword + query_semantic) / total_reviewed,
                "semantic_lift": query_semantic / total_reviewed,
                "automated_mrr": review.get("automated_metrics", {}).get("mrr", 0),
            })

    # Compute aggregate metrics
    if results["total_results_reviewed"] > 0:
        total = results["total_results_reviewed"]
        results["aggregate"] = {
            "keyword_match_rate": results["keyword_matches"] / total,
            "semantic_match_rate": results["semantic_matches"] / total,
            "false_positive_rate": results["false_positives"] / total,
            "semantic_precision": (results["keyword_matches"] + results["semantic_matches"]) / total,
            "semantic_lift": results["semantic_matches"] / total,
        }

    return results


def organize_reviews() -> Dict[str, int]:
    """
    Organize existing review files into subdirectories based on review needs.

    Moves files to:
    - keyword_match/: All results auto-filled, no human review needed
    - manual_review/: At least one result needs human judgment

    Returns dict with counts of files moved.
    """
    import shutil

    KEYWORD_MATCH_DIR.mkdir(exist_ok=True)
    MANUAL_REVIEW_DIR.mkdir(exist_ok=True)

    stats = {
        "moved_to_keyword_match": 0,
        "moved_to_manual_review": 0,
        "already_organized": 0,
        "errors": 0,
    }

    # Find all review files in root directory (not in subdirectories)
    root_files = list(HUMAN_REVIEW_DIR.glob("review_*.yaml"))
    # Filter out files already in subdirectories
    root_files = [f for f in root_files if f.parent == HUMAN_REVIEW_DIR]

    for filepath in root_files:
        try:
            with open(filepath, encoding='utf-8') as f:
                review = yaml.safe_load(f)

            needs_review = review.get("summary", {}).get("needs_human_review", 0)

            if needs_review > 0:
                dest = MANUAL_REVIEW_DIR / filepath.name
                shutil.move(str(filepath), str(dest))
                stats["moved_to_manual_review"] += 1
                logger.info(f"Moved to manual_review/: {filepath.name}")
            else:
                dest = KEYWORD_MATCH_DIR / filepath.name
                shutil.move(str(filepath), str(dest))
                stats["moved_to_keyword_match"] += 1
                logger.info(f"Moved to keyword_match/: {filepath.name}")

        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")
            stats["errors"] += 1

    # Count already organized files
    stats["already_organized"] = (
        len(list(KEYWORD_MATCH_DIR.glob("review_*.yaml"))) +
        len(list(MANUAL_REVIEW_DIR.glob("review_*.yaml")))
    )

    return stats


def compute_combined_metrics(minilm_auto: Dict, mika_auto: Dict,
                             minilm_human: Dict, mika_human: Dict) -> Dict:
    """
    Combine automated and human evaluation metrics for final scoring.
    """
    combined = {
        "timestamp": datetime.now().isoformat(),
        "automated_categories": ["incident_lookup", "section_queries", "aircraft_queries", "phase_queries"],
        "human_reviewed_categories": ["conceptual_queries", "comparative_queries"],
        "minilm": {
            "automated": {
                "mean_mrr": minilm_auto.get("aggregate_metrics", {}).get("mean_mrr", 0),
                "mean_hit_at_10": minilm_auto.get("aggregate_metrics", {}).get("mean_hit_at_10", 0),
            },
            "human_reviewed": {
                "semantic_precision": minilm_human.get("aggregate", {}).get("semantic_precision", 0),
                "semantic_lift": minilm_human.get("aggregate", {}).get("semantic_lift", 0),
                "false_positive_rate": minilm_human.get("aggregate", {}).get("false_positive_rate", 0),
            },
        },
        "mika": {
            "automated": {
                "mean_mrr": mika_auto.get("aggregate_metrics", {}).get("mean_mrr", 0),
                "mean_hit_at_10": mika_auto.get("aggregate_metrics", {}).get("mean_hit_at_10", 0),
            },
            "human_reviewed": {
                "semantic_precision": mika_human.get("aggregate", {}).get("semantic_precision", 0),
                "semantic_lift": mika_human.get("aggregate", {}).get("semantic_lift", 0),
                "false_positive_rate": mika_human.get("aggregate", {}).get("false_positive_rate", 0),
            },
        },
        "comparison": {},
        "recommendation": "",
    }

    # Compute differences
    auto_mrr_diff = combined["mika"]["automated"]["mean_mrr"] - combined["minilm"]["automated"]["mean_mrr"]
    semantic_lift_diff = combined["mika"]["human_reviewed"]["semantic_lift"] - combined["minilm"]["human_reviewed"]["semantic_lift"]

    combined["comparison"] = {
        "automated_mrr_difference": auto_mrr_diff,
        "semantic_lift_difference": semantic_lift_diff,
        "mika_auto_advantage": auto_mrr_diff > 0.02,
        "mika_semantic_advantage": semantic_lift_diff > 0.05,
    }

    # Generate recommendation
    if combined["comparison"]["mika_semantic_advantage"]:
        combined["recommendation"] = "MIKA - Higher semantic lift means better domain understanding"
    elif combined["comparison"]["mika_auto_advantage"]:
        combined["recommendation"] = "MIKA - Better automated metrics"
    elif auto_mrr_diff < -0.02:
        combined["recommendation"] = "MiniLM - Better automated metrics and faster"
    else:
        combined["recommendation"] = "MiniLM - Similar performance, prefer smaller/faster model"

    return combined


def export_streamlit_data(combined: Dict, minilm_auto: Dict, mika_auto: Dict) -> Dict[str, Path]:
    """
    Export all benchmark data in Streamlit-optimized Parquet format.

    Creates multiple Parquet files for different visualization needs:
    - query_comparison.parquet: Per-query metrics for both models side-by-side
    - aggregate_metrics.parquet: Summary metrics for dashboard cards
    - category_metrics.parquet: Breakdown by category for bar charts
    - human_review_summary.parquet: Human evaluation results (if available)

    Returns dict of created file paths.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    exported = {}

    # 1. Query-level comparison (most important for detailed analysis)
    if minilm_auto.get("query_results") and mika_auto.get("query_results"):
        minilm_results = {r["query_id"]: r for r in minilm_auto["query_results"]}
        mika_results = {r["query_id"]: r for r in mika_auto["query_results"]}

        comparison_rows = []
        for qid in minilm_results:
            if qid not in mika_results:
                continue
            m = minilm_results[qid]
            k = mika_results[qid]

            comparison_rows.append({
                "query_id": qid,
                "query_text": m.get("query_text", ""),
                "category": m.get("category", ""),
                "difficulty": m.get("difficulty", ""),
                "intent": m.get("intent", ""),
                # MiniLM metrics
                "minilm_mrr": m.get("mrr", 0),
                "minilm_hit_at_10": m.get("hit_at_10", False),
                "minilm_precision_at_10": m.get("precision_at_10", 0),
                "minilm_recall_at_10": m.get("recall_at_10", 0),
                "minilm_ndcg_at_10": m.get("ndcg_at_10", 0),
                "minilm_section_accuracy": m.get("section_accuracy_at_10", 0),
                "minilm_latency_ms": m.get("total_latency_ms", 0),
                # MIKA metrics
                "mika_mrr": k.get("mrr", 0),
                "mika_hit_at_10": k.get("hit_at_10", False),
                "mika_precision_at_10": k.get("precision_at_10", 0),
                "mika_recall_at_10": k.get("recall_at_10", 0),
                "mika_ndcg_at_10": k.get("ndcg_at_10", 0),
                "mika_section_accuracy": k.get("section_accuracy_at_10", 0),
                "mika_latency_ms": k.get("total_latency_ms", 0),
                # Computed comparisons
                "mrr_diff": k.get("mrr", 0) - m.get("mrr", 0),
                "winner": "MIKA" if k.get("mrr", 0) > m.get("mrr", 0) + 0.01 else (
                    "MiniLM" if m.get("mrr", 0) > k.get("mrr", 0) + 0.01 else "Tie"
                ),
            })

        if comparison_rows:
            df = pd.DataFrame(comparison_rows)
            path = RESULTS_DIR / "query_comparison.parquet"
            df.to_parquet(path, index=False)
            exported["query_comparison"] = path
            logger.info(f"Exported query comparison: {path}")

    # 2. Aggregate metrics for dashboard summary cards
    aggregate_rows = []
    for model_name, auto_data, human_key in [
        ("MiniLM", minilm_auto, "minilm"),
        ("MIKA", mika_auto, "mika"),
    ]:
        if not auto_data:
            continue

        agg = auto_data.get("aggregate_metrics", {})
        lat = auto_data.get("latency_stats", {})
        human = combined.get(human_key, {}).get("human_reviewed", {})

        aggregate_rows.append({
            "model": model_name,
            "model_id": human_key,
            # Automated metrics
            "mean_mrr": agg.get("mean_mrr", 0),
            "mean_hit_at_10": agg.get("mean_hit_at_10", 0),
            "mean_precision_at_10": agg.get("mean_precision_at_10", 0),
            "mean_recall_at_10": agg.get("mean_recall_at_10", 0),
            "mean_ndcg_at_10": agg.get("mean_ndcg_at_10", 0),
            "mean_section_accuracy": agg.get("mean_section_accuracy", 0),
            # Latency
            "mean_latency_ms": lat.get("mean_total_ms", 0),
            "p95_latency_ms": lat.get("p95_total_ms", 0),
            "mean_embed_ms": lat.get("mean_embed_ms", 0),
            "mean_search_ms": lat.get("mean_search_ms", 0),
            # Human evaluation (may be empty)
            "semantic_precision": human.get("semantic_precision", None),
            "semantic_lift": human.get("semantic_lift", None),
            "false_positive_rate": human.get("false_positive_rate", None),
        })

    if aggregate_rows:
        df = pd.DataFrame(aggregate_rows)
        path = RESULTS_DIR / "aggregate_metrics.parquet"
        df.to_parquet(path, index=False)
        exported["aggregate_metrics"] = path
        logger.info(f"Exported aggregate metrics: {path}")

    # 3. Category-level metrics for bar charts
    category_rows = []
    for model_name, auto_data in [("MiniLM", minilm_auto), ("MIKA", mika_auto)]:
        if not auto_data:
            continue
        for cat, metrics in auto_data.get("metrics_by_category", {}).items():
            category_rows.append({
                "model": model_name,
                "category": cat,
                "count": metrics.get("count", 0),
                "mrr": metrics.get("mrr", 0),
                "hit_at_10": metrics.get("hit_at_10", 0),
                "precision_at_10": metrics.get("precision_at_10", 0),
                "ndcg_at_10": metrics.get("ndcg_at_10", 0),
                "section_accuracy": metrics.get("section_accuracy", 0),
                "latency_ms": metrics.get("latency_ms", 0),
            })

    if category_rows:
        df = pd.DataFrame(category_rows)
        path = RESULTS_DIR / "category_metrics.parquet"
        df.to_parquet(path, index=False)
        exported["category_metrics"] = path
        logger.info(f"Exported category metrics: {path}")

    # 4. Difficulty-level metrics
    difficulty_rows = []
    for model_name, auto_data in [("MiniLM", minilm_auto), ("MIKA", mika_auto)]:
        if not auto_data:
            continue
        for diff, metrics in auto_data.get("metrics_by_difficulty", {}).items():
            difficulty_rows.append({
                "model": model_name,
                "difficulty": diff,
                "count": metrics.get("count", 0),
                "mrr": metrics.get("mrr", 0),
                "hit_at_10": metrics.get("hit_at_10", 0),
                "latency_ms": metrics.get("latency_ms", 0),
            })

    if difficulty_rows:
        df = pd.DataFrame(difficulty_rows)
        path = RESULTS_DIR / "difficulty_metrics.parquet"
        df.to_parquet(path, index=False)
        exported["difficulty_metrics"] = path
        logger.info(f"Exported difficulty metrics: {path}")

    # 5. Human review per-query results (if available)
    human_review_rows = []
    for model_name, human_key in [("MiniLM", "minilm"), ("MIKA", "mika")]:
        human_path = RESULTS_DIR / f"human_review_{human_key}.json"
        if human_path.exists():
            with open(human_path, encoding='utf-8') as f:
                human_data = json.load(f)
            for pq in human_data.get("per_query", []):
                human_review_rows.append({
                    "model": model_name,
                    "query_id": pq.get("query_id", ""),
                    "category": pq.get("category", ""),
                    "keyword_matches": pq.get("keyword_matches", 0),
                    "semantic_matches": pq.get("semantic_matches", 0),
                    "false_positives": pq.get("false_positives", 0),
                    "semantic_precision": pq.get("semantic_precision", 0),
                    "semantic_lift": pq.get("semantic_lift", 0),
                    "automated_mrr": pq.get("automated_mrr", 0),
                })

    if human_review_rows:
        df = pd.DataFrame(human_review_rows)
        path = RESULTS_DIR / "human_review_details.parquet"
        df.to_parquet(path, index=False)
        exported["human_review_details"] = path
        logger.info(f"Exported human review details: {path}")

    # 6. Final comparison summary (single row with decision info)
    comparison_summary = {
        "timestamp": [combined.get("timestamp", "")],
        "minilm_mrr": [combined.get("minilm", {}).get("automated", {}).get("mean_mrr", 0)],
        "mika_mrr": [combined.get("mika", {}).get("automated", {}).get("mean_mrr", 0)],
        "minilm_semantic_lift": [combined.get("minilm", {}).get("human_reviewed", {}).get("semantic_lift")],
        "mika_semantic_lift": [combined.get("mika", {}).get("human_reviewed", {}).get("semantic_lift")],
        "mrr_difference": [combined.get("comparison", {}).get("automated_mrr_difference", 0)],
        "semantic_lift_difference": [combined.get("comparison", {}).get("semantic_lift_difference", 0)],
        "mika_auto_advantage": [combined.get("comparison", {}).get("mika_auto_advantage", False)],
        "mika_semantic_advantage": [combined.get("comparison", {}).get("mika_semantic_advantage", False)],
        "recommendation": [combined.get("recommendation", "")],
    }

    df = pd.DataFrame(comparison_summary)
    path = RESULTS_DIR / "benchmark_decision.parquet"
    df.to_parquet(path, index=False)
    exported["benchmark_decision"] = path
    logger.info(f"Exported benchmark decision: {path}")

    return exported


def generate_final_report(combined: Dict) -> Path:
    """Generate final markdown report with combined automated + human metrics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# RiskRADAR Final Benchmark Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "This report combines automated benchmark metrics with human evaluation of semantic search quality.",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"**Recommendation: {combined['recommendation']}**",
        "",
        "---",
        "",
        "## Automated Metrics (30 queries)",
        "",
        "Categories: Incident Lookup, Section Queries, Aircraft Queries, Phase Queries",
        "",
        "| Model | Mean MRR | Hit@10 |",
        "|-------|----------|--------|",
        f"| MiniLM | {combined['minilm']['automated']['mean_mrr']:.3f} | {combined['minilm']['automated']['mean_hit_at_10']:.1%} |",
        f"| MIKA | {combined['mika']['automated']['mean_mrr']:.3f} | {combined['mika']['automated']['mean_hit_at_10']:.1%} |",
        "",
        f"**Difference (MIKA - MiniLM):** {combined['comparison']['automated_mrr_difference']:+.3f} MRR",
        "",
        "---",
        "",
        "## Human-Evaluated Metrics (20 queries)",
        "",
        "Categories: Conceptual Queries, Comparative Queries",
        "",
        "| Model | Semantic Precision | Semantic Lift | False Positive Rate |",
        "|-------|-------------------|---------------|---------------------|",
        f"| MiniLM | {combined['minilm']['human_reviewed']['semantic_precision']:.1%} | {combined['minilm']['human_reviewed']['semantic_lift']:.1%} | {combined['minilm']['human_reviewed']['false_positive_rate']:.1%} |",
        f"| MIKA | {combined['mika']['human_reviewed']['semantic_precision']:.1%} | {combined['mika']['human_reviewed']['semantic_lift']:.1%} | {combined['mika']['human_reviewed']['false_positive_rate']:.1%} |",
        "",
        "**Key Metrics Explained:**",
        "- **Semantic Precision:** Fraction of results that are relevant (keyword + semantic matches)",
        "- **Semantic Lift:** Fraction of results that are relevant but wouldn't match keywords (the VALUE of semantic search)",
        "- **False Positive Rate:** Fraction of results that are not relevant",
        "",
        f"**Semantic Lift Difference (MIKA - MiniLM):** {combined['comparison']['semantic_lift_difference']:+.1%}",
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ]

    # Add interpretation based on results
    if combined["comparison"]["mika_semantic_advantage"]:
        lines.extend([
            "MIKA shows **significantly higher semantic lift**, meaning it finds more relevant content",
            "that keyword search would miss. This is the primary value of a domain-specific embedding model.",
            "",
            "Even if automated metrics are similar, MIKA's ability to surface semantically relevant",
            "content makes it the better choice for aviation safety research.",
        ])
    elif combined["comparison"]["mika_auto_advantage"]:
        lines.extend([
            "MIKA shows better automated metrics but similar semantic lift to MiniLM.",
            "The domain-specific training helps with keyword-matchable content but doesn't",
            "provide significant advantage for pure semantic understanding.",
        ])
    else:
        lines.extend([
            "Both models show similar performance across automated and human-evaluated metrics.",
            "MiniLM is recommended as it is smaller (384 vs 768 dimensions) and faster,",
            "with no significant quality tradeoff.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "*Report generated by `eval/benchmark.py final-report`*",
    ])

    report_path = EVAL_DIR / "final_benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save combined metrics as JSON
    json_path = RESULTS_DIR / "combined_metrics.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    return report_path


# =============================================================================
# CLI COMMANDS
# =============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def cmd_run(args):
    """Run benchmark command."""
    setup_logging(args.verbose)

    models = [args.model] if args.model != "both" else ["minilm", "mika"]
    runs = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {model_name.upper()}")
        print(f"{'='*60}\n")

        run = run_benchmark(model_name)
        json_path, parquet_path = save_results(run)
        runs[model_name] = run

        # Print summary
        print(f"\n{model_name.upper()} Results:")
        print(f"  Queries evaluated: {run.total_queries}")
        print(f"  Mean MRR: {run.mean_mrr:.3f}")
        print(f"  Hit@10: {run.mean_hit_at_10:.1%}")
        print(f"  nDCG@10: {run.mean_ndcg_at_10:.3f}")
        print(f"  Section Accuracy: {run.mean_section_accuracy:.1%}")
        print(f"  Mean latency: {run.mean_total_latency_ms:.0f}ms (p95: {run.p95_total_latency_ms:.0f}ms)")
        print(f"  Duration: {run.duration_sec:.1f}s")
        print(f"  Results: {json_path}")

    # Generate report if both models run
    if len(runs) == 2:
        print(f"\n{'='*60}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*60}\n")

        report_path = generate_report()
        print(f"Report: {report_path}")

        # Statistical comparison (using bootstrap CI and win/loss/tie per documentation)
        try:
            stats = compute_statistical_comparison(runs["minilm"], runs["mika"])
            ci = stats['bootstrap_95_ci']
            wlt = stats['win_loss_tie']

            print(f"\nStatistical Comparison:")
            print(f"  MRR difference (MIKA - MiniLM): {stats['difference']:+.3f}")
            print(f"  Bootstrap 95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")
            print(f"  Win/Loss/Tie: MIKA {wlt['mika_wins']} / MiniLM {wlt['minilm_wins']} / Tie {wlt['ties']}")

            # Interpretation
            if ci['significant']:
                if ci['lower'] > 0:
                    print(f"  Interpretation: MIKA significantly better (CI excludes 0)")
                else:
                    print(f"  Interpretation: MiniLM significantly better (CI excludes 0)")
            else:
                print(f"  Interpretation: No significant difference (CI includes 0)")

        except ImportError:
            print("\n  (Install scipy for statistical tests: pip install scipy)")

    return 0


def cmd_report(args):
    """Generate report from existing results."""
    setup_logging(args.verbose)

    try:
        report_path = generate_report()
        print(f"Report generated: {report_path}")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1


def cmd_validate(args):
    """Validate gold queries ground truth."""
    setup_logging(args.verbose)

    print("Validating gold queries ground truth...\n")

    gold_queries = load_gold_queries()

    # Count queries
    total = 0
    by_category = {}
    by_difficulty = {"easy": 0, "medium": 0, "hard": 0}

    categories = [
        "incident_lookup", "conceptual_queries", "section_queries",
        "comparative_queries", "aircraft_queries", "phase_queries"
    ]

    for cat in categories:
        queries = gold_queries.get(cat, [])
        by_category[cat] = len(queries)
        total += len(queries)
        for q in queries:
            diff = q.get("difficulty", "medium")
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

    print(f"Total queries: {total}")
    print(f"\nBy category:")
    for cat, count in by_category.items():
        print(f"  {cat}: {count}")

    print(f"\nBy difficulty:")
    for diff, count in by_difficulty.items():
        print(f"  {diff}: {count}")

    # Validate structure
    print("\nValidating query structure...")
    issues = []

    for cat in categories:
        for q in gold_queries.get(cat, []):
            if "id" not in q:
                issues.append(f"Missing id in {cat}")
            if "query" not in q:
                issues.append(f"Missing query in {cat}: {q.get('id', 'unknown')}")
            if "ground_truth" not in q and cat != "section_queries":
                issues.append(f"Missing ground_truth in {cat}: {q.get('id', 'unknown')}")

    if issues:
        print(f"\nIssues found ({len(issues)}):")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("\nAll queries valid!")

    return 0


def cmd_export_review(args):
    """Export benchmark results for human review."""
    setup_logging(args.verbose)

    models = [args.model] if args.model != "both" else ["minilm", "mika"]

    print("Exporting benchmark results for human review...")
    print(f"Output directory: {HUMAN_REVIEW_DIR}\n")

    total_exported = 0
    total_auto_filled = 0
    total_needs_review = 0

    for model_name in models:
        try:
            exported = export_for_human_review(model_name)
            total_exported += len(exported)

            # Count auto-filled vs needs review
            model_auto = 0
            model_needs = 0
            for f in exported:
                with open(f, encoding='utf-8') as rf:
                    review = yaml.safe_load(rf)
                    summary = review.get("summary", {})
                    model_auto += summary.get("auto_keyword_matches", 0)
                    model_needs += summary.get("needs_human_review", 0)

            total_auto_filled += model_auto
            total_needs_review += model_needs

            print(f"{model_name.upper()}: {len(exported)} queries exported")
            print(f"  - Auto-filled KEYWORD_MATCH: {model_auto} results")
            print(f"  - Needs human review: {model_needs} results")
            print()

        except FileNotFoundError as e:
            print(f"{model_name.upper()}: {e}\n")

    print(f"{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Total queries exported: {total_exported}")
    print(f"Auto-filled (KEYWORD_MATCH): {total_auto_filled} results")
    print(f"Needs human judgment: {total_needs_review} results")
    print(f"Time estimate: ~{total_needs_review * 0.5:.0f} minutes ({total_needs_review} judgments @ 30 sec each)")

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Open each YAML file in eval/human_reviews/

2. KEYWORD_MATCH judgments are already filled in automatically
   - Review these to confirm they're correct (can override if wrong)

3. For results with EMPTY judgment, assign:
   - SEMANTIC_MATCH: Relevant but doesn't match keywords (THE KEY METRIC!)
   - FALSE_POSITIVE: Not relevant to the query

4. Set 'review_complete: true' in metadata when done

5. Run: python -m eval.benchmark import-review
6. Run: python -m eval.benchmark final-report
""")

    return 0


def cmd_import_review(args):
    """Import human review files and compute metrics."""
    setup_logging(args.verbose)

    models = [args.model] if args.model != "both" else ["minilm", "mika"]

    print("Importing human reviews...\n")

    results = {}
    for model_name in models:
        try:
            human_results = import_human_reviews(model_name)
            results[model_name] = human_results

            print(f"{model_name.upper()} Human Review Results:")
            print(f"  Queries reviewed: {human_results['queries_reviewed']}")
            print(f"  Queries complete: {human_results['queries_complete']}")
            print(f"  Total judgments: {human_results['total_results_reviewed']}")

            if "aggregate" in human_results:
                agg = human_results["aggregate"]
                print(f"\n  Aggregate Metrics:")
                print(f"    Keyword Match Rate: {agg['keyword_match_rate']:.1%}")
                print(f"    Semantic Match Rate: {agg['semantic_match_rate']:.1%}")
                print(f"    False Positive Rate: {agg['false_positive_rate']:.1%}")
                print(f"    Semantic Precision: {agg['semantic_precision']:.1%}")
                print(f"    Semantic Lift: {agg['semantic_lift']:.1%}")
            print()

        except FileNotFoundError as e:
            print(f"{model_name.upper()}: {e}\n")

    # Save human review results
    if results:
        RESULTS_DIR.mkdir(exist_ok=True)
        for model_name, data in results.items():
            json_path = RESULTS_DIR / f"human_review_{model_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved: {json_path}")

    return 0


def cmd_organize_reviews(args):
    """Organize review files into subdirectories based on review needs."""
    setup_logging(args.verbose)

    print("Organizing review files...")
    print(f"  keyword_match/ : Files where all results auto-filled (no review needed)")
    print(f"  manual_review/ : Files with results needing human judgment")
    print()

    stats = organize_reviews()

    print(f"Results:")
    print(f"  Moved to keyword_match/: {stats['moved_to_keyword_match']}")
    print(f"  Moved to manual_review/: {stats['moved_to_manual_review']}")
    print(f"  Already in subdirectories: {stats['already_organized']}")
    if stats['errors'] > 0:
        print(f"  Errors: {stats['errors']}")

    print(f"\nYou can now focus on files in: eval/human_reviews/manual_review/")

    return 0


def cmd_final_report(args):
    """Generate final report combining automated and human metrics."""
    setup_logging(args.verbose)

    print("Generating final benchmark report...\n")

    # Load automated results
    minilm_auto = {}
    mika_auto = {}

    minilm_auto_files = sorted(RESULTS_DIR.glob("benchmark_minilm_*.json"), reverse=True)
    mika_auto_files = sorted(RESULTS_DIR.glob("benchmark_mika_*.json"), reverse=True)

    if minilm_auto_files:
        with open(minilm_auto_files[0], encoding='utf-8') as f:
            minilm_auto = json.load(f)
        print(f"Loaded automated results: {minilm_auto_files[0].name}")

    if mika_auto_files:
        with open(mika_auto_files[0], encoding='utf-8') as f:
            mika_auto = json.load(f)
        print(f"Loaded automated results: {mika_auto_files[0].name}")

    # Load human review results
    minilm_human = {}
    mika_human = {}

    minilm_human_path = RESULTS_DIR / "human_review_minilm.json"
    mika_human_path = RESULTS_DIR / "human_review_mika.json"

    if minilm_human_path.exists():
        with open(minilm_human_path, encoding='utf-8') as f:
            minilm_human = json.load(f)
        print(f"Loaded human review: {minilm_human_path.name}")
    else:
        print("WARNING: No human review found for MiniLM")

    if mika_human_path.exists():
        with open(mika_human_path, encoding='utf-8') as f:
            mika_human = json.load(f)
        print(f"Loaded human review: {mika_human_path.name}")
    else:
        print("WARNING: No human review found for MIKA")

    # Check if we have enough data
    if not minilm_auto and not mika_auto:
        print("\nERROR: No automated benchmark results found. Run 'benchmark run' first.")
        return 1

    if not minilm_human.get("aggregate") and not mika_human.get("aggregate"):
        print("\nWARNING: No completed human reviews found.")
        print("Human-adjusted metrics will be empty.")
        print("Complete human reviews and run 'benchmark import-review' first.")

    # Compute combined metrics
    combined = compute_combined_metrics(minilm_auto, mika_auto, minilm_human, mika_human)

    # Generate report
    report_path = generate_final_report(combined)

    # Export Streamlit-ready Parquet files
    print("\nExporting Streamlit-ready Parquet files...")
    streamlit_files = export_streamlit_data(combined, minilm_auto, mika_auto)

    print(f"\n{'='*60}")
    print("FINAL REPORT GENERATED")
    print("="*60)
    print(f"\nReport: {report_path}")
    print(f"Metrics: {RESULTS_DIR / 'combined_metrics.json'}")
    print(f"\nStreamlit Parquet Files ({len(streamlit_files)} created):")
    for name, path in streamlit_files.items():
        print(f"  - {path.name}")
    print(f"\nRecommendation: {combined['recommendation']}")

    return 0


def cmd_status(args):
    """Show current benchmark status and next steps."""
    setup_logging(args.verbose)

    print("="*60)
    print("RISKRADAR BENCHMARK STATUS")
    print("="*60)

    # Check automated results
    print("\n1. AUTOMATED BENCHMARK RESULTS")
    print("-" * 40)

    for model in ["minilm", "mika"]:
        files = sorted(RESULTS_DIR.glob(f"benchmark_{model}_*.json"), reverse=True)
        if files:
            with open(files[0], encoding='utf-8') as f:
                data = json.load(f)
            print(f"  {model.upper()}: {files[0].name}")
            print(f"    MRR: {data['aggregate_metrics']['mean_mrr']:.3f}")
            print(f"    Hit@10: {data['aggregate_metrics']['mean_hit_at_10']:.1%}")
        else:
            print(f"  {model.upper()}: NOT RUN")

    # Check human reviews
    print("\n2. HUMAN REVIEW STATUS")
    print("-" * 40)

    for model in ["minilm", "mika"]:
        review_files = find_review_files(model_name=model)
        if review_files:
            complete = 0
            in_manual = 0
            in_keyword = 0
            for f in review_files:
                with open(f, encoding='utf-8') as rf:
                    review = yaml.safe_load(rf)
                    if review.get("metadata", {}).get("review_complete"):
                        complete += 1
                # Count by directory
                if MANUAL_REVIEW_DIR in f.parents or f.parent == MANUAL_REVIEW_DIR:
                    in_manual += 1
                elif KEYWORD_MATCH_DIR in f.parents or f.parent == KEYWORD_MATCH_DIR:
                    in_keyword += 1
            print(f"  {model.upper()}: {complete}/{len(review_files)} reviews complete")
            print(f"    - In manual_review/: {in_manual} (need attention)")
            print(f"    - In keyword_match/: {in_keyword} (auto-filled)")
        else:
            print(f"  {model.upper()}: No review files (run 'export-review' first)")

    # Check imported human reviews
    print("\n3. IMPORTED HUMAN METRICS")
    print("-" * 40)

    for model in ["minilm", "mika"]:
        human_path = RESULTS_DIR / f"human_review_{model}.json"
        if human_path.exists():
            with open(human_path, encoding='utf-8') as f:
                data = json.load(f)
            if "aggregate" in data:
                print(f"  {model.upper()}: Semantic Lift = {data['aggregate']['semantic_lift']:.1%}")
            else:
                print(f"  {model.upper()}: Imported but no complete reviews")
        else:
            print(f"  {model.upper()}: NOT IMPORTED")

    # Check final report
    print("\n4. FINAL REPORT")
    print("-" * 40)

    final_report = EVAL_DIR / "final_benchmark_report.md"
    combined_metrics = RESULTS_DIR / "combined_metrics.json"

    if final_report.exists():
        print(f"  Report: {final_report}")
        if combined_metrics.exists():
            with open(combined_metrics, encoding='utf-8') as f:
                data = json.load(f)
            print(f"  Recommendation: {data.get('recommendation', 'N/A')}")
    else:
        print("  NOT GENERATED")

    # Print next steps
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    steps = []
    minilm_results = list(RESULTS_DIR.glob("benchmark_minilm_*.json"))
    mika_results = list(RESULTS_DIR.glob("benchmark_mika_*.json"))

    if not minilm_results or not mika_results:
        steps.append("1. Run automated benchmark: python -m eval.benchmark run")

    review_files = find_review_files()
    if not review_files:
        steps.append("2. Export for human review: python -m eval.benchmark export-review")
    else:
        # Check if any are incomplete in manual_review directory
        manual_files = [f for f in review_files if MANUAL_REVIEW_DIR in f.parents or f.parent == MANUAL_REVIEW_DIR]
        incomplete = sum(1 for f in manual_files
                        if not yaml.safe_load(open(f, encoding='utf-8')).get("metadata", {}).get("review_complete"))
        if incomplete > 0:
            steps.append(f"2. Complete {incomplete} human reviews in eval/human_reviews/manual_review/")

    minilm_human = RESULTS_DIR / "human_review_minilm.json"
    mika_human = RESULTS_DIR / "human_review_mika.json"

    if not minilm_human.exists() or not mika_human.exists():
        steps.append("3. Import human reviews: python -m eval.benchmark import-review")

    if not final_report.exists():
        steps.append("4. Generate final report: python -m eval.benchmark final-report")

    if steps:
        for step in steps:
            print(f"  {step}")
    else:
        print("  All steps complete! Review final_benchmark_report.md")

    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RiskRADAR Embedding Benchmark v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete Workflow:
  1. python -m eval.benchmark run              # Run automated benchmark
  2. python -m eval.benchmark export-review    # Export for human review
  3. [Complete human reviews in eval/human_reviews/]
  4. python -m eval.benchmark import-review    # Import human judgments
  5. python -m eval.benchmark final-report     # Generate combined report

Individual Commands:
  python -m eval.benchmark status             # Check current progress
  python -m eval.benchmark run -m minilm      # Benchmark single model
  python -m eval.benchmark validate           # Validate query definitions
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show benchmark status and next steps")
    status_parser.add_argument("-v", "--verbose", action="store_true")
    status_parser.set_defaults(func=cmd_status)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run automated benchmark")
    run_parser.add_argument("-m", "--model", choices=["minilm", "mika", "both"], default="both")
    run_parser.add_argument("-v", "--verbose", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    # Export for human review
    export_parser = subparsers.add_parser("export-review", help="Export results for human review")
    export_parser.add_argument("-m", "--model", choices=["minilm", "mika", "both"], default="both")
    export_parser.add_argument("-v", "--verbose", action="store_true")
    export_parser.set_defaults(func=cmd_export_review)

    # Import human reviews
    import_parser = subparsers.add_parser("import-review", help="Import completed human reviews")
    import_parser.add_argument("-m", "--model", choices=["minilm", "mika", "both"], default="both")
    import_parser.add_argument("-v", "--verbose", action="store_true")
    import_parser.set_defaults(func=cmd_import_review)

    # Organize reviews into subdirectories
    organize_parser = subparsers.add_parser("organize-reviews", help="Organize review files into keyword_match/ and manual_review/ subdirectories")
    organize_parser.add_argument("-v", "--verbose", action="store_true")
    organize_parser.set_defaults(func=cmd_organize_reviews)

    # Final report with combined metrics
    final_parser = subparsers.add_parser("final-report", help="Generate final report with human metrics")
    final_parser.add_argument("-v", "--verbose", action="store_true")
    final_parser.set_defaults(func=cmd_final_report)

    # Report command (automated only)
    report_parser = subparsers.add_parser("report", help="Generate automated-only report")
    report_parser.add_argument("-v", "--verbose", action="store_true")
    report_parser.set_defaults(func=cmd_report)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate query definitions")
    validate_parser.add_argument("-v", "--verbose", action="store_true")
    validate_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
