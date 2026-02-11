"""
search/cli.py
-------------
CLI entry point for hybrid search.

Usage:
    python -m search.cli build-index          # Build BM25 index from chunks.jsonl
    python -m search.cli search <query>       # Hybrid search (default)
    python -m search.cli search <query> --mode bm25
    python -m search.cli search <query> --mode semantic
    python -m search.cli search <query> --filter-l1 LOC-I
    python -m search.cli benchmark            # Compare all 3 modes on gold queries
    python -m search.cli stats                # Index stats
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import SEARCH_CONFIG

PROJECT_ROOT = Path(__file__).parent.parent


def setup_logging(command: str, verbose: bool = False) -> logging.Logger:
    """Configure logging to console and file."""
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = logs_dir / f"search_{command}_{timestamp}.log"

    log_level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    )
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def cmd_build_index(args, logger):
    """Build BM25 index from chunks.jsonl."""
    from .bm25 import BM25Index

    index = BM25Index()
    index.build(chunks_jsonl_path=SEARCH_CONFIG.chunks_jsonl_path)
    index.save(SEARCH_CONFIG.bm25_index_path)

    print("\n" + "=" * 60)
    print("BM25 INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Corpus size: {index.corpus_size:,} chunks")
    print(f"Index file: {SEARCH_CONFIG.bm25_index_path}")
    size_mb = SEARCH_CONFIG.bm25_index_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    print("\nNext step: Run a search")
    print('  python -m search.cli search "engine failure during takeoff"')
    print("=" * 60)
    return 0


def cmd_search(args, logger):
    """Run a search query."""
    from .hybrid import HybridSearcher

    query = args.query
    mode = args.mode
    limit = args.limit

    filters = None
    if args.filter_l1:
        filters = {"l1_categories": args.filter_l1}

    searcher = HybridSearcher()

    start = time.perf_counter()
    results = searcher.search(query, limit=limit, mode=mode, filters=filters)
    elapsed = (time.perf_counter() - start) * 1000

    print("\n" + "=" * 70)
    print(f"SEARCH RESULTS  [{mode.upper()}]  ({elapsed:.0f}ms)")
    print(f"Query: {query}")
    if filters:
        print(f"Filters: {filters}")
    print("=" * 70)

    if not results:
        print("\nNo results found.")
        return 0

    # Table header
    print(f"\n{'Rank':>4}  {'Score':>7}  {'Source':>8}  {'Report ID':<16}  {'Section':<25}  {'Chunk ID'}")
    print("-" * 100)

    for r in results:
        print(
            f"{r['rank']:>4}  "
            f"{r['score']:>7.4f}  "
            f"{r['source']:>8}  "
            f"{r['report_id']:<16}  "
            f"{r['section_name'][:25]:<25}  "
            f"{r['chunk_id']}"
        )

    print("-" * 100)
    print(f"Showing {len(results)} of {limit} requested results")

    # Show rank diagnostics for hybrid mode
    if mode == "hybrid":
        both_count = sum(1 for r in results if r["source"] == "both")
        bm25_only = sum(1 for r in results if r["source"] == "bm25")
        sem_only = sum(1 for r in results if r["source"] == "semantic")
        print(f"\nSource breakdown: both={both_count}, bm25_only={bm25_only}, semantic_only={sem_only}")

    print("=" * 70)
    return 0


def cmd_benchmark(args, logger):
    """Compare BM25, semantic, and hybrid search on gold queries."""
    import yaml
    from .hybrid import HybridSearcher

    # Load gold queries
    gold_path = PROJECT_ROOT / "eval" / "gold_queries.yaml"
    if not gold_path.exists():
        print(f"Error: Gold queries not found at {gold_path}")
        return 1

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = yaml.safe_load(f)

    # Collect all queries across categories
    queries = []
    category_keys = [k for k in gold_data.keys() if k != "metadata"]
    for cat_key in category_keys:
        for q in gold_data[cat_key]:
            queries.append({
                "id": q["id"],
                "query": q["query"],
                "category": cat_key,
                "difficulty": q.get("difficulty", "medium"),
                "expected_report_ids": set(q["ground_truth"]["expected_report_ids"]),
            })

    print(f"\nLoaded {len(queries)} gold queries across {len(category_keys)} categories")

    searcher = HybridSearcher()
    modes = ["bm25", "semantic", "hybrid"]
    mode_results = {m: {"mrr_list": [], "hit10_list": []} for m in modes}

    for mode in modes:
        print(f"\nRunning {mode.upper()} search...")
        for i, q in enumerate(queries):
            try:
                results = searcher.search(q["query"], limit=10, mode=mode)
                retrieved_reports = [r["report_id"] for r in results]
                relevant = q["expected_report_ids"]

                # MRR
                mrr = 0.0
                for rank_idx, report_id in enumerate(retrieved_reports):
                    if report_id in relevant:
                        mrr = 1.0 / (rank_idx + 1)
                        break
                mode_results[mode]["mrr_list"].append(mrr)

                # Hit@10
                hit10 = 1.0 if any(r in relevant for r in retrieved_reports) else 0.0
                mode_results[mode]["hit10_list"].append(hit10)

            except Exception as e:
                logger.warning(f"Error on query {q['id']} ({mode}): {e}")
                mode_results[mode]["mrr_list"].append(0.0)
                mode_results[mode]["hit10_list"].append(0.0)

            if (i + 1) % 10 == 0:
                logger.info(f"  {mode}: {i + 1}/{len(queries)} queries done")

    # Print comparison table
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"\n{'Mode':<12}  {'MRR':>8}  {'Hit@10':>8}  {'Queries':>8}")
    print("-" * 45)

    for mode in modes:
        mrr_vals = mode_results[mode]["mrr_list"]
        hit_vals = mode_results[mode]["hit10_list"]
        n = len(mrr_vals)
        avg_mrr = sum(mrr_vals) / n if n > 0 else 0
        avg_hit = sum(hit_vals) / n if n > 0 else 0
        print(f"{mode:<12}  {avg_mrr:>8.3f}  {avg_hit * 100:>7.1f}%  {n:>8}")

    print("-" * 45)

    # Per-difficulty breakdown
    difficulties = sorted(set(q["difficulty"] for q in queries))
    if len(difficulties) > 1:
        print("\nPer-difficulty MRR:")
        print(f"{'Difficulty':<12}", end="")
        for mode in modes:
            print(f"  {mode:>10}", end="")
        print()
        print("-" * 45)

        for diff in difficulties:
            diff_indices = [i for i, q in enumerate(queries) if q["difficulty"] == diff]
            print(f"{diff:<12}", end="")
            for mode in modes:
                vals = [mode_results[mode]["mrr_list"][i] for i in diff_indices]
                avg = sum(vals) / len(vals) if vals else 0
                print(f"  {avg:>10.3f}", end="")
            print()

    print("\n" + "=" * 60)
    return 0


def cmd_stats(args, logger):
    """Show BM25 index stats."""
    index_path = SEARCH_CONFIG.bm25_index_path

    print("\n" + "=" * 60)
    print("SEARCH INDEX STATS")
    print("=" * 60)

    if not index_path.exists():
        print(f"\nBM25 index not found at: {index_path}")
        print("Run: python -m search.cli build-index")
        return 1

    from .bm25 import BM25Index

    index = BM25Index.load(index_path)

    size_mb = index_path.stat().st_size / (1024 * 1024)

    print(f"\nBM25 Index:")
    print(f"  Path: {index_path}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Corpus size: {index.corpus_size:,} chunks")
    print(f"  Unique reports: {len(set(index.report_ids)):,}")
    print(f"  Unique sections: {len(set(index.section_names)):,}")

    # Vocab estimate from BM25 idf
    if index.index is not None and hasattr(index.index, "idf"):
        vocab_size = len(index.index.idf)
        print(f"  Vocabulary size: {vocab_size:,} terms")

    # Chunks JSONL source
    chunks_path = SEARCH_CONFIG.chunks_jsonl_path
    if chunks_path.exists():
        chunks_mb = chunks_path.stat().st_size / (1024 * 1024)
        print(f"\nSource JSONL:")
        print(f"  Path: {chunks_path}")
        print(f"  File size: {chunks_mb:.1f} MB")

    print(f"\nSearch Config:")
    print(f"  Default model: {SEARCH_CONFIG.default_model}")
    print(f"  Default collection: {SEARCH_CONFIG.default_collection}")
    print(f"  RRF k: {SEARCH_CONFIG.rrf_k}")
    print(f"  Semantic weight: {SEARCH_CONFIG.semantic_weight}")
    print(f"  BM25 weight: {SEARCH_CONFIG.bm25_weight}")

    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RiskRADAR Hybrid Search CLI - BM25 + Semantic + RRF Fusion"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-index
    subparsers.add_parser("build-index", help="Build BM25 index from chunks.jsonl")

    # search
    search_parser = subparsers.add_parser("search", help="Search for a query")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument(
        "--mode",
        choices=["bm25", "semantic", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    search_parser.add_argument(
        "--limit", type=int, default=20, help="Number of results (default: 20)"
    )
    search_parser.add_argument(
        "--filter-l1",
        help="Filter by L1 category code (e.g., LOC-I). Semantic/hybrid only.",
    )

    # benchmark
    subparsers.add_parser(
        "benchmark", help="Compare BM25/semantic/hybrid on gold queries"
    )

    # stats
    subparsers.add_parser("stats", help="Show BM25 index stats")

    args = parser.parse_args()

    logger = setup_logging(args.command, args.verbose)

    commands = {
        "build-index": cmd_build_index,
        "search": cmd_search,
        "benchmark": cmd_benchmark,
        "stats": cmd_stats,
    }

    try:
        exit_code = commands[args.command](args, logger)
        sys.exit(exit_code)
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
