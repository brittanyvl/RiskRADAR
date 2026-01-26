"""
taxonomy/cli.py
---------------
CLI entry point for CICTT-anchored causal taxonomy.

Usage:
    # L1 (CICTT) Classification
    python -m taxonomy.cli map              # Map reports to CICTT categories
    python -m taxonomy.cli review           # Export L1 for review (HTML)
    python -m taxonomy.cli review --csv     # Export L1 as CSV
    python -m taxonomy.cli stats            # Show taxonomy statistics
    python -m taxonomy.cli categories       # List CICTT categories

    # L2 Subcategory Classification (builds on L1)
    python -m taxonomy.cli classify-l2      # Run L2 on existing L1 results
    python -m taxonomy.cli classify-l2 --l1-run 1  # Specify L1 run to extend
    python -m taxonomy.cli review-l2        # Export L2 for review (HTML)
    python -m taxonomy.cli review-l2 --csv  # Export L2 as CSV
    python -m taxonomy.cli subcategories    # List L2 subcategories

    # Legacy commands
    python -m taxonomy.cli export-review    # Export hierarchical review (CSV)
    python -m taxonomy.cli import-review FILE  # Import reviewed results
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from riskradar.config import DB_PATH


def setup_logging(command: str, verbose: bool = False) -> logging.Logger:
    """Configure logging to console and file."""
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = logs_dir / f"taxonomy_{command}_{timestamp}.log"

    log_level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    ))
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def cmd_map(args, logger):
    """Map reports to CICTT categories."""
    from .mapper import map_reports_to_cictt

    logger.info("Starting CICTT Taxonomy Mapping")

    results = map_reports_to_cictt(run_id=args.run_id)

    print("\n" + "=" * 60)
    print("CICTT MAPPING COMPLETE")
    print("=" * 60)

    stats = results["stats"]
    print(f"Reports mapped: {stats['n_reports']}")
    print(f"Chunks analyzed: {stats['n_chunks_processed']:,}")
    print(f"Categories used: {stats['categories_used']}")
    print(f"Total assignments: {stats['n_report_assignments']}")

    print("\nCategory Distribution (top 10):")
    cat_dist = results["category_distribution"].head(10)
    for code, row in cat_dist.iterrows():
        from .cictt import CICTT_BY_CODE
        name = CICTT_BY_CODE[code].name if code in CICTT_BY_CODE else code
        print(f"  {code:8} {name:40} {row['n_reports']:>5} reports")

    print("\nOutput files:")
    for name, path in results["paths"].items():
        print(f"  {name}: {path}")

    print("\nNext step: Export for review")
    print("  python -m taxonomy.cli review")
    print("=" * 60)

    return 0


def cmd_review(args, logger):
    """Export mappings for human review."""
    if args.csv:
        from .review import export_review_csv
        output_path = export_review_csv(run_id=args.run_id)
        format_name = "CSV"
    else:
        from .review import export_review_html
        output_path = export_review_html(
            run_id=args.run_id,
            max_reports_per_cat=args.max_reports
        )
        format_name = "HTML"

    print("\n" + "=" * 60)
    print(f"REVIEW EXPORT COMPLETE ({format_name})")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print("\nInstructions:")
    print("1. Open the file in your browser (HTML) or spreadsheet (CSV)")
    print("2. Review category assignments for each report")
    print("3. Click category headers to expand/collapse (HTML)")
    print("4. Note any incorrect assignments or missing categories")
    print("=" * 60)

    return 0


def cmd_stats(args, logger):
    """Show taxonomy statistics."""
    from .config import TAXONOMY_DATA_DIR
    from .cictt import get_primary_categories
    import json

    print("\n" + "=" * 60)
    print("CICTT TAXONOMY STATISTICS")
    print("=" * 60)

    # Check for mapping runs
    for run_id in range(1, 10):
        stats_path = TAXONOMY_DATA_DIR / f"mapping_stats_run{run_id}.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

            print(f"\nRun {run_id}:")
            print(f"  Reports mapped: {stats['n_reports']}")
            print(f"  Chunks analyzed: {stats['n_chunks_processed']:,}")
            print(f"  Categories used: {stats['categories_used']}")

            # Load category distribution
            import pandas as pd
            report_path = TAXONOMY_DATA_DIR / f"report_categories_run{run_id}.parquet"
            if report_path.exists():
                report_cats = pd.read_parquet(report_path)
                print("\n  Top categories:")
                top_cats = report_cats["category_code"].value_counts().head(5)
                from .cictt import CICTT_BY_CODE
                for code, count in top_cats.items():
                    name = CICTT_BY_CODE[code].name if code in CICTT_BY_CODE else code
                    print(f"    {code}: {count} ({name})")

    # Show CICTT summary
    print("\n" + "-" * 40)
    print("CICTT Categories Available:")
    categories = get_primary_categories()
    print(f"  {len(categories)} primary occurrence categories")

    print("=" * 60)
    return 0


def cmd_categories(args, logger):
    """List all CICTT categories."""
    from .cictt import get_primary_categories

    print("\n" + "=" * 60)
    print("CICTT OCCURRENCE CATEGORIES")
    print("=" * 60)

    categories = get_primary_categories()

    for cat in categories:
        print(f"\n{cat.code} - {cat.name}")
        print("-" * 50)
        print(f"  {cat.description[:200]}...")
        print(f"  Keywords: {', '.join(cat.keywords[:5])}...")

    print("\n" + "=" * 60)
    print(f"Total: {len(categories)} categories")
    print("Reference: CICTT v4.7 (December 2017)")
    print("=" * 60)
    return 0


# =============================================================================
# Hierarchical Classification Commands (L1 + L2)
# =============================================================================

def cmd_classify_l2(args, logger):
    """Run L2 subcategory classification on existing L1 results."""
    from .pipeline import run_l2_classification

    logger.info("Starting L2 Subcategory Classification")
    logger.info(f"Building on L1 run: {args.l1_run}")

    results = run_l2_classification(
        l1_run_id=args.l1_run,
        l2_run_id=args.run_id,
    )

    stats = results["stats"]

    print("\n" + "=" * 60)
    print("L2 SUBCATEGORY CLASSIFICATION COMPLETE")
    print("=" * 60)
    print(f"L1 source run: {stats['l1_run_id']}")
    print(f"L2 run ID: {stats['l2_run_id']}")
    print(f"\nL1 Source:")
    print(f"  Reports: {stats['l1_source']['unique_reports']}")
    print(f"  Chunk assignments: {stats['l1_source']['chunk_assignments']:,}")
    print(f"\nL2 Results:")
    print(f"  Eligible chunks: {stats['l2']['eligible_chunks']:,}")
    print(f"  Chunk assignments: {stats['l2']['chunk_assignments']:,}")
    print(f"  Report assignments: {stats['l2']['report_assignments']:,}")
    print(f"  Subcategories used: {stats['l2']['subcategories_used']}")
    print(f"  Reports with L2: {stats['l2']['reports_with_l2']}")
    print(f"\nPerformance:")
    print(f"  Total time: {stats['performance']['total_time_sec']:.1f}s")

    print("\nNext step: Export for human review")
    print("  python -m taxonomy.cli review-l2")
    print("=" * 60)

    return 0


def cmd_review_l2(args, logger):
    """Export L2 subcategory results for human review (HTML or CSV)."""
    if args.csv:
        from .review import export_l2_review_csv
        output_path = export_l2_review_csv(
            l1_run_id=args.l1_run,
            l2_run_id=args.l2_run,
        )
        format_name = "CSV"
    else:
        from .review import export_l2_review_html
        output_path = export_l2_review_html(
            l1_run_id=args.l1_run,
            l2_run_id=args.l2_run,
            max_reports_per_subcat=args.max_reports,
        )
        format_name = "HTML"

    print("\n" + "=" * 60)
    print(f"L2 REVIEW EXPORT COMPLETE ({format_name})")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print("\nInstructions:")
    print("1. Open the file in your browser (HTML) or spreadsheet (CSV)")
    print("2. Review subcategory assignments under each L1 category")
    print("3. Click L1 category headers to expand/collapse (HTML)")
    print("4. Click subcategory headers to see reports (HTML)")
    print("5. Note any incorrect assignments or missing subcategories")
    print("=" * 60)

    return 0


def cmd_export_review(args, logger):
    """Export hierarchical results for human review."""
    from .review import export_hierarchical_review_csv

    output_path = export_hierarchical_review_csv(
        run_id=args.run_id,
        sample_size=args.sample_size,
        stratified=not args.no_stratify,
    )

    print("\n" + "=" * 60)
    print("HIERARCHICAL REVIEW EXPORT COMPLETE")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"\nSample size: {args.sample_size} reports")
    print(f"Stratified: {'No' if args.no_stratify else 'Yes (proportional to L1 categories)'}")
    print("\nInstructions:")
    print("1. Open the CSV in a spreadsheet application")
    print("2. For each row, fill in 'your_decision' column:")
    print("   - APPROVE: Classification is correct")
    print("   - REJECT: Classification is wrong")
    print("   - CHANGE: Classification should be different (fill in 'correct_code')")
    print("   - UNCERTAIN: Not sure")
    print("3. Add notes in 'notes' column if helpful")
    print("4. Import results: python -m taxonomy.cli import-review <file>")
    print("=" * 60)

    return 0


def cmd_import_review(args, logger):
    """Import reviewed hierarchical classifications."""
    from .review import import_hierarchical_review
    from pathlib import Path

    review_file = Path(args.file)
    if not review_file.exists():
        print(f"Error: File not found: {review_file}")
        return 1

    summary = import_hierarchical_review(
        review_file=review_file,
        run_id=args.run_id,
    )

    if "error" in summary:
        print(f"Error: {summary['error']}")
        return 1

    print("\n" + "=" * 60)
    print("REVIEW IMPORT COMPLETE")
    print("=" * 60)
    print(f"Reports reviewed: {summary['overall']['reports_reviewed']}")
    print(f"Total classifications reviewed: {summary['overall']['total_reviewed']}")
    print(f"\nL1 Results:")
    print(f"  Precision: {summary['l1']['precision']:.1f}%")
    print(f"  Approved: {summary['l1']['approved']}")
    print(f"  Rejected: {summary['l1']['rejected']}")
    print(f"  Changed: {summary['l1']['changed']}")

    if summary['l2']['total'] > 0:
        print(f"\nL2 Results:")
        print(f"  Precision: {summary['l2']['precision']:.1f}%")
        print(f"  Approved: {summary['l2']['approved']}")
        print(f"  Rejected: {summary['l2']['rejected']}")
        print(f"  Changed: {summary['l2']['changed']}")

    print("=" * 60)
    return 0


def cmd_subcategories(args, logger):
    """List all L2 subcategories."""
    from .subcategories import PARENT_TO_SUBCATEGORIES, HFACS_SUBCATEGORIES
    from .cictt import CICTT_BY_CODE

    print("\n" + "=" * 60)
    print("L2 SUBCATEGORIES")
    print("=" * 60)

    total_subs = 0

    for parent_code, subcats in PARENT_TO_SUBCATEGORIES.items():
        parent_name = CICTT_BY_CODE[parent_code].name if parent_code in CICTT_BY_CODE else parent_code
        print(f"\n{parent_code} - {parent_name}")
        print("-" * 50)

        for subcat in subcats:
            print(f"  {subcat.code}: {subcat.name}")
            total_subs += 1

    print(f"\nHFACS (Human Factors) - Applied across applicable categories")
    print("-" * 50)
    for hfacs in HFACS_SUBCATEGORIES:
        print(f"  {hfacs.code}: {hfacs.name}")
        total_subs += 1

    print("\n" + "=" * 60)
    print(f"Total: {total_subs} subcategories")
    print("Sources: IATA LOC-I/CFIT Analysis, HFACS, FAA/EASA")
    print("=" * 60)
    return 0


def cmd_review_stats(args, logger):
    """Show review statistics for a run."""
    from .review import get_review_statistics

    stats = get_review_statistics(args.run_id)

    if "error" in stats:
        print(f"Error: {stats['error']}")
        return 1

    print("\n" + "=" * 60)
    print(f"REVIEW STATISTICS (Run {stats['run_id']})")
    print("=" * 60)
    print(f"Review file: {stats['review_file']}")
    print(f"\nOverall:")
    print(f"  Reports reviewed: {stats['overall']['reports_reviewed']}")
    print(f"  Total reviewed: {stats['overall']['total_reviewed']}")
    print(f"\nL1 Metrics:")
    print(f"  Precision: {stats['overall']['l1_precision']:.1f}%")
    print(f"\nL2 Metrics:")
    print(f"  Precision: {stats['overall']['l2_precision']:.1f}%")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RiskRADAR CICTT Taxonomy CLI - Map reports to occurrence categories"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # map command
    map_parser = subparsers.add_parser(
        "map", help="Map reports to CICTT categories"
    )
    map_parser.add_argument(
        "--run-id", type=int, default=1, help="Run identifier"
    )

    # review command
    review_parser = subparsers.add_parser(
        "review", help="Export mappings for human review"
    )
    review_parser.add_argument(
        "--run-id", type=int, default=1, help="Run identifier"
    )
    review_parser.add_argument(
        "--csv", action="store_true", help="Export as CSV instead of HTML"
    )
    review_parser.add_argument(
        "--max-reports", type=int, default=10,
        help="Max reports per category in HTML (default: 10)"
    )

    # stats command
    subparsers.add_parser("stats", help="Show taxonomy statistics")

    # categories command
    subparsers.add_parser("categories", help="List CICTT categories")

    # =========================================================================
    # L2 Subcategory Classification Commands
    # =========================================================================

    # classify-l2 command (builds on existing L1)
    classify_l2_parser = subparsers.add_parser(
        "classify-l2", help="Run L2 subcategory classification on existing L1 results"
    )
    classify_l2_parser.add_argument(
        "--l1-run", type=int, default=1,
        help="L1 run ID to build on (default: 1)"
    )
    classify_l2_parser.add_argument(
        "--run-id", type=int, default=1,
        help="L2 run identifier (default: 1)"
    )

    # review-l2 command (HTML review like L1)
    review_l2_parser = subparsers.add_parser(
        "review-l2", help="Export L2 results for human review (HTML)"
    )
    review_l2_parser.add_argument(
        "--l1-run", type=int, default=1, help="L1 run ID"
    )
    review_l2_parser.add_argument(
        "--l2-run", type=int, default=1, help="L2 run ID"
    )
    review_l2_parser.add_argument(
        "--max-reports", type=int, default=10,
        help="Max reports per subcategory (default: 10)"
    )
    review_l2_parser.add_argument(
        "--csv", action="store_true",
        help="Export as CSV instead of HTML"
    )

    # export-review command (for backward compat, points to review-l2)
    export_review_parser = subparsers.add_parser(
        "export-review", help="[Deprecated] Use review-l2 instead"
    )
    export_review_parser.add_argument(
        "--run-id", type=int, default=1, help="Run identifier"
    )
    export_review_parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Number of reports to include in review (default: 50)"
    )
    export_review_parser.add_argument(
        "--no-stratify", action="store_true",
        help="Disable stratified sampling (random instead)"
    )

    # import-review command
    import_review_parser = subparsers.add_parser(
        "import-review", help="Import reviewed hierarchical classifications"
    )
    import_review_parser.add_argument(
        "file", help="Path to reviewed CSV file"
    )
    import_review_parser.add_argument(
        "--run-id", type=int, default=1, help="Original run identifier"
    )

    # subcategories command
    subparsers.add_parser("subcategories", help="List L2 subcategories")

    # review-stats command
    review_stats_parser = subparsers.add_parser(
        "review-stats", help="Show review statistics for a run"
    )
    review_stats_parser.add_argument(
        "--run-id", type=int, default=1, help="Run identifier"
    )

    args = parser.parse_args()

    logger = setup_logging(args.command, args.verbose)

    commands = {
        # Original L1 commands
        "map": cmd_map,
        "review": cmd_review,
        "stats": cmd_stats,
        "categories": cmd_categories,
        # L2 Subcategory commands
        "classify-l2": cmd_classify_l2,
        "review-l2": cmd_review_l2,
        # Legacy hierarchical commands
        "export-review": cmd_export_review,
        "import-review": cmd_import_review,
        "subcategories": cmd_subcategories,
        "review-stats": cmd_review_stats,
    }

    try:
        exit_code = commands[args.command](args, logger)
        sys.exit(exit_code)
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
