"""
taxonomy/cli.py
---------------
CLI entry point for CICTT-anchored causal taxonomy.

Usage:
    python -m taxonomy.cli map              # Map reports to CICTT categories
    python -m taxonomy.cli review           # Export for human review (HTML)
    python -m taxonomy.cli review --csv     # Export as CSV
    python -m taxonomy.cli stats            # Show taxonomy statistics
    python -m taxonomy.cli categories       # List CICTT categories
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

    args = parser.parse_args()

    logger = setup_logging(args.command, args.verbose)

    commands = {
        "map": cmd_map,
        "review": cmd_review,
        "stats": cmd_stats,
        "categories": cmd_categories,
    }

    try:
        exit_code = commands[args.command](args, logger)
        sys.exit(exit_code)
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
