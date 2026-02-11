"""
CLI for Risk Profiler module.

Usage:
    python -m risk_profiler.cli init          # Initialize database
    python -m risk_profiler.cli coverage      # Show coverage report
    python -m risk_profiler.cli export-parquet # Export to Parquet for analytics
"""

import argparse
import sys
from pathlib import Path


def cmd_init(args):
    """Initialize the database."""
    from .db_init import full_init

    results = full_init(db_path=args.db, verbose=True)
    return 0


def cmd_coverage(args):
    """Show coverage report."""
    import sqlite3
    from .schema import print_coverage_report

    conn = sqlite3.connect(args.db)
    print_coverage_report(conn)
    conn.close()
    return 0


def cmd_export_parquet(args):
    """Export feature data to Parquet for analytics."""
    import sqlite3
    from datetime import datetime

    try:
        import duckdb
    except ImportError:
        print("Error: duckdb required for Parquet export. Install with: pip install duckdb")
        return 1

    conn = sqlite3.connect(args.db)
    duck = duckdb.connect()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {output_dir}/")

    # Export report_features
    print("  Exporting report_features...")
    duck.execute(f"""
        COPY (SELECT * FROM sqlite_scan('{args.db}', 'report_features'))
        TO '{output_dir}/report_features.parquet' (FORMAT PARQUET)
    """)

    # Export report_taxonomy
    print("  Exporting report_taxonomy...")
    duck.execute(f"""
        COPY (SELECT * FROM sqlite_scan('{args.db}', 'report_taxonomy'))
        TO '{output_dir}/report_taxonomy.parquet' (FORMAT PARQUET)
    """)

    # Export joined view for easy analytics
    print("  Exporting enriched features view...")
    duck.execute(f"""
        COPY (
            SELECT
                f.*,
                r.title,
                r.pdf_url
            FROM sqlite_scan('{args.db}', 'report_features') f
            JOIN sqlite_scan('{args.db}', 'reports') r
                ON f.report_id = r.filename
        )
        TO '{output_dir}/features_enriched.parquet' (FORMAT PARQUET)
    """)

    print("\nExport complete!")
    print(f"  - {output_dir}/report_features.parquet")
    print(f"  - {output_dir}/report_taxonomy.parquet")
    print(f"  - {output_dir}/features_enriched.parquet")

    conn.close()
    return 0


def cmd_validate(args):
    """Validate database setup."""
    import sqlite3
    from .db_init import validate_setup

    conn = sqlite3.connect(args.db)
    results = validate_setup(conn, verbose=True)
    conn.close()
    return 0


def cmd_extract(args):
    """Run feature extraction pipeline."""
    import sqlite3
    from .extract_features import run_feature_extraction

    conn = sqlite3.connect(args.db)
    results = run_feature_extraction(conn, verbose=True)
    conn.close()
    return 0


def cmd_classify_types(args):
    """Classify reports by type (accident, safety_study, etc.)."""
    import sqlite3
    from .report_types import classify_all_reports

    conn = sqlite3.connect(args.db)
    results = classify_all_reports(conn, dry_run=args.dry_run)
    conn.close()
    return 0


def cmd_load_chunks(args):
    """Load chunk metadata from JSONL into SQLite."""
    import sqlite3
    from extraction.load_chunks import load_chunks

    conn = sqlite3.connect(args.db)
    load_chunks(conn)
    conn.close()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Risk Profiler CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db",
        default="sqlite/riskradar.db",
        help="Path to SQLite database"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize database tables")

    # coverage command
    coverage_parser = subparsers.add_parser("coverage", help="Show extraction coverage")

    # export-parquet command
    export_parser = subparsers.add_parser("export-parquet", help="Export to Parquet")
    export_parser.add_argument(
        "--output", "-o",
        default="analytics/data/risk_profiler",
        help="Output directory for Parquet files"
    )

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate database setup")

    # extract command
    extract_parser = subparsers.add_parser("extract", help="Run feature extraction")

    # classify-types command
    classify_types_parser = subparsers.add_parser(
        "classify-types", help="Classify reports by type"
    )
    classify_types_parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify without writing to database"
    )

    # load-chunks command
    subparsers.add_parser("load-chunks", help="Load chunk metadata from JSONL into SQLite")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "init": cmd_init,
        "coverage": cmd_coverage,
        "export-parquet": cmd_export_parquet,
        "validate": cmd_validate,
        "extract": cmd_extract,
        "classify-types": cmd_classify_types,
        "load-chunks": cmd_load_chunks,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
