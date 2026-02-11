"""
CLI for Risk Profiler module.

Usage:
    python -m risk_profiler.cli init              # Initialize database
    python -m risk_profiler.cli coverage           # Show coverage report
    python -m risk_profiler.cli export-parquet     # Export to Parquet for analytics
    python -m risk_profiler.cli extract-weather    # Extract VMC/IMC from text
    python -m risk_profiler.cli extract-time       # Extract time-of-day from text
    python -m risk_profiler.cli validate-model     # Run cross-validation
    python -m risk_profiler.cli train-model        # Train + save Bayesian model
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


def cmd_extract_weather(args):
    """Extract VMC/IMC weather classification from report text."""
    import sqlite3
    from .extract_weather import run_weather_extraction

    conn = sqlite3.connect(args.db)
    jsonl_path = args.jsonl if args.jsonl else None
    results = run_weather_extraction(conn, jsonl_path=jsonl_path, verbose=True)
    conn.close()
    return 0


def cmd_extract_time(args):
    """Extract time-of-day from report text."""
    import sqlite3
    from .extract_time import run_time_extraction

    conn = sqlite3.connect(args.db)
    jsonl_path = args.jsonl if args.jsonl else None
    results = run_time_extraction(conn, jsonl_path=jsonl_path, verbose=True)
    conn.close()
    return 0


def cmd_validate_model(args):
    """Run leave-one-out cross-validation on the Bayesian model."""
    import sqlite3
    from .bayesian_model import BayesianRiskModel, VALID_FEATURES

    conn = sqlite3.connect(args.db)

    # Parse features
    features = None
    if args.features:
        features = [f.strip() for f in args.features.split(',')]
        for f in features:
            if f not in VALID_FEATURES:
                print(f"Error: Invalid feature '{f}'. Valid: {sorted(VALID_FEATURES)}")
                return 1

    print("Building model...")
    model = BayesianRiskModel(conn, features=features)

    print()
    results = model.validate(verbose=True)

    conn.close()
    return 0


def cmd_train_model(args):
    """Train and save the Bayesian model to database."""
    import sqlite3
    from .bayesian_model import BayesianRiskModel, VALID_FEATURES
    from sqlite.schema import BAYES_PRIORS_TABLE, BAYES_LIKELIHOODS_TABLE

    conn = sqlite3.connect(args.db)

    # Migrate old schema if needed (v7 -> v8: add label column to likelihoods)
    cursor = conn.cursor()
    cols = [row[1] for row in cursor.execute("PRAGMA table_info(bayes_likelihoods)").fetchall()]
    if 'label' not in cols:
        print("Migrating bayes tables to v8 schema (binary relevance)...")
        cursor.execute("DROP TABLE IF EXISTS bayes_likelihoods")
        cursor.execute("DROP TABLE IF EXISTS bayes_priors")
        conn.commit()

    # Ensure persistence tables exist with current schema
    conn.execute(BAYES_PRIORS_TABLE)
    conn.execute(BAYES_LIKELIHOODS_TABLE)
    conn.commit()

    # Parse features
    features = None
    if args.features:
        features = [f.strip() for f in args.features.split(',')]
        for f in features:
            if f not in VALID_FEATURES:
                print(f"Error: Invalid feature '{f}'. Valid: {sorted(VALID_FEATURES)}")
                return 1

    print("Training model...")
    model = BayesianRiskModel(conn, features=features)

    print("\nSaving to database...")
    model.save_to_db()

    if not args.skip_validate:
        print("\nRunning cross-validation...")
        model.validate(verbose=True)

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
    subparsers.add_parser("init", help="Initialize database tables")

    # coverage command
    subparsers.add_parser("coverage", help="Show extraction coverage")

    # export-parquet command
    export_parser = subparsers.add_parser("export-parquet", help="Export to Parquet")
    export_parser.add_argument(
        "--output", "-o",
        default="analytics/data/risk_profiler",
        help="Output directory for Parquet files"
    )

    # validate command
    subparsers.add_parser("validate", help="Validate database setup")

    # extract command
    subparsers.add_parser("extract", help="Run feature extraction")

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

    # extract-weather command
    weather_parser = subparsers.add_parser(
        "extract-weather", help="Extract VMC/IMC from report text"
    )
    weather_parser.add_argument(
        "--jsonl",
        default=None,
        help="Path to chunks JSONL file (default: extraction/json_data/chunks_v2.jsonl)"
    )

    # extract-time command
    time_parser = subparsers.add_parser(
        "extract-time", help="Extract time-of-day from report text"
    )
    time_parser.add_argument(
        "--jsonl",
        default=None,
        help="Path to chunks JSONL file (default: extraction/json_data/chunks_v2.jsonl)"
    )

    # validate-model command
    val_model_parser = subparsers.add_parser(
        "validate-model", help="Run LOO cross-validation on Bayesian model"
    )
    val_model_parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature list (default: aircraft_category,season,region)"
    )

    # train-model command
    train_parser = subparsers.add_parser(
        "train-model", help="Train and save Bayesian model"
    )
    train_parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature list (default: aircraft_category,season,region)"
    )
    train_parser.add_argument(
        "--skip-validate", action="store_true",
        help="Skip cross-validation after training"
    )

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
        "extract-weather": cmd_extract_weather,
        "extract-time": cmd_extract_time,
        "validate-model": cmd_validate_model,
        "train-model": cmd_train_model,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
