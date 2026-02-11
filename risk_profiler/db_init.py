"""
Database initialization and migration for Risk Profiler.

This script:
1. Creates all new tables
2. Populates region_lookup with US Census data
3. Migrates taxonomy from Parquet to SQLite
4. Validates the setup
"""

import sqlite3
from pathlib import Path
from datetime import datetime


def init_database(db_path: str, verbose: bool = True):
    """
    Initialize the risk profiler database tables.

    Args:
        db_path: Path to SQLite database
        verbose: Print progress messages

    Returns:
        Connection object
    """
    from .schema import init_risk_profiler_tables, populate_region_lookup

    conn = sqlite3.connect(db_path)

    if verbose:
        print(f"Initializing risk profiler tables in {db_path}")

    # Create tables
    created = init_risk_profiler_tables(conn)
    if verbose:
        print(f"  Created {len(created)} tables: {', '.join(created)}")

    # Populate region lookup
    count = populate_region_lookup(conn)
    if verbose:
        print(f"  Populated region_lookup with {count} US states/territories")

    return conn


def migrate_taxonomy_from_parquet(conn, parquet_dir: str = "taxonomy/data", verbose: bool = True):
    """
    Migrate taxonomy data from Parquet files to SQLite.

    Source files:
    - report_categories_run1.parquet (L1)
    - report_l2_run1.parquet (L2)

    Args:
        conn: SQLite connection
        parquet_dir: Directory containing Parquet files
        verbose: Print progress

    Returns:
        Tuple of (l1_count, l2_count)
    """
    try:
        import duckdb
    except ImportError:
        raise ImportError("duckdb required for Parquet migration: pip install duckdb")

    parquet_path = Path(parquet_dir)
    cursor = conn.cursor()

    # Check if already migrated
    existing = cursor.execute("SELECT COUNT(*) FROM report_taxonomy").fetchone()[0]
    if existing > 0:
        if verbose:
            print(f"  Taxonomy already migrated ({existing} records). Skipping.")
        return (0, 0)

    duck = duckdb.connect()

    # Migrate L1 categories
    l1_file = parquet_path / "report_categories_run1.parquet"
    if l1_file.exists():
        if verbose:
            print(f"  Migrating L1 from {l1_file}")

        l1_data = duck.execute(f"""
            SELECT
                report_id,
                category_code,
                category_name,
                score as confidence,
                rank
            FROM read_parquet('{l1_file}')
        """).fetchall()

        cursor.executemany("""
            INSERT INTO report_taxonomy
            (report_id, level, category_code, category_name, parent_code, confidence, rank, source_run_id)
            VALUES (?, 'L1', ?, ?, NULL, ?, ?, 1)
        """, l1_data)

        l1_count = len(l1_data)
        if verbose:
            print(f"    Inserted {l1_count} L1 records")
    else:
        l1_count = 0
        if verbose:
            print(f"  Warning: {l1_file} not found")

    # Migrate L2 subcategories
    l2_file = parquet_path / "report_l2_run1.parquet"
    if l2_file.exists():
        if verbose:
            print(f"  Migrating L2 from {l2_file}")

        # Use ROW_NUMBER to deduplicate - keep highest confidence per report/code
        l2_data = duck.execute(f"""
            WITH ranked AS (
                SELECT
                    report_id,
                    subcategory_code,
                    subcategory_name,
                    parent_code,
                    score as confidence,
                    rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY report_id, subcategory_code
                        ORDER BY score DESC
                    ) as rn
                FROM read_parquet('{l2_file}')
            )
            SELECT report_id, subcategory_code, subcategory_name, parent_code, confidence, rank
            FROM ranked
            WHERE rn = 1
        """).fetchall()

        cursor.executemany("""
            INSERT INTO report_taxonomy
            (report_id, level, category_code, category_name, parent_code, confidence, rank, source_run_id)
            VALUES (?, 'L2', ?, ?, ?, ?, ?, 1)
        """, l2_data)

        l2_count = len(l2_data)
        if verbose:
            print(f"    Inserted {l2_count} L2 records (deduplicated)")
    else:
        l2_count = 0
        if verbose:
            print(f"  Warning: {l2_file} not found")

    conn.commit()
    return (l1_count, l2_count)


def initialize_report_features(conn, verbose: bool = True):
    """
    Initialize report_features table with basic data from reports table.

    This pre-populates:
    - report_id
    - location_raw
    - accident_date
    - Derived temporal features (year, decade, month, season)

    Args:
        conn: SQLite connection
        verbose: Print progress

    Returns:
        Number of records initialized
    """
    cursor = conn.cursor()

    # Check if already initialized
    existing = cursor.execute("SELECT COUNT(*) FROM report_features").fetchone()[0]
    if existing > 0:
        if verbose:
            print(f"  report_features already has {existing} records. Skipping init.")
        return 0

    if verbose:
        print("  Initializing report_features from reports table...")

    # Insert with derived temporal features
    cursor.execute("""
        INSERT INTO report_features (
            report_id,
            location_raw,
            accident_date,
            year,
            decade,
            month,
            season,
            extracted_at
        )
        SELECT
            filename as report_id,
            location as location_raw,
            accident_date,
            CAST(strftime('%Y', accident_date) AS INTEGER) as year,
            (CAST(strftime('%Y', accident_date) AS INTEGER) / 10) * 10 as decade,
            CAST(strftime('%m', accident_date) AS INTEGER) as month,
            CASE
                WHEN CAST(strftime('%m', accident_date) AS INTEGER) IN (12, 1, 2) THEN 'Winter'
                WHEN CAST(strftime('%m', accident_date) AS INTEGER) IN (3, 4, 5) THEN 'Spring'
                WHEN CAST(strftime('%m', accident_date) AS INTEGER) IN (6, 7, 8) THEN 'Summer'
                WHEN CAST(strftime('%m', accident_date) AS INTEGER) IN (9, 10, 11) THEN 'Fall'
                ELSE NULL
            END as season,
            datetime('now') as extracted_at
        FROM reports
    """)

    count = cursor.rowcount
    conn.commit()

    if verbose:
        print(f"    Initialized {count} records with temporal features")

    return count


def validate_setup(conn, verbose: bool = True):
    """
    Validate that all tables are properly set up.

    Returns:
        Dictionary with validation results
    """
    cursor = conn.cursor()
    results = {}

    # Check table existence and counts
    tables_to_check = [
        "report_features",
        "report_taxonomy",
        "aircraft_lookup",
        "region_lookup",
        "feature_extraction_runs",
        "feature_validation",
        "extraction_coverage",
    ]

    if verbose:
        print("\nValidation Results:")
        print("-" * 40)

    for table in tables_to_check:
        try:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            results[table] = {"exists": True, "count": count}
            if verbose:
                print(f"  {table}: {count} records")
        except Exception as e:
            results[table] = {"exists": False, "error": str(e)}
            if verbose:
                print(f"  {table}: ERROR - {e}")

    # Check region lookup completeness
    region_count = cursor.execute("SELECT COUNT(DISTINCT region) FROM region_lookup").fetchone()[0]
    results["regions_defined"] = region_count
    if verbose:
        print(f"\n  Distinct regions: {region_count}")

    # Check taxonomy migration
    l1_reports = cursor.execute(
        "SELECT COUNT(DISTINCT report_id) FROM report_taxonomy WHERE level='L1'"
    ).fetchone()[0]
    l2_reports = cursor.execute(
        "SELECT COUNT(DISTINCT report_id) FROM report_taxonomy WHERE level='L2'"
    ).fetchone()[0]
    results["l1_reports"] = l1_reports
    results["l2_reports"] = l2_reports
    if verbose:
        print(f"  Reports with L1 taxonomy: {l1_reports}")
        print(f"  Reports with L2 taxonomy: {l2_reports}")

    # Check season coverage
    season_count = cursor.execute(
        "SELECT COUNT(*) FROM report_features WHERE season IS NOT NULL"
    ).fetchone()[0]
    total = cursor.execute("SELECT COUNT(*) FROM report_features").fetchone()[0]
    results["season_coverage"] = (season_count, total)
    if verbose:
        pct = round(season_count / total * 100, 1) if total > 0 else 0
        print(f"  Season extracted: {season_count}/{total} ({pct}%)")

    return results


def full_init(db_path: str = "sqlite/riskradar.db", verbose: bool = True):
    """
    Run full initialization:
    1. Create tables
    2. Populate region lookup
    3. Migrate taxonomy
    4. Initialize report_features
    5. Validate

    Args:
        db_path: Path to database
        verbose: Print progress

    Returns:
        Validation results dictionary
    """
    if verbose:
        print("=" * 50)
        print("RISK PROFILER DATABASE INITIALIZATION")
        print("=" * 50)
        print()

    # Initialize
    conn = init_database(db_path, verbose=verbose)

    # Migrate taxonomy
    if verbose:
        print("\nMigrating taxonomy data...")
    l1, l2 = migrate_taxonomy_from_parquet(conn, verbose=verbose)

    # Initialize features
    if verbose:
        print("\nInitializing report features...")
    initialize_report_features(conn, verbose=verbose)

    # Validate
    results = validate_setup(conn, verbose=verbose)

    conn.close()

    if verbose:
        print("\n" + "=" * 50)
        print("INITIALIZATION COMPLETE")
        print("=" * 50)

    return results


if __name__ == "__main__":
    full_init()
