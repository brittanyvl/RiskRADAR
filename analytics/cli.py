"""
RiskRADAR Analytics CLI

Interactive DuckDB shell for querying pages, documents, and chunks data.

Usage:
    py -m analytics.cli                           # Interactive shell
    py -m analytics.cli --query "SELECT ..."      # Run single query
    py -m analytics.cli --help                    # Show help
"""

import argparse
import sys
from pathlib import Path

import duckdb

from analytics.views import register_views, list_views

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_DIR = Path(__file__).parent / "data"
SQLITE_PATH = PROJECT_ROOT / "sqlite" / "riskradar.db"


def setup_connection() -> duckdb.DuckDBPyConnection:
    """
    Create DuckDB connection with Parquet tables and SQLite attached.

    Returns:
        Configured DuckDB connection
    """
    conn = duckdb.connect(":memory:")

    # Load Parquet files as tables
    parquet_files = {
        "pages": PARQUET_DIR / "pages.parquet",
        "documents": PARQUET_DIR / "documents.parquet",
        "chunks": PARQUET_DIR / "chunks.parquet",
    }

    missing = []
    for name, path in parquet_files.items():
        if path.exists():
            conn.execute(f"CREATE TABLE {name} AS SELECT * FROM '{path.as_posix()}'")
        else:
            missing.append(name)

    if missing:
        print(f"Warning: Missing Parquet files: {missing}")
        print("Run 'py -m analytics.convert' to generate them.")
        print()

    # Attach SQLite database
    sqlite_attached = False
    if SQLITE_PATH.exists():
        try:
            conn.execute(f"ATTACH '{SQLITE_PATH.as_posix()}' AS sqlite (TYPE SQLITE)")
            sqlite_attached = True
        except Exception as e:
            print(f"Warning: Could not attach SQLite database: {e}")
    else:
        print(f"Warning: SQLite database not found at {SQLITE_PATH}")

    # Register pre-built views
    register_views(conn, include_sqlite=sqlite_attached)

    return conn


def get_data_summary(conn: duckdb.DuckDBPyConnection) -> dict:
    """Get counts of loaded data."""
    summary = {}

    try:
        result = conn.execute("SELECT COUNT(*) FROM pages").fetchone()
        summary["pages"] = result[0] if result else 0
    except Exception:
        summary["pages"] = 0

    try:
        result = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        summary["documents"] = result[0] if result else 0
    except Exception:
        summary["documents"] = 0

    try:
        result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        summary["chunks"] = result[0] if result else 0
    except Exception:
        summary["chunks"] = 0

    try:
        result = conn.execute("SELECT COUNT(*) FROM sqlite.reports").fetchone()
        summary["reports"] = result[0] if result else 0
    except Exception:
        summary["reports"] = 0

    return summary


def print_banner(summary: dict):
    """Print startup banner with data summary."""
    print()
    print("=" * 60)
    print("RiskRADAR Analytics Engine (DuckDB)")
    print("=" * 60)
    print(f"Loaded: {summary['pages']:,} pages | {summary['documents']:,} documents | {summary['chunks']:,} chunks")

    if summary.get("reports"):
        print(f"SQLite attached: {SQLITE_PATH} ({summary['reports']} reports)")

    print()
    print("Available views:")
    views = list_views()
    # Print views in columns
    for i in range(0, len(views), 3):
        row = views[i:i+3]
        print("  " + ", ".join(row))

    print()
    print("Type SQL queries, '.help' for commands, or '.exit' to quit.")
    print("=" * 60)
    print()


def run_query(conn: duckdb.DuckDBPyConnection, query: str) -> bool:
    """
    Execute a query and print results.

    Returns:
        True if query succeeded, False otherwise
    """
    try:
        result = conn.execute(query)

        # Check if query returns results
        if result.description:
            # Fetch and display results
            df = result.fetchdf()
            if len(df) > 0:
                print(df.to_string(index=False))
            else:
                print("(0 rows)")
        else:
            print("OK")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def interactive_shell(conn: duckdb.DuckDBPyConnection):
    """Run interactive SQL shell."""
    summary = get_data_summary(conn)
    print_banner(summary)

    query_buffer = []

    while True:
        try:
            # Show prompt
            if query_buffer:
                prompt = "...> "
            else:
                prompt = "D > "

            line = input(prompt)

        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            query_buffer = []
            continue

        # Handle special commands
        stripped = line.strip().lower()

        if stripped in (".exit", ".quit", "exit", "quit"):
            break

        if stripped == ".help":
            print("""
Commands:
  .exit, .quit    Exit the shell
  .help           Show this help
  .tables         List available tables
  .views          List available views
  .schema TABLE   Show table schema

Example queries:
  SELECT * FROM data_summary;
  SELECT * FROM chunks_by_section LIMIT 10;
  SELECT * FROM chunks WHERE chunk_text ILIKE '%engine%' LIMIT 5;
  SELECT * FROM chunks_enriched WHERE accident_date > '2000-01-01' LIMIT 10;
""")
            continue

        if stripped == ".tables":
            run_query(conn, "SHOW TABLES")
            continue

        if stripped == ".views":
            for view in list_views():
                print(f"  {view}")
            continue

        if stripped.startswith(".schema "):
            table = stripped[8:].strip()
            run_query(conn, f"DESCRIBE {table}")
            continue

        # Accumulate multi-line queries
        query_buffer.append(line)
        full_query = " ".join(query_buffer)

        # Check if query is complete (ends with semicolon)
        if full_query.strip().endswith(";"):
            run_query(conn, full_query)
            query_buffer = []
        elif not full_query.strip():
            query_buffer = []


def main():
    parser = argparse.ArgumentParser(
        description="RiskRADAR Analytics CLI - DuckDB query interface"
    )
    parser.add_argument(
        "--query", "-q",
        help="Run a single query and exit"
    )

    args = parser.parse_args()

    # Check if Parquet files exist
    parquet_exists = any((PARQUET_DIR / f"{name}.parquet").exists()
                        for name in ["pages", "documents", "chunks"])

    if not parquet_exists:
        print("No Parquet files found. Run conversion first:")
        print("  py -m analytics.convert")
        sys.exit(1)

    # Setup connection
    conn = setup_connection()

    if args.query:
        # Single query mode
        success = run_query(conn, args.query)
        conn.close()
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        interactive_shell(conn)
        conn.close()


if __name__ == "__main__":
    main()
