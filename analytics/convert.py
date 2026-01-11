"""
JSONL to Parquet conversion for RiskRADAR analytics.

Converts pages.jsonl, documents.jsonl, and chunks.jsonl to Parquet format
for fast analytical queries with DuckDB.

Usage:
    py -m analytics.convert
"""

import logging
from pathlib import Path

import duckdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
JSONL_DIR = PROJECT_ROOT / "extraction" / "json_data"
PARQUET_DIR = Path(__file__).parent / "data"


def convert_pages(conn: duckdb.DuckDBPyConnection) -> int:
    """Convert pages.jsonl to pages.parquet."""
    jsonl_path = JSONL_DIR / "pages.jsonl"
    parquet_path = PARQUET_DIR / "pages.parquet"

    if not jsonl_path.exists():
        logger.warning(f"pages.jsonl not found at {jsonl_path}")
        return 0

    logger.info(f"Converting {jsonl_path} -> {parquet_path}")

    # Read JSONL and select/rename columns for cleaner schema
    conn.execute(f"""
        COPY (
            SELECT
                report_id,
                page_number,
                text,
                source,
                char_count,
                alphabetic_ratio,
                garbage_ratio,
                ocr_confidence,
                status
            FROM read_json_auto('{jsonl_path.as_posix()}')
        ) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Get row count
    result = conn.execute(f"SELECT COUNT(*) FROM '{parquet_path.as_posix()}'").fetchone()
    count = result[0] if result else 0

    logger.info(f"  Written {count:,} pages ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return count


def convert_documents(conn: duckdb.DuckDBPyConnection) -> int:
    """Convert documents.jsonl to documents.parquet."""
    jsonl_path = JSONL_DIR / "documents.jsonl"
    parquet_path = PARQUET_DIR / "documents.parquet"

    if not jsonl_path.exists():
        logger.warning(f"documents.jsonl not found at {jsonl_path}")
        return 0

    logger.info(f"Converting {jsonl_path} -> {parquet_path}")

    # Read JSONL - documents has simple schema
    conn.execute(f"""
        COPY (
            SELECT
                report_id,
                full_text
            FROM read_json_auto('{jsonl_path.as_posix()}')
        ) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    result = conn.execute(f"SELECT COUNT(*) FROM '{parquet_path.as_posix()}'").fetchone()
    count = result[0] if result else 0

    logger.info(f"  Written {count:,} documents ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return count


def convert_chunks(conn: duckdb.DuckDBPyConnection) -> int:
    """Convert chunks.jsonl to chunks.parquet."""
    jsonl_path = JSONL_DIR / "chunks.jsonl"
    parquet_path = PARQUET_DIR / "chunks.parquet"

    if not jsonl_path.exists():
        logger.warning(f"chunks.jsonl not found at {jsonl_path}")
        return 0

    logger.info(f"Converting {jsonl_path} -> {parquet_path}")

    # Read JSONL - chunks has the most columns
    # We flatten/exclude nested arrays for simpler queries
    conn.execute(f"""
        COPY (
            SELECT
                chunk_id,
                report_id,
                chunk_sequence,
                page_start,
                page_end,
                char_start,
                char_end,
                section_name,
                section_number,
                section_detection_method,
                chunk_text,
                token_count,
                overlap_tokens,
                text_source,
                has_footnotes,
                pipeline_version,
                created_at
            FROM read_json_auto('{jsonl_path.as_posix()}')
        ) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    result = conn.execute(f"SELECT COUNT(*) FROM '{parquet_path.as_posix()}'").fetchone()
    count = result[0] if result else 0

    logger.info(f"  Written {count:,} chunks ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return count


def main():
    """Run the full conversion pipeline."""
    logger.info("=" * 60)
    logger.info("RiskRADAR JSONL -> Parquet Conversion")
    logger.info("=" * 60)

    # Ensure output directory exists
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    # Create in-memory DuckDB connection for conversion
    conn = duckdb.connect(":memory:")

    # Convert each file
    pages_count = convert_pages(conn)
    docs_count = convert_documents(conn)
    chunks_count = convert_chunks(conn)

    conn.close()

    # Summary
    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info(f"  Pages:     {pages_count:,}")
    logger.info(f"  Documents: {docs_count:,}")
    logger.info(f"  Chunks:    {chunks_count:,}")

    # Report total size
    total_size = sum(
        f.stat().st_size for f in PARQUET_DIR.glob("*.parquet")
    )
    logger.info(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
