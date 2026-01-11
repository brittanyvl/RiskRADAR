"""
extraction/processing/consolidate_pages.py
------------------------------------------
Pass 0: Consolidate scattered JSON files into pages.jsonl

Merges per-page JSON files from passed/ and ocr_retry/ directories
into a single ordered JSONL file with deduplication.

Deduplication Logic:
- For each (report_id, page_number) pair, prefer embedded over OCR
- Exclude OCR pages with mean_ocr_confidence < 50
- Sort by (report_id, page_number) for correct assembly order

Output: extraction/json_data/pages.jsonl
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from riskradar.config import DB_PATH
from sqlite.connection import init_db

logger = logging.getLogger(__name__)


# Quality threshold for OCR pages
MIN_OCR_CONFIDENCE = 50.0

# Pipeline version
CONSOLIDATE_PAGES_VERSION = "1.0.0"


def get_pages_for_consolidation(conn) -> list[dict]:
    """
    Query database for pages to include in consolidation.

    Deduplication: For each (report_id, page_number), prefer embedded.
    Quality filter: Exclude OCR pages with low confidence.

    Returns:
        List of page records with json_path and metadata
    """
    # This query handles deduplication by preferring embedded over ocr
    # Using a subquery to rank by extraction_method
    query = """
        WITH ranked_pages AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY report_id, page_number
                    ORDER BY
                        CASE extraction_method
                            WHEN 'embedded' THEN 1
                            WHEN 'ocr' THEN 2
                        END
                ) as rn
            FROM pages p
            WHERE status IN ('passed', 'ocr_retry')
              AND (
                  extraction_method = 'embedded'
                  OR (extraction_method = 'ocr' AND mean_ocr_confidence >= ?)
              )
        )
        SELECT
            report_id,
            page_number,
            json_path,
            extraction_method,
            char_count,
            alphabetic_ratio,
            garbage_ratio,
            mean_ocr_confidence,
            status
        FROM ranked_pages
        WHERE rn = 1
        ORDER BY report_id, page_number
    """

    cursor = conn.execute(query, (MIN_OCR_CONFIDENCE,))
    columns = [desc[0] for desc in cursor.description]

    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def load_page_json(json_path: Path) -> dict | None:
    """
    Load a page JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Parsed JSON dict or None if failed
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return None


def consolidate_pages(
    output_path: Path,
    extraction_base: Path,
    limit: int | None = None,
    conn=None
) -> dict:
    """
    Consolidate page JSON files into pages.jsonl.

    Args:
        output_path: Path to write pages.jsonl
        extraction_base: Base path for extraction (to resolve json_path)
        limit: Optional limit on number of reports to process
        conn: Optional database connection (will create if not provided)

    Returns:
        Dict with consolidation statistics
    """
    close_conn = False
    if conn is None:
        conn = init_db(DB_PATH)
        close_conn = True

    try:
        # Get pages from database
        pages = get_pages_for_consolidation(conn)
        logger.info(f"Found {len(pages)} pages for consolidation")

        if limit:
            # Group by report and limit
            reports = {}
            for page in pages:
                report_id = page["report_id"]
                if report_id not in reports:
                    reports[report_id] = []
                reports[report_id].append(page)

            # Take first N reports
            limited_reports = list(reports.keys())[:limit]
            pages = [p for p in pages if p["report_id"] in limited_reports]
            logger.info(f"Limited to {len(pages)} pages from {limit} reports")

        # Statistics
        stats = {
            "total_pages": len(pages),
            "embedded_pages": 0,
            "ocr_pages": 0,
            "skipped_low_confidence": 0,
            "skipped_duplicate": 0,
            "failed_to_load": 0,
            "written_pages": 0,
            "unique_reports": len(set(p["report_id"] for p in pages)),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write to JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for page_record in pages:
                # Resolve JSON path
                json_path = extraction_base / page_record["json_path"]

                # Load page JSON
                page_data = load_page_json(json_path)
                if page_data is None:
                    stats["failed_to_load"] += 1
                    logger.warning(
                        f"Failed to load: {page_record['report_id']} "
                        f"page {page_record['page_number']}"
                    )
                    continue

                # Build output record
                output_record = {
                    "report_id": page_record["report_id"],
                    "page_number": page_record["page_number"],
                    "text": page_data.get("text", ""),
                    "source": page_record["extraction_method"],
                    "char_count": page_record["char_count"],
                    "alphabetic_ratio": page_record["alphabetic_ratio"],
                    "garbage_ratio": page_record["garbage_ratio"],
                    "ocr_confidence": page_record["mean_ocr_confidence"],
                    "status": page_record["status"],
                }

                # Write line
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                stats["written_pages"] += 1

                # Count by source
                if page_record["extraction_method"] == "embedded":
                    stats["embedded_pages"] += 1
                else:
                    stats["ocr_pages"] += 1

        stats["completed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Consolidation complete: {stats['written_pages']} pages written "
            f"({stats['embedded_pages']} embedded, {stats['ocr_pages']} OCR)"
        )

        return stats

    finally:
        if close_conn:
            conn.close()


def get_consolidation_stats(pages_jsonl_path: Path) -> dict:
    """
    Get statistics from an existing pages.jsonl file.

    Args:
        pages_jsonl_path: Path to pages.jsonl

    Returns:
        Dict with file statistics
    """
    if not pages_jsonl_path.exists():
        return {"exists": False}

    stats = {
        "exists": True,
        "total_lines": 0,
        "unique_reports": set(),
        "embedded_count": 0,
        "ocr_count": 0,
        "file_size_bytes": pages_jsonl_path.stat().st_size,
    }

    with open(pages_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                stats["total_lines"] += 1
                try:
                    record = json.loads(line)
                    stats["unique_reports"].add(record.get("report_id"))
                    if record.get("source") == "embedded":
                        stats["embedded_count"] += 1
                    else:
                        stats["ocr_count"] += 1
                except json.JSONDecodeError:
                    pass

    stats["unique_reports"] = len(stats["unique_reports"])
    return stats


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Determine paths
    extraction_base = Path(__file__).parent.parent
    output_path = extraction_base / "json_data" / "pages.jsonl"

    # Parse optional limit argument
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            logger.info(f"Limiting to {limit} reports")
        except ValueError:
            logger.error(f"Invalid limit: {sys.argv[1]}")
            sys.exit(1)

    # Run consolidation
    stats = consolidate_pages(
        output_path=output_path,
        extraction_base=extraction_base,
        limit=limit
    )

    # Print summary
    print("\nConsolidation Summary:")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Written pages: {stats['written_pages']}")
    print(f"  Embedded pages: {stats['embedded_pages']}")
    print(f"  OCR pages: {stats['ocr_pages']}")
    print(f"  Failed to load: {stats['failed_to_load']}")
    print(f"  Unique reports: {stats['unique_reports']}")
    print(f"\nOutput: {output_path}")
