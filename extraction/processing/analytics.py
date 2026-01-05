"""
processing/analytics.py
-----------------------
Reporting and analytics queries for extraction quality assessment.

Provides queryable analytics for troubleshooting and quality monitoring.
"""

import sqlite3
from typing import Optional


def get_quality_summary(conn: sqlite3.Connection) -> dict:
    """
    Get overall quality statistics.

    Returns:
        Dict with keys:
        - total_pages: int
        - total_reports: int
        - embedded_count: int
        - ocr_count: int
        - passed_count: int
        - failed_count: int
        - avg_alphabetic_ratio: float
        - avg_ocr_confidence: float (for OCR pages only)
        - ocr_good_count: int (mean_conf >= 80)
        - ocr_acceptable_count: int (60-80)
        - ocr_poor_count: int (<60)
    """
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total_pages,
            COUNT(DISTINCT report_id) as total_reports,
            SUM(CASE WHEN extraction_method = 'embedded' THEN 1 ELSE 0 END) as embedded_count,
            SUM(CASE WHEN extraction_method = 'ocr' THEN 1 ELSE 0 END) as ocr_count,
            SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed_count,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count,
            AVG(alphabetic_ratio) as avg_alphabetic_ratio,
            AVG(CASE WHEN extraction_method = 'ocr' THEN mean_ocr_confidence ELSE NULL END) as avg_ocr_confidence,
            SUM(CASE WHEN mean_ocr_confidence >= 80.0 THEN 1 ELSE 0 END) as ocr_good_count,
            SUM(CASE WHEN mean_ocr_confidence >= 60.0 AND mean_ocr_confidence < 80.0 THEN 1 ELSE 0 END) as ocr_acceptable_count,
            SUM(CASE WHEN mean_ocr_confidence < 60.0 AND mean_ocr_confidence IS NOT NULL THEN 1 ELSE 0 END) as ocr_poor_count
        FROM pages
        WHERE status IN ('passed', 'failed', 'ocr_retry')
        """
    )

    row = cursor.fetchone()
    if row:
        return dict(row)
    return {}


def get_quality_by_decade(conn: sqlite3.Connection) -> list[dict]:
    """
    Get quality breakdown by publication decade.

    Extracts decade from report_id (assumes format like "AIR99XX.pdf" for 1990s).

    Returns:
        List of dicts with keys:
        - decade: str (e.g., "60", "70", "90", "00", "10", "20")
        - total_pages: int
        - embedded_count: int
        - ocr_count: int
        - avg_quality: float (avg alphabetic_ratio)
        - avg_ocr_conf: float (avg OCR confidence)
    """
    cursor = conn.execute(
        """
        SELECT
            SUBSTR(report_id, 4, 2) as decade,
            COUNT(*) as total_pages,
            SUM(CASE WHEN extraction_method = 'embedded' THEN 1 ELSE 0 END) as embedded_count,
            SUM(CASE WHEN extraction_method = 'ocr' THEN 1 ELSE 0 END) as ocr_count,
            AVG(alphabetic_ratio) as avg_quality,
            AVG(mean_ocr_confidence) as avg_ocr_conf
        FROM pages
        WHERE status IN ('passed', 'ocr_retry')
        GROUP BY decade
        ORDER BY decade
        """
    )

    return [dict(row) for row in cursor.fetchall()]


def get_low_confidence_pages(
    conn: sqlite3.Connection,
    threshold: float = 60.0,
    limit: Optional[int] = 100
) -> list[dict]:
    """
    Get pages with OCR confidence below threshold.

    Args:
        threshold: Confidence threshold (default: 60.0)
        limit: Maximum number of results (default: 100)

    Returns:
        List of dicts with keys:
        - report_id: str
        - page_number: int
        - json_path: str
        - mean_ocr_confidence: float
        - low_confidence_word_count: int
        - title: str (from reports table)
        - accident_date: str (from reports table)
    """
    query = """
        SELECT
            p.report_id,
            p.page_number,
            p.json_path,
            p.mean_ocr_confidence,
            p.low_confidence_word_count,
            r.title,
            r.accident_date
        FROM pages p
        JOIN reports r ON p.report_id = r.filename
        WHERE p.mean_ocr_confidence < ?
          AND p.extraction_method = 'ocr'
        ORDER BY p.mean_ocr_confidence ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query, (threshold,))
    return [dict(row) for row in cursor.fetchall()]


def get_extraction_runs(conn: sqlite3.Connection) -> list[dict]:
    """
    Get history of extraction runs.

    Returns:
        List of dicts with keys:
        - id: int
        - run_type: str ('initial', 'ocr_retry', 'full_reprocess')
        - started_at: str
        - completed_at: str
        - status: str
        - total_reports: int
        - total_pages: int
        - passed_pages: int
        - failed_pages: int
        - error_pages: int
    """
    cursor = conn.execute(
        "SELECT * FROM extraction_runs ORDER BY id DESC"
    )
    return [dict(row) for row in cursor.fetchall()]


def get_run_errors(
    conn: sqlite3.Connection,
    run_id: Optional[int] = None
) -> list[dict]:
    """
    Get error log for a specific run or all runs.

    Args:
        run_id: Filter by run ID (None = all runs)

    Returns:
        List of dicts with keys:
        - id: int
        - run_id: int
        - report_id: str
        - page_number: int
        - error_type: str
        - error_message: str
        - created_at: str
    """
    if run_id:
        cursor = conn.execute(
            "SELECT * FROM extraction_errors WHERE run_id = ? ORDER BY id DESC",
            (run_id,)
        )
    else:
        cursor = conn.execute(
            "SELECT * FROM extraction_errors ORDER BY id DESC"
        )

    return [dict(row) for row in cursor.fetchall()]


def get_status_by_report(conn: sqlite3.Connection, report_id: str) -> dict:
    """
    Get detailed status for a specific report.

    Args:
        report_id: Report filename

    Returns:
        Dict with keys:
        - report_id: str
        - total_pages: int
        - passed_count: int
        - failed_count: int
        - ocr_retry_count: int
        - avg_alphabetic_ratio: float
        - avg_ocr_confidence: float (if any OCR pages)
        - pages: list of page details
    """
    # Summary stats
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total_pages,
            SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) as passed_count,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count,
            SUM(CASE WHEN status = 'ocr_retry' THEN 1 ELSE 0 END) as ocr_retry_count,
            AVG(alphabetic_ratio) as avg_alphabetic_ratio,
            AVG(CASE WHEN extraction_method = 'ocr' THEN mean_ocr_confidence ELSE NULL END) as avg_ocr_confidence
        FROM pages
        WHERE report_id = ?
        """,
        (report_id,)
    )

    summary = dict(cursor.fetchone())
    summary["report_id"] = report_id

    # Page details
    cursor = conn.execute(
        """
        SELECT
            page_number, status, extraction_method, char_count,
            alphabetic_ratio, mean_ocr_confidence
        FROM pages
        WHERE report_id = ?
        ORDER BY page_number
        """,
        (report_id,)
    )

    summary["pages"] = [dict(row) for row in cursor.fetchall()]

    return summary


def print_quality_summary(conn: sqlite3.Connection) -> None:
    """
    Print a formatted quality summary report.

    Args:
        conn: Database connection
    """
    summary = get_quality_summary(conn)

    if not summary or summary.get("total_pages", 0) == 0:
        print("No extraction data available yet.")
        return

    print("\n" + "=" * 60)
    print("Extraction Quality Summary")
    print("=" * 60)

    total = summary["total_pages"]
    embedded = summary["embedded_count"]
    ocr = summary["ocr_count"]
    passed = summary["passed_count"]
    failed = summary["failed_count"]

    print(f"\nTotal Pages: {total:,}")
    print(f"  Passed (embedded):   {passed - summary['ocr_good_count'] - summary['ocr_acceptable_count']:>7,} ({(passed - summary['ocr_good_count'] - summary['ocr_acceptable_count'])/total*100:>5.1f}%)")
    print(f"  Passed (OCR):        {summary['ocr_good_count'] + summary['ocr_acceptable_count']:>7,} ({(summary['ocr_good_count'] + summary['ocr_acceptable_count'])/total*100:>5.1f}%)")
    print(f"  Failed/Low Quality:  {failed:>7,} ({failed/total*100:>5.1f}%)")

    if ocr > 0:
        print(f"\nOCR Confidence Distribution:")
        print(f"  Good (≥80):      {summary['ocr_good_count']:>7,} ({summary['ocr_good_count']/ocr*100:>5.1f}%)")
        print(f"  Acceptable:      {summary['ocr_acceptable_count']:>7,} ({summary['ocr_acceptable_count']/ocr*100:>5.1f}%)")
        print(f"  Poor (<60):      {summary['ocr_poor_count']:>7,} ({summary['ocr_poor_count']/ocr*100:>5.1f}%)")
        print(f"\nAvg OCR confidence: {summary['avg_ocr_confidence']:.1f}%")

    print(f"\nReports processed: {summary['total_reports']:,}")
    print("=" * 60 + "\n")


def print_quality_by_decade(conn: sqlite3.Connection) -> None:
    """
    Print a formatted per-decade quality breakdown.

    Args:
        conn: Database connection
    """
    data = get_quality_by_decade(conn)

    if not data:
        print("No extraction data available yet.")
        return

    print("\n" + "=" * 70)
    print("Quality Breakdown by Decade")
    print("=" * 70)
    print(f"{'Decade':<10} {'Pages':>8} {'Embedded':>10} {'OCR':>8} {'Avg Quality':>12} {'Avg OCR Conf':>13}")
    print("-" * 70)

    for row in data:
        decade = f"19{row['decade']}" if int(row['decade']) >= 60 else f"20{row['decade']}"
        ocr_conf = f"{row['avg_ocr_conf']:.1f}%" if row['avg_ocr_conf'] else "N/A"
        print(f"{decade:<10} {row['total_pages']:>8,} {row['embedded_count']:>10,} "
              f"{row['ocr_count']:>8,} {row['avg_quality']:>11.2f} {ocr_conf:>13}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test analytics
    from sqlite.connection import init_db
    from riskradar.config import DB_PATH

    conn = init_db(DB_PATH)

    print_quality_summary(conn)
    print_quality_by_decade(conn)

    # Test low confidence pages
    low_conf = get_low_confidence_pages(conn, threshold=60.0, limit=10)
    if low_conf:
        print(f"\nTop 10 Low Confidence Pages (<60%):")
        print("-" * 60)
        for page in low_conf:
            print(f"  {page['report_id']} p{page['page_number']}: "
                  f"{page['mean_ocr_confidence']:.1f}% confidence")

    conn.close()
