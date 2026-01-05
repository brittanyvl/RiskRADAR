"""
sqlite/queries.py
-----------------
CRUD operations for RiskRADAR database.
"""
import sqlite3
from datetime import datetime


# --- Reports table ---

def is_already_downloaded(conn: sqlite3.Connection, filename: str) -> bool:
    """Check if a PDF has already been downloaded successfully."""
    cursor = conn.execute(
        "SELECT 1 FROM reports WHERE filename = ? AND status = 'completed'",
        (filename,)
    )
    return cursor.fetchone() is not None


def insert_report(conn: sqlite3.Connection, report: dict) -> int:
    """
    Insert or update report metadata.

    Args:
        conn: Database connection
        report: Dict with keys: filename, title, location, accident_date,
                report_date, report_number, pdf_url

    Returns:
        Row ID of inserted/updated report
    """
    cursor = conn.execute(
        """
        INSERT INTO reports (filename, title, location, accident_date,
                            report_date, report_number, pdf_url, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
        ON CONFLICT(filename) DO UPDATE SET
            title = excluded.title,
            location = excluded.location,
            accident_date = excluded.accident_date,
            report_date = excluded.report_date,
            report_number = excluded.report_number,
            pdf_url = excluded.pdf_url
        """,
        (
            report["filename"],
            report.get("title"),
            report.get("location"),
            report.get("accident_date"),
            report.get("report_date"),
            report.get("report_number"),
            report.get("pdf_url"),
        )
    )
    conn.commit()
    return cursor.lastrowid


def update_report_status(
    conn: sqlite3.Connection,
    filename: str,
    status: str,
    local_path: str | None = None,
    sha256: str | None = None,
) -> None:
    """Update report status and optionally set local_path, sha256, downloaded_at."""
    if status == "completed" and local_path:
        conn.execute(
            """
            UPDATE reports
            SET status = ?, local_path = ?, sha256 = ?, downloaded_at = ?
            WHERE filename = ?
            """,
            (status, local_path, sha256, datetime.now().isoformat(), filename)
        )
    else:
        conn.execute(
            "UPDATE reports SET status = ? WHERE filename = ?",
            (status, filename)
        )
    conn.commit()


def get_pending_reports(conn: sqlite3.Connection) -> list[dict]:
    """Get all reports that haven't been downloaded yet."""
    cursor = conn.execute(
        "SELECT * FROM reports WHERE status IN ('pending', 'failed')"
    )
    return [dict(row) for row in cursor.fetchall()]


# --- Scrape progress tracking ---

def get_resume_point(conn: sqlite3.Connection) -> tuple[int, int]:
    """
    Get the last saved scraping position.

    Returns:
        (last_page, last_report_index) tuple, or (0, 0) if no progress saved
    """
    cursor = conn.execute(
        "SELECT last_page, last_report_index FROM scrape_progress WHERE id = 1"
    )
    row = cursor.fetchone()
    if row:
        return row["last_page"], row["last_report_index"]
    return 0, 0


def update_resume_point(
    conn: sqlite3.Connection,
    page: int,
    report_index: int
) -> None:
    """Save current scraping position for resume."""
    conn.execute(
        """
        INSERT INTO scrape_progress (id, last_page, last_report_index, updated_at)
        VALUES (1, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            last_page = excluded.last_page,
            last_report_index = excluded.last_report_index,
            updated_at = excluded.updated_at
        """,
        (page, report_index, datetime.now().isoformat())
    )
    conn.commit()


def reset_resume_point(conn: sqlite3.Connection) -> None:
    """Reset progress to start from beginning."""
    conn.execute("DELETE FROM scrape_progress WHERE id = 1")
    conn.commit()


# --- Error logging ---

def log_error(
    conn: sqlite3.Connection,
    filename: str | None,
    error_type: str,
    message: str,
) -> None:
    """Record a scraping error for debugging."""
    conn.execute(
        """
        INSERT INTO scrape_errors (report_filename, error_type, error_message, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (filename, error_type, message, datetime.now().isoformat())
    )
    conn.commit()


# --- Statistics ---

def get_scrape_stats(conn: sqlite3.Connection) -> dict:
    """Get counts of reports by status."""
    cursor = conn.execute(
        """
        SELECT status, COUNT(*) as count
        FROM reports
        GROUP BY status
        """
    )
    stats = {row["status"]: row["count"] for row in cursor.fetchall()}

    cursor = conn.execute("SELECT COUNT(*) as total FROM reports")
    stats["total"] = cursor.fetchone()["total"]

    return stats


# --- Phase 3: Pages table ---

def insert_page(conn: sqlite3.Connection, page_data: dict) -> int:
    """
    Insert or update page extraction results.

    Args:
        page_data: Dict with keys: report_id, page_number, json_path, status,
                   extraction_method, extraction_pass, extraction_time_ms,
                   char_count, alphabetic_ratio, garbage_ratio, word_count,
                   passes_threshold, mean_ocr_confidence (optional),
                   low_confidence_word_count (optional)

    Returns:
        Row ID of inserted/updated page
    """
    cursor = conn.execute(
        """
        INSERT INTO pages (
            report_id, page_number, json_path, status,
            extraction_method, extraction_pass, extraction_time_ms, extracted_at,
            char_count, alphabetic_ratio, garbage_ratio, word_count, passes_threshold,
            mean_ocr_confidence, low_confidence_word_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(report_id, page_number) DO UPDATE SET
            json_path = excluded.json_path,
            status = excluded.status,
            extraction_method = excluded.extraction_method,
            extraction_pass = excluded.extraction_pass,
            extraction_time_ms = excluded.extraction_time_ms,
            extracted_at = excluded.extracted_at,
            char_count = excluded.char_count,
            alphabetic_ratio = excluded.alphabetic_ratio,
            garbage_ratio = excluded.garbage_ratio,
            word_count = excluded.word_count,
            passes_threshold = excluded.passes_threshold,
            mean_ocr_confidence = excluded.mean_ocr_confidence,
            low_confidence_word_count = excluded.low_confidence_word_count
        """,
        (
            page_data["report_id"],
            page_data["page_number"],
            page_data["json_path"],
            page_data["status"],
            page_data["extraction_method"],
            page_data["extraction_pass"],
            page_data.get("extraction_time_ms"),
            datetime.now().isoformat(),
            page_data.get("char_count"),
            page_data.get("alphabetic_ratio"),
            page_data.get("garbage_ratio"),
            page_data.get("word_count"),
            page_data.get("passes_threshold"),
            page_data.get("mean_ocr_confidence"),
            page_data.get("low_confidence_word_count"),
        )
    )
    conn.commit()
    return cursor.lastrowid


def get_extracted_pages(conn: sqlite3.Connection, report_id: str) -> set[int]:
    """Get set of page numbers already extracted for a report."""
    cursor = conn.execute(
        "SELECT page_number FROM pages WHERE report_id = ?",
        (report_id,)
    )
    return {row["page_number"] for row in cursor.fetchall()}


def get_failed_pages(conn: sqlite3.Connection) -> list[dict]:
    """Get all pages that failed quality checks and need OCR."""
    cursor = conn.execute(
        """
        SELECT p.*, r.local_path as pdf_path
        FROM pages p
        JOIN reports r ON p.report_id = r.filename
        WHERE p.status = 'failed'
        ORDER BY p.report_id, p.page_number
        """
    )
    return [dict(row) for row in cursor.fetchall()]


# --- Extraction runs ---

def create_extraction_run(
    conn: sqlite3.Connection,
    run_type: str,
    config_json: str | None = None
) -> int:
    """
    Create a new extraction run.

    Args:
        run_type: 'initial', 'ocr_retry', or 'full_reprocess'
        config_json: JSON string of configuration used

    Returns:
        Run ID
    """
    cursor = conn.execute(
        """
        INSERT INTO extraction_runs (run_type, started_at, status, config_json)
        VALUES (?, ?, 'running', ?)
        """,
        (run_type, datetime.now().isoformat(), config_json)
    )
    conn.commit()
    return cursor.lastrowid


def update_extraction_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str | None = None,
    **stats
) -> None:
    """
    Update extraction run status and stats.

    Args:
        run_id: Run ID to update
        status: New status ('running', 'completed', 'failed', 'interrupted')
        **stats: Stats to update (total_pages, passed_pages, failed_pages, etc.)
    """
    updates = []
    values = []

    if status:
        updates.append("status = ?")
        values.append(status)
        if status in ('completed', 'failed', 'interrupted'):
            updates.append("completed_at = ?")
            values.append(datetime.now().isoformat())

    for key, value in stats.items():
        updates.append(f"{key} = ?")
        values.append(value)

    if updates:
        values.append(run_id)
        conn.execute(
            f"UPDATE extraction_runs SET {', '.join(updates)} WHERE id = ?",
            tuple(values)
        )
        conn.commit()


def get_latest_run(conn: sqlite3.Connection, run_type: str | None = None) -> dict | None:
    """Get the most recent extraction run, optionally filtered by type."""
    query = "SELECT * FROM extraction_runs"
    params = ()

    if run_type:
        query += " WHERE run_type = ?"
        params = (run_type,)

    query += " ORDER BY id DESC LIMIT 1"

    cursor = conn.execute(query, params)
    row = cursor.fetchone()
    return dict(row) if row else None


# --- Extraction errors ---

def log_extraction_error(
    conn: sqlite3.Connection,
    report_id: str,
    error_type: str,
    error_message: str,
    run_id: int | None = None,
    page_number: int | None = None,
    stack_trace: str | None = None,
) -> None:
    """Record an extraction error."""
    conn.execute(
        """
        INSERT INTO extraction_errors
        (run_id, report_id, page_number, error_type, error_message, stack_trace, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, report_id, page_number, error_type, error_message, stack_trace,
         datetime.now().isoformat())
    )
    conn.commit()
