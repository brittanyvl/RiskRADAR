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
