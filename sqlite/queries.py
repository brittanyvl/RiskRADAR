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


# ============================================================================
# Phase 4: Chunking
# ============================================================================

# --- Pages consolidation queries ---

def get_pages_for_consolidation(
    conn: sqlite3.Connection,
    min_ocr_confidence: float = 50.0
) -> list[dict]:
    """
    Get all pages ready for consolidation, ordered by report and page number.

    Applies quality filters:
    - Include 'passed' pages (embedded text)
    - Include 'ocr_retry' pages with mean_ocr_confidence >= min_ocr_confidence
    - Exclude low-confidence OCR pages

    When both embedded and OCR exist for same page, returns both
    (caller should prefer embedded).

    Returns:
        List of page dicts with report_id, page_number, json_path, status,
        extraction_method, quality metrics, etc.
    """
    cursor = conn.execute(
        """
        SELECT
            p.*,
            r.local_path as pdf_path
        FROM pages p
        JOIN reports r ON p.report_id = r.filename
        WHERE
            p.status = 'passed'
            OR (p.status = 'ocr_retry' AND
                (p.mean_ocr_confidence IS NULL OR p.mean_ocr_confidence >= ?))
        ORDER BY p.report_id, p.page_number
        """,
        (min_ocr_confidence,)
    )
    return [dict(row) for row in cursor.fetchall()]


def get_report_ids_with_pages(conn: sqlite3.Connection) -> list[str]:
    """Get list of report_ids that have extracted pages."""
    cursor = conn.execute(
        """
        SELECT DISTINCT report_id
        FROM pages
        WHERE status IN ('passed', 'ocr_retry')
        ORDER BY report_id
        """
    )
    return [row["report_id"] for row in cursor.fetchall()]


# --- Documents table ---

def insert_document(conn: sqlite3.Connection, doc_data: dict) -> int:
    """
    Insert or update a consolidated document.

    Args:
        doc_data: Dict with keys: report_id, total_pages, included_pages,
                  skipped_pages_json, primary_source, embedded_page_count,
                  ocr_page_count, excluded_low_confidence, footnotes_json,
                  page_boundaries_json, token_count, jsonl_path,
                  pipeline_version, run_id

    Returns:
        Row ID of inserted/updated document
    """
    cursor = conn.execute(
        """
        INSERT INTO documents (
            report_id, total_pages, included_pages, skipped_pages_json,
            primary_source, embedded_page_count, ocr_page_count, excluded_low_confidence,
            footnotes_json, page_boundaries_json, token_count, jsonl_path,
            pipeline_version, run_id, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(report_id) DO UPDATE SET
            total_pages = excluded.total_pages,
            included_pages = excluded.included_pages,
            skipped_pages_json = excluded.skipped_pages_json,
            primary_source = excluded.primary_source,
            embedded_page_count = excluded.embedded_page_count,
            ocr_page_count = excluded.ocr_page_count,
            excluded_low_confidence = excluded.excluded_low_confidence,
            footnotes_json = excluded.footnotes_json,
            page_boundaries_json = excluded.page_boundaries_json,
            token_count = excluded.token_count,
            jsonl_path = excluded.jsonl_path,
            pipeline_version = excluded.pipeline_version,
            run_id = excluded.run_id,
            created_at = excluded.created_at
        """,
        (
            doc_data["report_id"],
            doc_data["total_pages"],
            doc_data["included_pages"],
            doc_data.get("skipped_pages_json"),
            doc_data["primary_source"],
            doc_data.get("embedded_page_count", 0),
            doc_data.get("ocr_page_count", 0),
            doc_data.get("excluded_low_confidence", 0),
            doc_data.get("footnotes_json"),
            doc_data.get("page_boundaries_json"),
            doc_data.get("token_count"),
            doc_data["jsonl_path"],
            doc_data.get("pipeline_version"),
            doc_data.get("run_id"),
            datetime.now().isoformat(),
        )
    )
    conn.commit()
    return cursor.lastrowid


def get_document(conn: sqlite3.Connection, report_id: str) -> dict | None:
    """Get a document by report_id."""
    cursor = conn.execute(
        "SELECT * FROM documents WHERE report_id = ?",
        (report_id,)
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def get_consolidated_report_ids(conn: sqlite3.Connection) -> set[str]:
    """Get set of report_ids that have been consolidated into documents."""
    cursor = conn.execute("SELECT report_id FROM documents")
    return {row["report_id"] for row in cursor.fetchall()}


def get_reports_needing_documents(conn: sqlite3.Connection) -> list[str]:
    """Get report_ids with pages but no document entry."""
    cursor = conn.execute(
        """
        SELECT DISTINCT p.report_id
        FROM pages p
        LEFT JOIN documents d ON p.report_id = d.report_id
        WHERE d.id IS NULL AND p.status IN ('passed', 'ocr_retry')
        ORDER BY p.report_id
        """
    )
    return [row["report_id"] for row in cursor.fetchall()]


# --- Chunks table ---

def insert_chunk(conn: sqlite3.Connection, chunk_data: dict) -> int:
    """
    Insert or update a chunk.

    Args:
        chunk_data: Dict with chunk fields (see schema)

    Returns:
        Row ID of inserted/updated chunk
    """
    cursor = conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, report_id, chunk_sequence,
            page_start, page_end, page_list_json, char_start, char_end,
            section_name, section_number, section_detection_method,
            token_count, overlap_tokens,
            text_source, page_sources_json, source_quality_json,
            has_footnotes, footnotes_json, quality_flags_json,
            jsonl_path, pipeline_version, run_id, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            chunk_sequence = excluded.chunk_sequence,
            page_start = excluded.page_start,
            page_end = excluded.page_end,
            page_list_json = excluded.page_list_json,
            char_start = excluded.char_start,
            char_end = excluded.char_end,
            section_name = excluded.section_name,
            section_number = excluded.section_number,
            section_detection_method = excluded.section_detection_method,
            token_count = excluded.token_count,
            overlap_tokens = excluded.overlap_tokens,
            text_source = excluded.text_source,
            page_sources_json = excluded.page_sources_json,
            source_quality_json = excluded.source_quality_json,
            has_footnotes = excluded.has_footnotes,
            footnotes_json = excluded.footnotes_json,
            quality_flags_json = excluded.quality_flags_json,
            jsonl_path = excluded.jsonl_path,
            pipeline_version = excluded.pipeline_version,
            run_id = excluded.run_id,
            created_at = excluded.created_at
        """,
        (
            chunk_data["chunk_id"],
            chunk_data["report_id"],
            chunk_data["chunk_sequence"],
            chunk_data["page_start"],
            chunk_data["page_end"],
            chunk_data.get("page_list_json"),
            chunk_data["char_start"],
            chunk_data["char_end"],
            chunk_data.get("section_name"),
            chunk_data.get("section_number"),
            chunk_data.get("section_detection_method"),
            chunk_data["token_count"],
            chunk_data.get("overlap_tokens", 0),
            chunk_data["text_source"],
            chunk_data.get("page_sources_json"),
            chunk_data.get("source_quality_json"),
            chunk_data.get("has_footnotes", 0),
            chunk_data.get("footnotes_json"),
            chunk_data.get("quality_flags_json"),
            chunk_data["jsonl_path"],
            chunk_data.get("pipeline_version"),
            chunk_data.get("run_id"),
            datetime.now().isoformat(),
        )
    )
    conn.commit()
    return cursor.lastrowid


def get_chunks_for_report(conn: sqlite3.Connection, report_id: str) -> list[dict]:
    """Get all chunks for a report, ordered by sequence."""
    cursor = conn.execute(
        """
        SELECT * FROM chunks
        WHERE report_id = ?
        ORDER BY chunk_sequence
        """,
        (report_id,)
    )
    return [dict(row) for row in cursor.fetchall()]


def get_chunked_report_ids(conn: sqlite3.Connection) -> set[str]:
    """Get set of report_ids that have been chunked."""
    cursor = conn.execute("SELECT DISTINCT report_id FROM chunks")
    return {row["report_id"] for row in cursor.fetchall()}


def get_reports_needing_chunking(conn: sqlite3.Connection) -> list[str]:
    """Get report_ids with documents but no chunks."""
    cursor = conn.execute(
        """
        SELECT d.report_id
        FROM documents d
        LEFT JOIN chunks c ON d.report_id = c.report_id
        WHERE c.id IS NULL
        ORDER BY d.report_id
        """
    )
    return [row["report_id"] for row in cursor.fetchall()]


def delete_chunks_for_report(conn: sqlite3.Connection, report_id: str) -> int:
    """Delete all chunks for a report (for reprocessing). Returns count deleted."""
    cursor = conn.execute(
        "DELETE FROM chunks WHERE report_id = ?",
        (report_id,)
    )
    conn.commit()
    return cursor.rowcount


# --- Chunking runs ---

def create_chunking_run(
    conn: sqlite3.Connection,
    run_type: str,
    config_json: str | None = None
) -> int:
    """
    Create a new chunking run.

    Args:
        run_type: 'pages', 'documents', 'chunks', 'full', or 'retry_failed'
        config_json: JSON string of configuration used

    Returns:
        Run ID
    """
    cursor = conn.execute(
        """
        INSERT INTO chunking_runs (run_type, started_at, status, config_json)
        VALUES (?, ?, 'running', ?)
        """,
        (run_type, datetime.now().isoformat(), config_json)
    )
    conn.commit()
    return cursor.lastrowid


def update_chunking_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str | None = None,
    **stats
) -> None:
    """
    Update chunking run status and stats.

    Args:
        run_id: Run ID to update
        status: New status ('running', 'completed', 'failed', 'interrupted')
        **stats: Stats to update (documents_created, chunks_created, etc.)
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
            f"UPDATE chunking_runs SET {', '.join(updates)} WHERE id = ?",
            tuple(values)
        )
        conn.commit()


def get_latest_chunking_run(
    conn: sqlite3.Connection,
    run_type: str | None = None
) -> dict | None:
    """Get the most recent chunking run, optionally filtered by type."""
    query = "SELECT * FROM chunking_runs"
    params = ()

    if run_type:
        query += " WHERE run_type = ?"
        params = (run_type,)

    query += " ORDER BY id DESC LIMIT 1"

    cursor = conn.execute(query, params)
    row = cursor.fetchone()
    return dict(row) if row else None


# --- Chunking errors ---

def log_chunking_error(
    conn: sqlite3.Connection,
    report_id: str,
    error_type: str,
    error_message: str,
    run_id: int | None = None,
    stack_trace: str | None = None,
) -> None:
    """Record a chunking error."""
    conn.execute(
        """
        INSERT INTO chunking_errors
        (run_id, report_id, error_type, error_message, stack_trace, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, report_id, error_type, error_message, stack_trace,
         datetime.now().isoformat())
    )
    conn.commit()


# --- Chunking analytics ---

def get_chunking_stats(conn: sqlite3.Connection) -> dict:
    """Get comprehensive chunking statistics."""
    stats = {}

    # Document stats
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total_documents,
            SUM(total_pages) as total_pages,
            SUM(included_pages) as included_pages,
            SUM(embedded_page_count) as embedded_pages,
            SUM(ocr_page_count) as ocr_pages,
            SUM(excluded_low_confidence) as excluded_low_confidence,
            AVG(token_count) as avg_doc_tokens
        FROM documents
        """
    )
    row = cursor.fetchone()
    stats["documents"] = dict(row) if row else {}

    # Chunk stats
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total_chunks,
            AVG(token_count) as avg_tokens,
            MIN(token_count) as min_tokens,
            MAX(token_count) as max_tokens,
            AVG(overlap_tokens) as avg_overlap,
            SUM(has_footnotes) as chunks_with_footnotes
        FROM chunks
        """
    )
    row = cursor.fetchone()
    stats["chunks"] = dict(row) if row else {}

    # Token distribution
    cursor = conn.execute(
        """
        SELECT
            CASE
                WHEN token_count < 500 THEN 'under_500'
                WHEN token_count BETWEEN 500 AND 600 THEN '500-600'
                WHEN token_count BETWEEN 600 AND 700 THEN '600-700'
                ELSE 'over_700'
            END as bucket,
            COUNT(*) as count
        FROM chunks
        GROUP BY bucket
        """
    )
    stats["token_distribution"] = {row["bucket"]: row["count"] for row in cursor.fetchall()}

    # Section detection success
    cursor = conn.execute(
        """
        SELECT
            section_detection_method,
            COUNT(*) as count
        FROM chunks
        GROUP BY section_detection_method
        """
    )
    stats["section_detection"] = {
        row["section_detection_method"]: row["count"]
        for row in cursor.fetchall()
    }

    # Source distribution
    cursor = conn.execute(
        """
        SELECT text_source, COUNT(*) as count
        FROM chunks
        GROUP BY text_source
        """
    )
    stats["source_distribution"] = {
        row["text_source"]: row["count"]
        for row in cursor.fetchall()
    }

    # Error count
    cursor = conn.execute("SELECT COUNT(*) as count FROM chunking_errors")
    stats["error_count"] = cursor.fetchone()["count"]

    return stats
