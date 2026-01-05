"""
processing/extract.py
---------------------
Main extraction pipeline orchestration for Phase 3.

Multi-pass workflow:
- Pass 1: Initial extraction (embedded text) → quality gate → passed/failed
- Pass 2: OCR retry on failed pages → ocr_retry
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from .pdf_reader import extract_page_to_json as extract_embedded, get_page_count
from .ocr import ocr_page_to_json as extract_ocr
from sqlite.connection import init_db
from sqlite.queries import (
    insert_page,
    get_extracted_pages,
    get_failed_pages,
    create_extraction_run,
    update_extraction_run,
    log_extraction_error,
)
from riskradar.config import DB_PATH, NAS_PATH, PROJECT_ROOT


# Extraction directories
EXTRACTION_ROOT = PROJECT_ROOT / "extraction"
JSON_DATA_DIR = EXTRACTION_ROOT / "json_data"
PASSED_DIR = JSON_DATA_DIR / "passed"
FAILED_DIR = JSON_DATA_DIR / "failed"
OCR_RETRY_DIR = JSON_DATA_DIR / "ocr_retry"
TEMP_DIR = EXTRACTION_ROOT / "temp"


def setup_logging(log_name: str = "extract") -> logging.Logger:
    """
    Configure logging for extraction.

    Args:
        log_name: Log file prefix

    Returns:
        Configured logger
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = log_dir / f"{log_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,  # Override any existing config
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def run_quality_gate(
    temp_dir: Path,
    passed_dir: Path,
    failed_dir: Path,
    conn,
    logger: logging.Logger,
) -> dict:
    """
    Run quality gate: move temp files to passed/ or failed/ based on quality metrics.

    This function:
    1. Reads all JSON files in temp/
    2. Checks quality_metrics.passes_threshold
    3. Moves to passed/ if True, failed/ if False
    4. Updates pages table status
    5. Returns stats

    Args:
        temp_dir: Temporary extraction directory
        passed_dir: Destination for passed pages
        failed_dir: Destination for failed pages
        conn: Database connection
        logger: Logger instance

    Returns:
        Dict with keys: passed_count, failed_count, error_count
    """
    stats = {"passed_count": 0, "failed_count": 0, "error_count": 0}

    # Find all JSON files in temp/
    json_files = list(temp_dir.rglob("*.json"))

    logger.info(f"Running quality gate on {len(json_files)} pages...")

    for json_path in json_files:
        try:
            # Read JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            report_id = data["report_id"]
            page_number = data["page_number"]
            passes_threshold = data["quality_metrics"]["passes_threshold"]

            # Determine destination
            if passes_threshold:
                dest_dir = passed_dir / report_id
                new_status = "passed"
                stats["passed_count"] += 1
            else:
                dest_dir = failed_dir / report_id
                new_status = "failed"
                stats["failed_count"] += 1

            # Create destination directory
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Move file
            dest_path = dest_dir / json_path.name
            shutil.move(str(json_path), str(dest_path))

            # Update database status and json_path
            relative_path = dest_path.relative_to(EXTRACTION_ROOT)
            page_data = {
                "report_id": report_id,
                "page_number": page_number,
                "json_path": str(relative_path),
                "status": new_status,
                "extraction_method": data["extraction_method"],
                "extraction_pass": data["extraction_pass"],
                "extraction_time_ms": data["metadata"]["extraction_time_ms"],
                "char_count": data["quality_metrics"]["char_count"],
                "alphabetic_ratio": data["quality_metrics"]["alphabetic_ratio"],
                "garbage_ratio": data["quality_metrics"]["garbage_ratio"],
                "word_count": data["quality_metrics"]["word_count"],
                "passes_threshold": 1 if passes_threshold else 0,
            }
            insert_page(conn, page_data)

            logger.debug(f"{report_id} p{page_number}: {new_status}")

        except Exception as e:
            logger.error(f"Quality gate error for {json_path}: {e}")
            stats["error_count"] += 1

    # Cleanup temp directories
    for report_dir in temp_dir.iterdir():
        if report_dir.is_dir():
            try:
                report_dir.rmdir()
            except OSError:
                pass  # Directory not empty, that's OK

    logger.info(
        f"Quality gate complete: {stats['passed_count']} passed, "
        f"{stats['failed_count']} failed, {stats['error_count']} errors"
    )

    return stats


def run_initial_extraction(
    limit: Optional[int] = None,
    resume: bool = True,
) -> dict:
    """
    Pass 1: Initial extraction using embedded text.

    Workflow:
    1. Get list of completed downloads
    2. For each report:
       - Extract embedded text from each page → temp/
       - Save as JSON with quality metrics
    3. Run quality gate:
       - Move passed pages → passed/
       - Move failed pages → failed/
    4. Update extraction_runs table
    5. Return stats

    Args:
        limit: Process only N reports (for testing)
        resume: Continue from last saved position

    Returns:
        Stats dict with keys:
        - reports_processed, pages_extracted, passed_count, failed_count, error_count
    """
    logger = setup_logging("extract_initial")
    logger.info("=" * 60)
    logger.info("Phase 3 - Pass 1: Initial Extraction (Embedded Text)")
    logger.info("=" * 60)

    # Initialize database
    conn = init_db(DB_PATH)

    # Create extraction run
    config_json = json.dumps({
        "limit": limit,
        "resume": resume,
        "pass": "initial",
    })
    run_id = create_extraction_run(conn, "initial", config_json)
    logger.info(f"Created extraction run #{run_id}")

    # Get list of completed downloads
    cursor = conn.execute(
        "SELECT filename, local_path FROM reports WHERE status = 'completed' ORDER BY filename"
    )
    reports = [(row["filename"], row["local_path"]) for row in cursor.fetchall()]

    logger.info(f"Found {len(reports)} completed downloads")

    # Stats
    stats = {
        "reports_processed": 0,
        "pages_extracted": 0,
        "passed_count": 0,
        "failed_count": 0,
        "error_count": 0,
    }

    # Process each report
    for filename, local_path in reports:
        # Check limit
        if limit and stats["reports_processed"] >= limit:
            logger.info(f"Reached limit of {limit} reports")
            break

        pdf_path = Path(local_path)
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {local_path}, skipping")
            continue

        logger.info(f"Processing {filename}...")

        try:
            # Get page count
            total_pages = get_page_count(pdf_path)

            # Get already extracted pages (if resuming)
            extracted_pages = set()
            if resume:
                extracted_pages = get_extracted_pages(conn, filename)
                if extracted_pages:
                    logger.info(f"  Resuming: {len(extracted_pages)} pages already extracted")

            # Extract each page
            for page_num in range(total_pages):
                if page_num in extracted_pages:
                    logger.debug(f"  Page {page_num}: already extracted, skipping")
                    continue

                try:
                    # Extract to temp/
                    data, json_path = extract_embedded(pdf_path, filename, page_num, TEMP_DIR)

                    stats["pages_extracted"] += 1

                    if stats["pages_extracted"] % 100 == 0:
                        logger.info(f"  Progress: {stats['pages_extracted']} pages extracted")

                except Exception as e:
                    logger.error(f"  Page {page_num}: extraction failed: {e}")
                    log_extraction_error(conn, filename, "extraction_error", str(e), run_id, page_num)
                    stats["error_count"] += 1

            stats["reports_processed"] += 1

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            log_extraction_error(conn, filename, "report_error", str(e), run_id)
            stats["error_count"] += 1

    logger.info(
        f"\nExtraction complete: {stats['reports_processed']} reports, "
        f"{stats['pages_extracted']} pages extracted"
    )

    # Run quality gate
    logger.info("\nRunning quality gate...")
    gate_stats = run_quality_gate(TEMP_DIR, PASSED_DIR, FAILED_DIR, conn, logger)

    stats["passed_count"] = gate_stats["passed_count"]
    stats["failed_count"] = gate_stats["failed_count"]
    stats["error_count"] += gate_stats["error_count"]

    # Update extraction run
    update_extraction_run(
        conn,
        run_id,
        status="completed",
        total_reports=stats["reports_processed"],
        total_pages=stats["pages_extracted"],
        passed_pages=stats["passed_count"],
        failed_pages=stats["failed_count"],
        error_pages=stats["error_count"],
    )

    logger.info("\n" + "=" * 60)
    logger.info("Pass 1 Complete!")
    logger.info(f"  Reports processed: {stats['reports_processed']}")
    logger.info(f"  Pages extracted:   {stats['pages_extracted']}")
    logger.info(f"  Passed:            {stats['passed_count']}")
    logger.info(f"  Failed:            {stats['failed_count']}")
    logger.info(f"  Errors:            {stats['error_count']}")
    logger.info("=" * 60)

    conn.close()
    return stats


def run_ocr_retry(
    limit: Optional[int] = None,
) -> dict:
    """
    Pass 2: OCR retry on failed pages.

    Workflow:
    1. Get list of failed pages from database
    2. For each failed page:
       - Run OCR with confidence scoring → ocr_retry/
       - Update pages table with OCR results
    3. Update extraction_runs table
    4. Return stats

    Args:
        limit: Process only N pages (for testing)

    Returns:
        Stats dict with keys:
        - pages_processed, ocr_success_count, low_confidence_count, error_count
    """
    logger = setup_logging("extract_ocr_retry")
    logger.info("=" * 60)
    logger.info("Phase 3 - Pass 2: OCR Retry (Failed Pages)")
    logger.info("=" * 60)

    # Initialize database
    conn = init_db(DB_PATH)

    # Create extraction run
    config_json = json.dumps({
        "limit": limit,
        "pass": "ocr_retry",
    })
    run_id = create_extraction_run(conn, "ocr_retry", config_json)
    logger.info(f"Created extraction run #{run_id}")

    # Get failed pages
    failed_pages = get_failed_pages(conn)
    logger.info(f"Found {len(failed_pages)} failed pages needing OCR")

    if not failed_pages:
        logger.info("No pages to process!")
        update_extraction_run(conn, run_id, status="completed")
        conn.close()
        return {
            "pages_processed": 0,
            "ocr_success_count": 0,
            "low_confidence_count": 0,
            "error_count": 0,
        }

    # Stats
    stats = {
        "pages_processed": 0,
        "ocr_success_count": 0,
        "low_confidence_count": 0,
        "error_count": 0,
    }

    # Process each failed page
    for page_info in failed_pages:
        # Check limit
        if limit and stats["pages_processed"] >= limit:
            logger.info(f"Reached limit of {limit} pages")
            break

        report_id = page_info["report_id"]
        page_number = page_info["page_number"]
        pdf_path = Path(page_info["pdf_path"])

        logger.info(f"OCR {report_id} page {page_number}...")

        try:
            # Run OCR
            data, json_path = extract_ocr(pdf_path, report_id, page_number, OCR_RETRY_DIR)

            # Update pages table
            relative_path = json_path.relative_to(EXTRACTION_ROOT)
            mean_conf = data["ocr_confidence"]["mean_confidence"]
            low_conf_words = data["ocr_confidence"]["low_confidence_words"]

            page_data = {
                "report_id": report_id,
                "page_number": page_number,
                "json_path": str(relative_path),
                "status": "ocr_retry",
                "extraction_method": "ocr",
                "extraction_pass": "ocr_retry",
                "extraction_time_ms": data["metadata"]["extraction_time_ms"],
                "char_count": data["quality_metrics"]["char_count"],
                "alphabetic_ratio": data["quality_metrics"]["alphabetic_ratio"],
                "garbage_ratio": data["quality_metrics"]["garbage_ratio"],
                "word_count": data["quality_metrics"]["word_count"],
                "passes_threshold": 1 if data["quality_metrics"]["passes_threshold"] else 0,
                "mean_ocr_confidence": mean_conf,
                "low_confidence_word_count": low_conf_words,
            }
            insert_page(conn, page_data)

            stats["pages_processed"] += 1
            stats["ocr_success_count"] += 1

            # Check if low confidence
            if mean_conf < 60.0:
                stats["low_confidence_count"] += 1
                logger.warning(f"  Low confidence: {mean_conf:.1f}% (flagged for review)")
            else:
                logger.info(f"  OCR confidence: {mean_conf:.1f}%")

        except Exception as e:
            logger.error(f"  OCR failed: {e}")
            log_extraction_error(conn, report_id, "ocr_error", str(e), run_id, page_number)
            stats["error_count"] += 1

    # Update extraction run
    update_extraction_run(
        conn,
        run_id,
        status="completed",
        total_pages=stats["pages_processed"],
        passed_pages=stats["ocr_success_count"],
        failed_pages=stats["low_confidence_count"],
        error_pages=stats["error_count"],
    )

    logger.info("\n" + "=" * 60)
    logger.info("Pass 2 Complete!")
    logger.info(f"  Pages processed:     {stats['pages_processed']}")
    logger.info(f"  OCR successful:      {stats['ocr_success_count']}")
    logger.info(f"  Low confidence:      {stats['low_confidence_count']}")
    logger.info(f"  Errors:              {stats['error_count']}")
    logger.info("=" * 60)

    conn.close()
    return stats


def run_full_pipeline(
    limit: Optional[int] = None,
    resume: bool = True,
) -> dict:
    """
    Run complete extraction pipeline: Pass 1 → Pass 2.

    Args:
        limit: Process only N reports (for testing)
        resume: Continue from last saved position

    Returns:
        Combined stats from both passes
    """
    logger = setup_logging("extract_full")
    logger.info("=" * 60)
    logger.info("Phase 3 - Full Pipeline: Initial + OCR Retry")
    logger.info("=" * 60)

    # Pass 1: Initial extraction
    stats1 = run_initial_extraction(limit, resume)

    # Pass 2: OCR retry
    stats2 = run_ocr_retry()

    # Combined stats
    combined_stats = {
        "pass1_reports": stats1["reports_processed"],
        "pass1_pages": stats1["pages_extracted"],
        "pass1_passed": stats1["passed_count"],
        "pass1_failed": stats1["failed_count"],
        "pass2_ocr": stats2["ocr_success_count"],
        "pass2_low_conf": stats2["low_confidence_count"],
        "total_errors": stats1["error_count"] + stats2["error_count"],
    }

    logger.info("\n" + "=" * 60)
    logger.info("Full Pipeline Complete!")
    logger.info(f"  Pass 1 - Reports processed: {combined_stats['pass1_reports']}")
    logger.info(f"  Pass 1 - Pages extracted:   {combined_stats['pass1_pages']}")
    logger.info(f"  Pass 1 - Passed:            {combined_stats['pass1_passed']}")
    logger.info(f"  Pass 1 - Failed:            {combined_stats['pass1_failed']}")
    logger.info(f"  Pass 2 - OCR'd:             {combined_stats['pass2_ocr']}")
    logger.info(f"  Pass 2 - Low confidence:    {combined_stats['pass2_low_conf']}")
    logger.info(f"  Total errors:               {combined_stats['total_errors']}")
    logger.info("=" * 60)

    return combined_stats


if __name__ == "__main__":
    # Test with a small limit
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

        if command == "initial":
            run_initial_extraction(limit=limit)
        elif command == "ocr":
            run_ocr_retry(limit=limit)
        elif command == "all":
            run_full_pipeline(limit=limit)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python extract.py [initial|ocr|all] [limit]")
    else:
        print("Usage: python extract.py [initial|ocr|all] [limit]")
        print("Examples:")
        print("  python extract.py initial 5    # Test initial extraction on 5 reports")
        print("  python extract.py ocr 10       # Test OCR on 10 failed pages")
        print("  python extract.py all 3        # Full pipeline on 3 reports")
