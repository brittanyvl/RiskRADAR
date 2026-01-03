"""
collection/scrape.py
--------------------
NTSB Aviation PDF scraping pipeline.

Scrapes report metadata and downloads PDFs from:
https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx

Usage:
    from collection.scrape import scrape_all_reports
    scrape_all_reports(resume=True, limit=None)
"""
import logging
import hashlib
import re
from pathlib import Path
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    StaleElementReferenceException,
)

from scraper.config import BrowserConfig
from scraper.browser import chrome
from scraper.actions import go_to, click
from scraper.waits import rate_limit, until
from scraper.download import wait_for_new_download, move_and_rename

from sqlite.connection import init_db
from sqlite.queries import (
    is_already_downloaded,
    insert_report,
    update_report_status,
    get_resume_point,
    update_resume_point,
    reset_resume_point,
    log_error,
    get_scrape_stats,
)

from riskradar.config import NAS_PATH, DB_PATH, LOG_DIR, NTSB_REPORTS_URL


# --- NTSB Page Locators ---
# Verified against live DOM on 2025-01-02

# Aviation mode button (not a dropdown - it's a filter button)
AVIATION_BUTTON = (By.ID, "btnaviation")

# Report container and blocks
REPORT_CONTAINER = (By.ID, "investigation_reports")
REPORT_BLOCKS = (By.CSS_SELECTOR, "#investigation_reports > div.block")

# Inside each report block:
BLOCK_TITLE = (By.CSS_SELECTOR, "div.desc > p:first-child > a")
BLOCK_LOCATION = (By.CSS_SELECTOR, "p.location")
BLOCK_DATA = (By.CSS_SELECTOR, "p.data")  # Contains accident_date and report_date
BLOCK_REPORT_NUM = (By.CSS_SELECTOR, "p.report")
BLOCK_PDF_LINK = (By.CSS_SELECTOR, "div.download a[href$='.pdf']")

# Pagination
NEXT_BUTTON = (By.ID, "NextLinkButton")
NEXT_BUTTON_ITEM = (By.ID, "NextLinkItem")  # Parent <li> - check for 'disabled' class
PREV_BUTTON = (By.ID, "PreviousLinkButton")
PAGE_NUMBER = (By.ID, "PageNumberItem")


# --- Utility Functions ---

def setup_logging() -> logging.Logger:
    """Configure logging to console and file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = LOG_DIR / f"scrape_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # Reduce noise from selenium/urllib3
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_date(date_str: str) -> str:
    """Convert MM/DD/YYYY to ISO format YYYY-MM-DD."""
    if not date_str:
        return ""
    match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str.strip())
    if match:
        month, day, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return date_str


# --- Page Parsing ---

def wait_for_page_load(driver: WebDriver, timeout: int = 30) -> None:
    """Wait for report blocks to be loaded."""
    from selenium.webdriver.support import expected_conditions as EC
    try:
        until(driver, EC.presence_of_element_located(REPORT_CONTAINER), timeout)
        # Also wait a moment for JavaScript to populate blocks
        import time
        time.sleep(1)
    except TimeoutException:
        pass  # May not find container on error


def extract_reports_from_page(driver: WebDriver, logger: logging.Logger) -> list[dict]:
    """
    Extract metadata for all reports on current page.

    Returns list of dicts with keys:
        filename, title, location, accident_date, report_date,
        report_number, pdf_url, pdf_element
    """
    reports = []

    # Find all report blocks
    try:
        blocks = driver.find_elements(*REPORT_BLOCKS)
    except NoSuchElementException:
        logger.warning("No report blocks found on page")
        return reports

    logger.debug(f"Found {len(blocks)} report blocks")

    for block in blocks:
        try:
            # Find PDF link in this block
            try:
                pdf_link = block.find_element(*BLOCK_PDF_LINK)
                pdf_url = pdf_link.get_attribute("href")
                if not pdf_url or not pdf_url.endswith(".pdf"):
                    continue
                filename = pdf_url.split("/")[-1]
            except NoSuchElementException:
                logger.debug("Block has no PDF link, skipping")
                continue

            # Extract title
            title = ""
            try:
                title_elem = block.find_element(*BLOCK_TITLE)
                title = title_elem.text.strip()
            except NoSuchElementException:
                pass

            # Extract location
            location = ""
            try:
                loc_elem = block.find_element(*BLOCK_LOCATION)
                location = loc_elem.text.strip()
            except NoSuchElementException:
                pass

            # Extract dates from p.data element
            # Format: "Accident Date: M/D/YYYY\nReport Date: M/D/YYYY"
            accident_date = ""
            report_date = ""
            try:
                data_elem = block.find_element(*BLOCK_DATA)
                data_text = data_elem.text
                # Parse accident date
                acc_match = re.search(r"Accident Date:\s*(\d{1,2}/\d{1,2}/\d{4})", data_text)
                if acc_match:
                    accident_date = parse_date(acc_match.group(1))
                # Parse report date
                rep_match = re.search(r"Report Date:\s*(\d{1,2}/\d{1,2}/\d{4})", data_text)
                if rep_match:
                    report_date = parse_date(rep_match.group(1))
            except NoSuchElementException:
                pass

            # Extract report number
            # Format: "Report Number: AIR-25-07"
            report_number = ""
            try:
                report_elem = block.find_element(*BLOCK_REPORT_NUM)
                report_text = report_elem.text
                num_match = re.search(r"Report Number:\s*(.+)", report_text)
                if num_match:
                    report_number = num_match.group(1).strip()
            except NoSuchElementException:
                pass

            report = {
                "filename": filename,
                "title": title,
                "location": location,
                "accident_date": accident_date,
                "report_date": report_date,
                "report_number": report_number,
                "pdf_url": pdf_url,
                "pdf_element": pdf_link,
            }
            reports.append(report)

        except StaleElementReferenceException:
            logger.warning("Stale element, skipping block")
            continue
        except Exception as e:
            logger.warning(f"Error parsing report block: {e}")
            continue

    logger.info(f"Found {len(reports)} reports on page")
    return reports


def has_next_page(driver: WebDriver) -> bool:
    """Check if Next button exists and is enabled."""
    try:
        # Check the parent <li> element for 'disabled' class
        next_item = driver.find_element(*NEXT_BUTTON_ITEM)
        classes = next_item.get_attribute("class") or ""
        return "disabled" not in classes
    except NoSuchElementException:
        return False


# --- Download Logic ---

def download_report(
    driver: WebDriver,
    config: BrowserConfig,
    report: dict,
    logger: logging.Logger,
    max_retries: int = 3,
) -> tuple[bool, str | None, str | None]:
    """
    Download a single PDF with retry logic.

    Uses direct URL navigation to avoid stale element issues.

    Returns:
        (success, local_path, sha256) tuple
    """
    filename = report["filename"]
    pdf_url = report["pdf_url"]

    for attempt in range(max_retries):
        try:
            # Navigate directly to PDF URL to trigger download
            # Chrome will download PDFs instead of displaying them due to our preferences
            driver.get(pdf_url)

            # Wait for download to complete
            downloaded = wait_for_new_download(config)

            # Move to NAS
            final_path = move_and_rename(downloaded, NAS_PATH, filename)

            # Compute hash for integrity check
            sha256 = compute_sha256(final_path)

            logger.info(f"Downloaded: {filename} -> {final_path}")
            return True, str(final_path), sha256

        except TimeoutError:
            logger.warning(f"Download timeout for {filename} (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e} (attempt {attempt + 1}/{max_retries})")

        # Wait before retry
        if attempt < max_retries - 1:
            rate_limit(5.0)  # Longer delay before retry

    return False, None, None


# --- Main Scraping Logic ---

def scrape_all_reports(
    resume: bool = True,
    fresh: bool = False,
    limit: int | None = None,
    retry_failed: bool = False,
) -> dict:
    """
    Scrape all NTSB aviation PDFs.

    Args:
        resume: Continue from last saved progress (default True)
        fresh: Start from beginning, ignore previous progress
        limit: Maximum number of PDFs to download (for testing)
        retry_failed: Only retry previously failed downloads

    Returns:
        Dict with statistics: downloaded, skipped, failed, total
    """
    logger = setup_logging()
    logger.info("Starting NTSB PDF scrape")
    logger.info(f"NAS path: {NAS_PATH}")
    logger.info(f"Database: {DB_PATH}")

    # Initialize database
    conn = init_db(DB_PATH)

    # Handle fresh start
    if fresh:
        reset_resume_point(conn)
        logger.info("Fresh start - reset progress")

    # Get starting point
    start_page, start_idx = (0, 0)
    if resume and not fresh:
        start_page, start_idx = get_resume_point(conn)
        if start_page > 0 or start_idx > 0:
            logger.info(f"Resuming from page {start_page}, report {start_idx}")

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    config = BrowserConfig()

    # Progress tracking
    TOTAL_REPORTS = 781  # Known total from NTSB site
    TOTAL_PAGES = 79     # ~10 per page

    with chrome(config) as driver:
        # Navigate to reports page
        logger.info(f"Navigating to {NTSB_REPORTS_URL}")
        go_to(driver, NTSB_REPORTS_URL)
        rate_limit(config.request_delay_sec)

        # Click Aviation filter button
        logger.info("Clicking Aviation filter button")
        try:
            click(driver, AVIATION_BUTTON)
            rate_limit(config.page_delay_sec)
            wait_for_page_load(driver)
        except Exception as e:
            logger.warning(f"Could not click Aviation button: {e}")
            # Continue anyway - might already be filtered

        # Navigate to resume page
        current_page = 0
        while current_page < start_page:
            if not has_next_page(driver):
                logger.error(f"Cannot navigate to page {start_page}")
                break
            click(driver, NEXT_BUTTON)
            rate_limit(config.page_delay_sec)
            current_page += 1
            logger.info(f"Navigated to page {current_page}")

        # Main scraping loop
        while True:
            logger.info(f"Processing page {current_page + 1}")
            wait_for_page_load(driver)

            reports = extract_reports_from_page(driver, logger)

            # Skip to resume point on first page
            report_start = start_idx if current_page == start_page else 0

            for idx, report in enumerate(reports[report_start:], start=report_start):
                filename = report["filename"]

                # Check limit
                if limit and stats["downloaded"] >= limit:
                    logger.info(f"Reached limit of {limit} downloads")
                    conn.close()
                    return stats

                # Skip if already downloaded
                if is_already_downloaded(conn, filename):
                    logger.debug(f"Skipping {filename} - already downloaded")
                    stats["skipped"] += 1
                    continue

                # Insert metadata
                insert_report(conn, report)
                update_report_status(conn, filename, "downloading")

                # Download PDF (this navigates away from reports page)
                success, local_path, sha256 = download_report(
                    driver, config, report, logger
                )

                if success:
                    update_report_status(conn, filename, "completed", local_path, sha256)
                    stats["downloaded"] += 1
                else:
                    update_report_status(conn, filename, "failed")
                    log_error(conn, filename, "download_failed", "Max retries exceeded")
                    stats["failed"] += 1

                # Save progress
                update_resume_point(conn, current_page, idx + 1)

                # Log progress
                total_processed = stats["downloaded"] + stats["skipped"] + stats["failed"]
                pct = (total_processed / TOTAL_REPORTS) * 100
                logger.info(
                    f"Progress: Page {current_page + 1}/{TOTAL_PAGES} | "
                    f"Processed {total_processed}/{TOTAL_REPORTS} ({pct:.1f}%) | "
                    f"Downloaded: {stats['downloaded']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}"
                )

                # Rate limit between downloads
                rate_limit(config.download_delay_sec)

            # Navigate back to reports page for pagination
            logger.info("Navigating back to reports page")
            go_to(driver, NTSB_REPORTS_URL)
            rate_limit(config.request_delay_sec)

            # Re-select Aviation filter
            try:
                click(driver, AVIATION_BUTTON)
                rate_limit(config.page_delay_sec)
                wait_for_page_load(driver)
            except Exception as e:
                logger.warning(f"Could not click Aviation button: {e}")

            # Navigate to current page + 1 (next page)
            for _ in range(current_page + 1):
                if not has_next_page(driver):
                    logger.info("Reached last page")
                    conn.close()
                    return stats
                click(driver, NEXT_BUTTON)
                rate_limit(config.page_delay_sec)

            current_page += 1
            update_resume_point(conn, current_page, 0)
            wait_for_page_load(driver)

    # Final stats
    db_stats = get_scrape_stats(conn)
    logger.info(f"Scrape complete: {stats}")
    logger.info(f"Database stats: {db_stats}")

    conn.close()
    return stats


if __name__ == "__main__":
    # Quick test
    scrape_all_reports(limit=5)
