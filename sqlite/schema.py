"""
sqlite/schema.py
----------------
Database schema definitions for RiskRADAR.

Only includes tables needed for the current phase.
Additional tables will be added as we implement later phases.
"""

SCHEMA_VERSION = 1

# Reports table - stores metadata from NTSB scraping
REPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    title TEXT,
    location TEXT,
    accident_date TEXT,
    report_date TEXT,
    report_number TEXT,
    pdf_url TEXT UNIQUE,
    local_path TEXT,
    sha256 TEXT,
    downloaded_at TEXT,
    status TEXT DEFAULT 'pending'
);
"""

# Resume tracking - allows scraper to pick up where it left off
SCRAPE_PROGRESS_TABLE = """
CREATE TABLE IF NOT EXISTS scrape_progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_page INTEGER DEFAULT 0,
    last_report_index INTEGER DEFAULT 0,
    updated_at TEXT
);
"""

# Error logging - track failures for retry/debugging
SCRAPE_ERRORS_TABLE = """
CREATE TABLE IF NOT EXISTS scrape_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_filename TEXT,
    error_type TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TEXT
);
"""

# Indexes for common queries
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_reports_filename ON reports(filename);",
    "CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);",
]

# All tables for Phase 1
ALL_TABLES = [
    REPORTS_TABLE,
    SCRAPE_PROGRESS_TABLE,
    SCRAPE_ERRORS_TABLE,
]
