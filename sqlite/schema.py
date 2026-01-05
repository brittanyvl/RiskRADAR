"""
sqlite/schema.py
----------------
Database schema definitions for RiskRADAR.

Only includes tables needed for the current phase.
Additional tables will be added as we implement later phases.

SCHEMA_VERSION history:
- v1: Phase 1 - Scraping (reports, scrape_progress, scrape_errors)
- v2: Phase 3 - Text Extraction (pages, extraction_runs, extraction_errors)
"""

SCHEMA_VERSION = 2

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

# Pages table - tracks extracted pages with JSON file pointers
PAGES_TABLE = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,

    -- File paths (relative to extraction/)
    json_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending_qc', 'passed', 'failed', 'ocr_retry', 'error')),

    -- Extraction metadata
    extraction_method TEXT NOT NULL CHECK(extraction_method IN ('embedded', 'ocr')),
    extraction_pass TEXT NOT NULL CHECK(extraction_pass IN ('initial', 'ocr_retry')),
    extraction_time_ms INTEGER,
    extracted_at TEXT,

    -- Quality metrics (for fast querying without opening JSON)
    char_count INTEGER,
    alphabetic_ratio REAL,
    garbage_ratio REAL,
    word_count INTEGER,
    passes_threshold INTEGER CHECK(passes_threshold IN (0, 1)),

    -- OCR confidence (NULL for embedded extractions)
    mean_ocr_confidence REAL,
    low_confidence_word_count INTEGER,

    FOREIGN KEY (report_id) REFERENCES reports(filename) ON DELETE CASCADE,
    UNIQUE (report_id, page_number)
);
"""

# Extraction runs - tracks pipeline executions
EXTRACTION_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS extraction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL CHECK(run_type IN ('initial', 'ocr_retry', 'full_reprocess')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),

    -- Stats
    total_reports INTEGER DEFAULT 0,
    total_pages INTEGER DEFAULT 0,
    passed_pages INTEGER DEFAULT 0,
    failed_pages INTEGER DEFAULT 0,
    error_pages INTEGER DEFAULT 0,

    -- Resume info
    last_report_id TEXT,
    last_page_number INTEGER,

    -- Config snapshot
    config_json TEXT
);
"""

# Extraction errors - detailed error logging
EXTRACTION_ERRORS_TABLE = """
CREATE TABLE IF NOT EXISTS extraction_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    report_id TEXT NOT NULL,
    page_number INTEGER,
    error_type TEXT NOT NULL,
    error_message TEXT,
    stack_trace TEXT,
    created_at TEXT NOT NULL,

    FOREIGN KEY (run_id) REFERENCES extraction_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (report_id) REFERENCES reports(filename) ON DELETE CASCADE
);
"""

# Indexes for common queries
PHASE1_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_reports_filename ON reports(filename);",
    "CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);",
]

PHASE3_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pages_report ON pages(report_id);",
    "CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(status);",
    "CREATE INDEX IF NOT EXISTS idx_pages_method ON pages(extraction_method);",
    "CREATE INDEX IF NOT EXISTS idx_pages_quality ON pages(passes_threshold);",
    "CREATE INDEX IF NOT EXISTS idx_runs_status ON extraction_runs(status);",
    "CREATE INDEX IF NOT EXISTS idx_errors_run ON extraction_errors(run_id);",
]

# All indexes
INDEXES = PHASE1_INDEXES + PHASE3_INDEXES

# All tables for Phase 1
PHASE1_TABLES = [
    REPORTS_TABLE,
    SCRAPE_PROGRESS_TABLE,
    SCRAPE_ERRORS_TABLE,
]

# All tables for Phase 3
PHASE3_TABLES = [
    PAGES_TABLE,
    EXTRACTION_RUNS_TABLE,
    EXTRACTION_ERRORS_TABLE,
]

# All tables combined
ALL_TABLES = PHASE1_TABLES + PHASE3_TABLES
