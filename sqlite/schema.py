"""
sqlite/schema.py
----------------
Database schema definitions for RiskRADAR.

Only includes tables needed for the current phase.
Additional tables will be added as we implement later phases.

SCHEMA_VERSION history:
- v1: Phase 1 - Scraping (reports, scrape_progress, scrape_errors)
- v2: Phase 3 - Text Extraction (pages, extraction_runs, extraction_errors)
- v3: Phase 4 - Chunking (documents, chunks, chunking_runs, chunking_errors)
"""

SCHEMA_VERSION = 3

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

# ============================================================================
# Phase 4: Chunking
# ============================================================================

# Documents table - consolidated full-text documents from pages
DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT UNIQUE NOT NULL,

    -- Text content (not stored in DB, only in JSONL - use jsonl_path)
    -- full_text is intentionally NOT stored here to keep DB size small

    -- Page statistics
    total_pages INTEGER NOT NULL,
    included_pages INTEGER NOT NULL,
    skipped_pages_json TEXT,           -- JSON array of skipped page numbers (TOC, etc.)

    -- Source tracking
    primary_source TEXT NOT NULL CHECK(primary_source IN ('embedded', 'ocr', 'mixed')),
    embedded_page_count INTEGER DEFAULT 0,
    ocr_page_count INTEGER DEFAULT 0,
    excluded_low_confidence INTEGER DEFAULT 0,

    -- Footnotes
    footnotes_json TEXT,               -- JSON array: [{"marker": "1/", "text": "..."}]

    -- Page boundaries for chunk-to-page mapping
    page_boundaries_json TEXT,         -- JSON array: [[start, end], ...]

    -- Token count (using tiktoken cl100k)
    token_count INTEGER,

    -- File reference
    jsonl_path TEXT NOT NULL,          -- Relative path to document JSONL

    -- Metadata
    pipeline_version TEXT,
    run_id INTEGER,
    created_at TEXT NOT NULL,

    FOREIGN KEY (report_id) REFERENCES reports(filename) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES chunking_runs(id) ON DELETE SET NULL
);
"""

# Chunks table - individual text chunks for vector search
CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT UNIQUE NOT NULL,     -- Format: {report_id}_chunk_{seq:04d}
    report_id TEXT NOT NULL,
    chunk_sequence INTEGER NOT NULL,   -- 0-indexed order within document

    -- Position tracking
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    page_list_json TEXT,               -- JSON array: [5, 6, 7] for precise tracking
    char_start INTEGER NOT NULL,       -- Character offset in full document text
    char_end INTEGER NOT NULL,

    -- Section info
    section_name TEXT,                 -- Detected section name (e.g., "SYNOPSIS")
    section_number TEXT,               -- Section number (e.g., "1.8")
    section_detection_method TEXT CHECK(section_detection_method IN
        ('pattern_match', 'paragraph_fallback', 'no_structure')),

    -- Content (not stored in DB - use jsonl_path)
    -- chunk_text is intentionally NOT stored here to keep DB size small
    token_count INTEGER NOT NULL,
    overlap_tokens INTEGER DEFAULT 0,  -- Tokens shared with previous chunk

    -- Source lineage
    text_source TEXT NOT NULL CHECK(text_source IN ('embedded', 'ocr', 'mixed')),
    page_sources_json TEXT,            -- JSON: [{"page": 5, "source": "embedded", ...}]
    source_quality_json TEXT,          -- JSON: {"min_alphabetic_ratio": 0.72, ...}

    -- Footnotes
    has_footnotes INTEGER DEFAULT 0 CHECK(has_footnotes IN (0, 1)),
    footnotes_json TEXT,               -- JSON array of appended footnotes

    -- Quality flags
    quality_flags_json TEXT,           -- JSON array of quality indicators

    -- File reference
    jsonl_path TEXT NOT NULL,          -- Relative path to chunks JSONL

    -- Metadata
    pipeline_version TEXT,
    run_id INTEGER,
    created_at TEXT NOT NULL,

    FOREIGN KEY (report_id) REFERENCES reports(filename) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES chunking_runs(id) ON DELETE SET NULL,
    UNIQUE (report_id, chunk_sequence)
);
"""

# Chunking runs - tracks pipeline executions
CHUNKING_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS chunking_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL CHECK(run_type IN
        ('pages', 'documents', 'chunks', 'full', 'retry_failed')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),

    -- Input stats
    total_reports INTEGER DEFAULT 0,
    total_pages INTEGER DEFAULT 0,

    -- Output stats
    documents_created INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,

    -- Quality stats
    toc_pages_skipped INTEGER DEFAULT 0,
    low_confidence_excluded INTEGER DEFAULT 0,
    sections_detected INTEGER DEFAULT 0,
    sections_fallback INTEGER DEFAULT 0,

    -- Error stats
    error_count INTEGER DEFAULT 0,

    -- Resume info
    last_report_id TEXT,

    -- Config snapshot (includes tokenizer settings, thresholds, etc.)
    config_json TEXT
);
"""

# Chunking errors - detailed error logging
CHUNKING_ERRORS_TABLE = """
CREATE TABLE IF NOT EXISTS chunking_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    report_id TEXT NOT NULL,
    error_type TEXT NOT NULL,          -- 'consolidation_error', 'chunking_error', etc.
    error_message TEXT,
    stack_trace TEXT,
    created_at TEXT NOT NULL,

    FOREIGN KEY (run_id) REFERENCES chunking_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (report_id) REFERENCES reports(filename) ON DELETE CASCADE
);
"""

# Phase 4 indexes
PHASE4_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_report ON documents(report_id);",
    "CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(primary_source);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_report ON chunks(report_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_name);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_sequence ON chunks(report_id, chunk_sequence);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_method ON chunks(section_detection_method);",
    "CREATE INDEX IF NOT EXISTS idx_chunking_runs_status ON chunking_runs(status);",
    "CREATE INDEX IF NOT EXISTS idx_chunking_errors_run ON chunking_errors(run_id);",
]

# All tables for Phase 4
PHASE4_TABLES = [
    DOCUMENTS_TABLE,
    CHUNKS_TABLE,
    CHUNKING_RUNS_TABLE,
    CHUNKING_ERRORS_TABLE,
]

# All indexes
INDEXES = PHASE1_INDEXES + PHASE3_INDEXES + PHASE4_INDEXES

# All tables combined
ALL_TABLES = PHASE1_TABLES + PHASE3_TABLES + PHASE4_TABLES
