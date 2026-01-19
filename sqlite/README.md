# sqlite - Database Layer

SQLite database schema and query utilities for RiskRADAR pipeline state management.

---

## Table of Contents

- [Overview](#overview)
- [Role in Pipeline](#role-in-pipeline)
- [Database Location](#database-location)
- [Schema Reference](#schema-reference)
- [Table Documentation](#table-documentation)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Query Patterns](#query-patterns)
- [Schema Versioning](#schema-versioning)
- [Limitations](#limitations)

---

## Overview

The `sqlite` module provides:

- **Schema definitions** for all 14 tables across phases 1-5
- **Connection management** with automatic table creation
- **Common query functions** for CRUD operations
- **Run tracking** for pipeline execution history
- **Error logging** with stack traces for debugging

The database serves as the source of truth for pipeline state, while bulk text data is stored in JSONL/Parquet files for efficiency.

---

## Role in Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         sqlite/riskradar.db                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Phase 1-2          │  Phase 3           │  Phase 4        │  Phase 5   │
│  ─────────────      │  ─────────────     │  ────────────   │  ───────── │
│  reports (510)      │  pages (30,602)    │  documents(510) │  embed_runs│
│  scrape_progress    │  extraction_runs   │  chunks(24,766) │  upload_run│
│  scrape_errors      │  extraction_errors │  chunking_runs  │  embed_err │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          Full text in JSONL files
                          (extraction/json_data/)
```

**Design Philosophy:**
- **Metadata in SQLite**: Small, queryable data (paths, counts, timestamps)
- **Bulk text in JSONL**: Large content (page text, chunk text, embeddings)
- **Run tracking**: Every pipeline execution is logged for reproducibility

---

## Database Location

```python
from riskradar.config import DB_PATH
# Default: sqlite/riskradar.db relative to project root
```

Override via environment variable:
```bash
export RISKRADAR_DB_PATH=/path/to/custom.db
```

---

## Schema Reference

### Schema Version: 4

| Version | Phase | Tables Added |
|---------|-------|--------------|
| v1 | Phase 1 | reports, scrape_progress, scrape_errors |
| v2 | Phase 3 | pages, extraction_runs, extraction_errors |
| v3 | Phase 4 | documents, chunks, chunking_runs, chunking_errors |
| v4 | Phase 5 | embedding_runs, qdrant_upload_runs, embedding_errors |

---

## Table Documentation

### Phase 1-2: Scraping

#### `reports` - PDF Metadata (510 rows)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `filename` | TEXT | Unique PDF filename (e.g., "AAR2001.pdf") |
| `title` | TEXT | Report title from NTSB |
| `location` | TEXT | Accident location |
| `accident_date` | TEXT | Date of accident |
| `report_date` | TEXT | Date report published |
| `report_number` | TEXT | NTSB report number |
| `pdf_url` | TEXT | Source URL |
| `local_path` | TEXT | NAS storage path |
| `sha256` | TEXT | File hash for deduplication |
| `downloaded_at` | TEXT | ISO timestamp |
| `status` | TEXT | 'pending', 'downloaded', 'error' |

#### `scrape_progress` - Resume State

| Column | Type | Description |
|--------|------|-------------|
| `last_page` | INTEGER | Last pagination page processed |
| `last_report_index` | INTEGER | Index on that page |
| `updated_at` | TEXT | Last update timestamp |

#### `scrape_errors` - Error Log

| Column | Type | Description |
|--------|------|-------------|
| `report_filename` | TEXT | Failed report |
| `error_type` | TEXT | Error classification |
| `error_message` | TEXT | Error details |
| `retry_count` | INTEGER | Number of retries |

### Phase 3: Extraction

#### `pages` - Extracted Pages (30,602 rows)

| Column | Type | Description |
|--------|------|-------------|
| `report_id` | TEXT | FK to reports.filename |
| `page_number` | INTEGER | 0-indexed page |
| `json_path` | TEXT | Path to JSON file |
| `status` | TEXT | 'passed', 'failed', 'ocr_retry' |
| `extraction_method` | TEXT | 'embedded' or 'ocr' |
| `extraction_pass` | TEXT | 'initial' or 'ocr_retry' |
| `char_count` | INTEGER | Character count |
| `alphabetic_ratio` | REAL | % alphabetic chars |
| `garbage_ratio` | REAL | % garbage chars |
| `passes_threshold` | INTEGER | 1 if quality passed |
| `mean_ocr_confidence` | REAL | OCR confidence (0-100) |

#### `extraction_runs` - Pipeline Runs

| Column | Type | Description |
|--------|------|-------------|
| `run_type` | TEXT | 'initial', 'ocr_retry' |
| `status` | TEXT | 'running', 'completed', 'failed' |
| `total_pages` | INTEGER | Pages processed |
| `passed_pages` | INTEGER | Pages that passed QC |
| `config_json` | TEXT | Pipeline configuration |

### Phase 4: Chunking

#### `documents` - Consolidated Documents (510 rows)

| Column | Type | Description |
|--------|------|-------------|
| `report_id` | TEXT | Unique FK to reports |
| `total_pages` | INTEGER | Total page count |
| `included_pages` | INTEGER | Pages included in text |
| `primary_source` | TEXT | 'embedded', 'ocr', 'mixed' |
| `token_count` | INTEGER | Total tokens |
| `jsonl_path` | TEXT | Path to documents.jsonl |

#### `chunks` - Search-Ready Chunks (24,766 rows)

| Column | Type | Description |
|--------|------|-------------|
| `chunk_id` | TEXT | Unique ID: `{report}_chunk_{seq}` |
| `report_id` | TEXT | FK to reports |
| `chunk_sequence` | INTEGER | Order within document |
| `page_start` | INTEGER | Starting page |
| `page_end` | INTEGER | Ending page |
| `section_name` | TEXT | Detected section |
| `token_count` | INTEGER | Tokens in chunk |
| `text_source` | TEXT | 'embedded', 'ocr', 'mixed' |
| `has_footnotes` | INTEGER | 1 if footnotes appended |
| `jsonl_path` | TEXT | Path to chunks.jsonl |

### Phase 5: Embeddings

#### `embedding_runs` - Embedding Generation

| Column | Type | Description |
|--------|------|-------------|
| `model_name` | TEXT | 'minilm' or 'mika' |
| `status` | TEXT | 'running', 'completed', 'failed' |
| `chunks_processed` | INTEGER | Number embedded |
| `output_path` | TEXT | Parquet file path |

#### `qdrant_upload_runs` - Vector Upload

| Column | Type | Description |
|--------|------|-------------|
| `model_name` | TEXT | 'minilm' or 'mika' |
| `collection_name` | TEXT | Qdrant collection |
| `vectors_uploaded` | INTEGER | Count uploaded |
| `status` | TEXT | 'running', 'completed', 'failed' |

---

## API Reference

### `connection.py`

#### `init_db(db_path: Path) -> sqlite3.Connection`

Initialize database with all tables.

```python
from sqlite.connection import init_db
from riskradar.config import DB_PATH

conn = init_db(DB_PATH)  # Creates tables if needed
```

### `queries.py`

#### Report Queries

```python
# Get report by filename
report = queries.get_report(conn, "AAR2001.pdf")

# Count reports by status
counts = queries.count_reports_by_status(conn)
```

#### Page Queries

```python
# Get pages for a report
pages = queries.get_pages_for_report(conn, "AAR2001.pdf")

# Count pages by extraction method
stats = queries.count_pages_by_method(conn)
```

#### Run Tracking

```python
# Create a new run
run_id = queries.create_extraction_run(conn, run_type="initial")

# Update run status
queries.update_extraction_run(conn, run_id, status="completed", total_pages=100)

# Log error
queries.log_extraction_error(conn, run_id, report_id, error_type, message, stack_trace)
```

---

## Usage Examples

### Direct SQL Access

```bash
# Open database
sqlite3 sqlite/riskradar.db

# View tables
.tables

# Query reports
SELECT filename, title, accident_date FROM reports LIMIT 5;

# Count by extraction method
SELECT extraction_method, COUNT(*) FROM pages GROUP BY extraction_method;

# View recent runs
SELECT * FROM extraction_runs ORDER BY id DESC LIMIT 5;
```

### Python Integration

```python
from riskradar.config import DB_PATH
from sqlite.connection import init_db
from sqlite import queries

conn = init_db(DB_PATH)

# Get pipeline statistics
report_count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
page_count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

print(f"Reports: {report_count}")
print(f"Pages: {page_count}")
print(f"Chunks: {chunk_count}")

conn.close()
```

---

## Query Patterns

### Quality Analysis

```sql
-- Pages by quality status
SELECT status, COUNT(*) as count
FROM pages
GROUP BY status;

-- Low-quality OCR pages
SELECT report_id, page_number, mean_ocr_confidence
FROM pages
WHERE extraction_method = 'ocr'
  AND mean_ocr_confidence < 70
ORDER BY mean_ocr_confidence;
```

### Chunk Distribution

```sql
-- Token distribution
SELECT
  CASE
    WHEN token_count < 400 THEN 'under_400'
    WHEN token_count <= 800 THEN '400_to_800'
    ELSE 'over_800'
  END as token_range,
  COUNT(*) as count
FROM chunks
GROUP BY token_range;

-- Chunks by section
SELECT section_name, COUNT(*) as count
FROM chunks
WHERE section_name IS NOT NULL
GROUP BY section_name
ORDER BY count DESC;
```

### Run History

```sql
-- Recent embedding runs
SELECT model_name, status, chunks_processed, created_at
FROM embedding_runs
ORDER BY id DESC
LIMIT 10;

-- Errors for a specific run
SELECT report_id, error_type, error_message
FROM extraction_errors
WHERE run_id = ?;
```

---

## Schema Versioning

The schema supports incremental upgrades:

```python
from sqlite.schema import SCHEMA_VERSION  # Currently 4

# Tables are created with IF NOT EXISTS
# New tables can be added without breaking existing data
```

**Migration Notes:**
- No formal migration system (schema additions only)
- Backward compatible (new tables, new columns)
- Full rebuild supported via pipeline re-execution

---

## Limitations

1. **No Full Text in DB**: Chunk text not stored in SQLite (use JSONL files). This keeps the database small (~5MB) but requires file access for text content.

2. **Single-Writer**: SQLite doesn't support concurrent writes well. Don't run multiple pipeline instances simultaneously.

3. **No Foreign Key Enforcement**: SQLite FK constraints defined but not enforced by default. Use `PRAGMA foreign_keys = ON;` if needed.

4. **Local File Only**: Database is local file, not network-accessible. For distributed setups, consider PostgreSQL.

5. **No Automatic Cleanup**: Old run records and errors accumulate. Manual cleanup may be needed for long-running projects.

---

## Files

| File | Purpose |
|------|---------|
| `schema.py` | All table definitions (14 tables) |
| `connection.py` | Connection management and initialization |
| `queries.py` | Common SQL query functions |
| `__init__.py` | Package marker |

---

## See Also

- [Main README](../README.md) - Project overview
- [riskradar/README.md](../riskradar/README.md) - Configuration
- [extraction/README.md](../extraction/README.md) - Pipeline that writes to pages/chunks tables
- [analytics/README.md](../analytics/README.md) - DuckDB analytics on Parquet exports
