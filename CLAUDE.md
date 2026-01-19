# RiskRADAR - Claude Code Context

## Project Overview

RiskRADAR (Retrieval and Discovery of Aviation Accident Reports) is an end-to-end data + ML portfolio project that transforms unstructured NTSB aviation accident PDFs into searchable, structured, explainable safety insights using embeddings, a vector database, and a hierarchical taxonomy.

---

## Current Status (Updated 2026-01-18)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 - Foundations | Complete | Repo scaffold, config, logging |
| Phase 1 - Scraping | Complete | 510 PDFs downloaded, metadata captured |
| Phase 2 - Metadata | Complete | Metadata stored during scraping |
| Phase 3 - Extraction | Complete | 30,602 pages extracted (14K embedded + 16K OCR) |
| Phase 4 - Chunking | v1 Complete, v2 In Progress | Chunking improvements underway |
| Phase 5 - Embeddings | v1 Complete, v2 Pending | Both models benchmarked, improvements planned |
| Phase 5b - Hybrid Search | **Planned** | BM25 + RRF after v2 chunking |
| Phase 6-8 | Not Started | |

**Phase 5 v1 Benchmark Results (2026-01-18):**

| Model | Automated MRR | Hit@10 | Semantic Precision | False Positive Rate |
|-------|---------------|--------|-------------------|---------------------|
| MiniLM | 0.669 | 94.9% | **75.5%** | **16.4%** |
| MIKA | 0.788 | 94.9% | 60.0% | 34.5% |

**Key Finding:** Human evaluation showed MiniLM outperforms MIKA despite MIKA's better automated metrics. MIKA has 2x the false positive rate.

**Root Cause Identified:** Chunking issues causing noise:
- 759 chunks under 10 tokens (2.7%)
- 140 reports start with header-only chunks like "1. The Accident"
- Section detection treats subsections (1.1) as siblings of parent sections (1.)

---

## Phase 5 v2 Improvement Plan (Active)

### Overview

Comprehensive chunking and retrieval improvements with full versioning for rollback capability and cross-version comparison.

### Problems Identified in v1

1. **Hierarchical section bug:** "1. The Accident" and "1.1 History of Flight" treated as siblings, creating 17-character header-only sections
2. **No minimum chunk size:** 759 chunks under 10 tokens, 2,501 under 50 tokens
3. **Overlap resets at section boundaries:** Context lost between sections
4. **Sentence regex issue:** Splits "1." from "The Accident" due to period detection
5. **Headers dominate tiny chunks:** Section headers without content pollute embedding space

### v2 Chunking Parameters

| Parameter | v1 Value | v2 Value | Rationale |
|-----------|----------|----------|-----------|
| Minimum tokens | 0 | 400 | Eliminate noise chunks |
| Maximum tokens | 700 | 800 | Allow slightly larger for context |
| Target tokens | 600 | 600 | Unchanged |
| Overlap | 20% | 25% | Better boundary coverage |
| Section prefix | No | Yes | "[SECTION] content..." format |
| Cross-section overlap | No | Yes | Maintain context continuity |

### Versioning Strategy

**File Structure:**
```
extraction/json_data/
├── pages.jsonl              # Source data (unchanged)
├── documents.jsonl          # Source data (unchanged)
├── chunks_v1.jsonl          # Original chunks (preserved)
└── chunks_v2.jsonl          # Improved chunks

embeddings_data/
├── v1/
│   ├── minilm_embeddings.parquet
│   └── mika_embeddings.parquet
└── v2/
    ├── minilm_embeddings.parquet
    └── mika_embeddings.parquet

analytics/data/
├── chunks_v1.parquet
└── chunks_v2.parquet

eval/
├── results_v1/              # v1 benchmark results
├── results_v2/              # v2 benchmark results
├── results_hybrid/          # BM25+RRF results
├── human_reviews_v1/        # v1 human reviews
└── human_reviews_v2/        # v2 human reviews
```

**Qdrant Collections:**
| Version | MiniLM Collection | MIKA Collection |
|---------|-------------------|-----------------|
| v1 | `riskradar_minilm_v1` | `riskradar_mika_v1` |
| v2 | `riskradar_minilm_v2` | `riskradar_mika_v2` |

### Implementation Stages

#### Stage 0: Version Current Artifacts
- [ ] Rename `chunks.jsonl` → `chunks_v1.jsonl`
- [ ] Create `embeddings_data/v1/` and move current parquets
- [ ] Rename/alias Qdrant collections to v1
- [ ] Move `eval/results/` → `eval/results_v1/`
- [ ] Move `eval/human_reviews/` → `eval/human_reviews_v1/`
- [ ] Update config.py with versioning support
- [ ] Commit: "chore: version v1 artifacts before chunking improvements"

#### Stage 1: Chunking Fixes
- [ ] 1a: Update `section_detect.py` for hierarchical numbering (1.1 is child of 1.)
- [ ] 1b: Add 400 token minimum with forward borrowing from next section
- [ ] 1c: Implement section prefix approach `[SECTION] content...`
- [ ] 1d: Increase overlap to 25%, carry across section boundaries
- [ ] 1e: Fix sentence regex to protect "1." patterns
- [ ] Unit tests for each fix
- [ ] Commit: "feat: chunking v2 - hierarchical sections, min size, prefix, overlap"

#### Stage 2: Re-Process Everything
- [ ] 2a: Run chunking pipeline → `chunks_v2.jsonl`
- [ ] 2b: Compare v1 vs v2 statistics (target: 0 chunks under 50 tokens)
- [ ] 2c: Embed MiniLM → `embeddings_data/v2/minilm_embeddings.parquet`
- [ ] 2d: Embed MIKA → `embeddings_data/v2/mika_embeddings.parquet`
- [ ] 2e: Upload to Qdrant v2 collections
- [ ] 2f: Verify v2 collections via CLI
- [ ] Commit: "feat: generate v2 chunks and embeddings"

#### Stage 3: Benchmark v2
- [ ] 3a: Update `eval/benchmark.py` to support version parameter
- [ ] 3b: Run automated benchmark on v2 (both models)
- [ ] 3c: Export for human review → `human_reviews_v2/`
- [ ] 3d: Complete human review
- [ ] 3e: Import human reviews and generate v2 report
- [ ] 3f: Generate v1 vs v2 comparison report
- [ ] Commit: "feat: v2 benchmark results"

#### Stage 4: Hybrid Search (BM25 + RRF)
- [ ] 4a: Implement BM25 index on v2 chunks
- [ ] 4b: Implement Reciprocal Rank Fusion (RRF)
- [ ] 4c: Add hybrid search mode to benchmark harness
- [ ] 4d: Run hybrid benchmark
- [ ] 4e: Compare: vector-only vs hybrid
- [ ] Commit: "feat: BM25 + RRF hybrid search"

#### Stage 5: Final Comparison & Documentation
- [ ] 5a: Generate comprehensive comparison report:
  - v1 MiniLM vs v2 MiniLM
  - v1 MIKA vs v2 MIKA
  - v2 MiniLM vs v2 MIKA
  - v2 Vector vs v2 Hybrid
- [ ] 5b: Document recommendations for production
- [ ] 5c: Update CLAUDE.md with final v2 status
- [ ] Commit: "docs: final benchmark comparison and recommendations"

### Expected Outcomes

| Metric | v1 Baseline | v2 Target |
|--------|-------------|-----------|
| Chunks under 50 tokens | 2,501 (8.8%) | 0 (0%) |
| Chunks under 100 tokens | 3,429 (12.1%) | <100 (<0.5%) |
| MiniLM False Positive Rate | 16.4% | <10% |
| MIKA False Positive Rate | 34.5% | <15% (if chunking was the issue) |
| Hybrid improvement over vector | N/A | +10-20% expected |

### Rollback Strategy

If v2 performs worse:
```python
# In config.py - single line rollback
ACTIVE_VERSION = "v1"

# Or via environment variable
RISKRADAR_VERSION=v1 python -m streamlit run app.py
```

All v1 artifacts preserved. Can A/B test versions.

### Cross-Version Comparison Report Structure

```markdown
# RiskRADAR Version Comparison Report

## Chunking Quality
| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| Total chunks | 28,321 | TBD | - |
| Chunks < 50 tokens | 2,501 | TBD | - |
| Avg tokens/chunk | 534 | TBD | - |

## MiniLM Performance
| Metric | v1 | v2 | Δ |
|--------|----|----|---|
| MRR | 0.669 | TBD | - |
| Semantic Precision | 75.5% | TBD | - |
| False Positive Rate | 16.4% | TBD | - |

## MIKA Performance
| Metric | v1 | v2 | Δ |
|--------|----|----|---|
| MRR | 0.788 | TBD | - |
| Semantic Precision | 60.0% | TBD | - |
| False Positive Rate | 34.5% | TBD | - |

## Hybrid Search (v2)
| Metric | Vector Only | Hybrid | Improvement |
|--------|-------------|--------|-------------|
```

### CLI Commands (v2)

```bash
# Chunking with version
python -m extraction.processing.chunk all --version v2

# Embedding with version
python -m embeddings.cli embed both --version v2
python -m embeddings.cli upload both --version v2

# Benchmark with version
python -m eval.benchmark run --version v2
python -m eval.benchmark compare v1 v2

# Hybrid search (after Stage 4)
python -m eval.benchmark run --version v2 --hybrid
```

---

**Phase 4 v1 Results (Preserved for Reference):**
- 30,602 pages consolidated into 510 documents
- 28,321 chunks created (avg 534 tokens, median 658 tokens)
- 15.1M total tokens across all chunks
- Text source: 46% embedded, 53% OCR, 1% mixed
- Section detection: 95% pattern match, 5% paragraph fallback

---

## Goals

- Scrape PDFs and manage raw file storage with lineage
- Extract and store rich metadata tied to each file
- OCR and text extraction with quality tracking
- Chunk documents reliably for retrieval + analysis
- Generate embeddings and build a vector search index
- Produce an interpretable "hierarchical cause map" (weakly supervised / rule + embedding signals)
- Ship a Streamlit application that supports search + insights

## Non-Goals

- Training a deep supervised model with large human-labeled datasets
- Predicting future accidents or making causal claims
- Building a multi-tenant SaaS (auth/billing/etc.) for production
- Perfect extraction of every field from every report (focus on robust pipeline + measurable quality)

---

## Tech Stack (Locked)

- **Language:** Python 3.9+
- **Scraping:** Selenium + webdriver-manager
- **PDF parsing:** pymupdf
- **OCR:** pytesseract + pdf2image (optional: opencv-python for cleanup)
- **Storage:** Local filesystem (NAS) + SQLite
- **Analytics:** DuckDB + Parquet (ad-hoc SQL queries)
- **Vector DB:** Qdrant Cloud
- **Embeddings:** sentence-transformers
  - Baseline: `sentence-transformers/all-MiniLM-L6-v2`
  - Domain IR: `NASA-AIML/MIKA_Custom_IR`
- **App:** Streamlit + secrets manager
- **Packaging:** pip

---

## Python Virtual Environment

**CRITICAL:** Always use the virtual environment for this project. The venv ensures consistent dependencies and avoids polluting the global Python installation.

### Setup (First Time)
```powershell
cd C:\Users\bvlma\CODE\riskRADAR
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e ./scraper
```

### Activation (Every Session)
```powershell
cd C:\Users\bvlma\CODE\riskRADAR
venv\Scripts\activate
```

Your prompt should show `(venv)` when activated.

### IMPORTANT: Use `python` NOT `py`

| Command | Behavior | Use When |
|---------|----------|----------|
| `python` | Uses venv Python when activated | **Always use this** |
| `py` | Windows launcher - ignores venv, uses global Python | Never use with venv |

**Correct usage (venv activated):**
```powershell
python -m analytics.cli                    # ✓ Correct
python -m extraction.processing.chunk all  # ✓ Correct
python -m pip install duckdb              # ✓ Correct
```

**Wrong usage (bypasses venv):**
```powershell
py -m analytics.cli                        # ✗ Wrong - uses global Python
py -m pip install duckdb                   # ✗ Wrong - installs globally
```

### Verify Venv is Active
```powershell
python -c "import sys; print(sys.executable)"
# Should print: C:\Users\bvlma\CODE\riskRADAR\venv\Scripts\python.exe
```

### Installing New Packages
Always install within the activated venv:
```powershell
python -m pip install <package>
```

Then add to `requirements.txt` if it's a project dependency.

---

## PDF Storage (NAS)

PDFs are stored on a NAS, NOT in the project directory:
- **Path:** `\\TRUENAS\Photos\RiskRADAR`
- **Reason:** Laptop storage constraints; PDFs should not be committed to git
- **Pattern:** Keep original filenames from NTSB (e.g., `AAR6903.pdf`, `AIR2507.pdf`)

### NAS Connection Troubleshooting (Windows)

If you get "network path not found" errors when accessing the NAS:

1. **Verify NAS is accessible:**
   ```powershell
   Test-Connection TRUENAS -Count 2
   ```

2. **Check if share is mounted:**
   ```powershell
   net use
   ```

3. **Re-mount if needed:**
   ```powershell
   # If disconnected, reconnect:
   net use \\TRUENAS\Photos /persistent:yes
   ```

4. **Verify path exists:**
   ```powershell
   Test-Path "\\TRUENAS\Photos\RiskRADAR"
   ```

5. **Common issues:**
   - NAS may be asleep - access any file to wake it
   - VPN may block local network - disconnect VPN
   - Firewall blocking SMB - check Windows Firewall settings

---

## SQLite Database

The project uses SQLite for metadata and progress tracking.

- **Location:** `sqlite/riskradar.db` (NOT in project root)
- **Config:** Set via `RISKRADAR_DB_PATH` env var or defaults to `sqlite/riskradar.db`

### Key Tables

| Table | Purpose |
|-------|---------|
| `reports` | PDF metadata from scraping (510 rows) |
| `pages` | Per-page extraction tracking (30,602 rows) |
| `extraction_runs` | Extraction pipeline execution history |
| `extraction_errors` | Extraction error log with details |
| `documents` | Consolidated document metadata (510 rows) |
| `chunks` | Search-ready chunks (28,321 rows) |
| `chunking_runs` | Chunking pipeline execution history |
| `chunking_errors` | Chunking error log with details |
| `embedding_runs` | Phase 5: Embedding generation runs |
| `qdrant_upload_runs` | Phase 5: Qdrant upload runs |
| `embedding_errors` | Phase 5: Embedding/upload error log |
| `scrape_progress` | Scraper resume state |
| `scrape_errors` | Scraping error log |

### Quick Database Access

```python
from sqlite.connection import init_db
from riskradar.config import DB_PATH

conn = init_db(DB_PATH)  # Creates tables if needed
cursor = conn.execute("SELECT COUNT(*) FROM reports")
print(cursor.fetchone()[0])  # 510
```

Or via command line:
```bash
sqlite3 sqlite/riskradar.db "SELECT COUNT(*) FROM reports;"
```

---

## Scraper Library

The `scraper/` library is a standalone Selenium wrapper.

### Installation
```bash
pip install -e ./scraper
```

### Basic Pattern
```python
from pathlib import Path
from scraper.config import BrowserConfig
from scraper.browser import chrome
from scraper.actions import go_to, click
from scraper.download import wait_for_new_download, move_and_rename

NAS_PATH = Path(r"\\TRUENAS\Photos\RiskRADAR")
config = BrowserConfig()

with chrome(config) as driver:
    go_to(driver, url)
    click(driver, pdf_link_locator)
    downloaded = wait_for_new_download(config)

final_path = move_and_rename(downloaded, NAS_PATH, f"{report_id}.pdf")
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `SCRAPER_HEADLESS` | `1` | Headless mode (set to `0` for debugging) |
| `SCRAPER_DOWNLOADS_DIR` | `.scraper_tmp_downloads` | Temp download directory |
| `SCRAPER_DOWNLOAD_TIMEOUT` | `120` | Download timeout (seconds) |
| `HTTP_USER_AGENT` | `RiskRADARBot/1.0` | Browser user agent |
| `SCRAPER_REQUEST_DELAY` | `2.0` | Delay between requests (seconds) |
| `SCRAPER_DOWNLOAD_DELAY` | `3.0` | Delay between downloads (seconds) |
| `SCRAPER_PAGE_DELAY` | `2.0` | Delay between page navigations (seconds) |

---

## Implementation Phases

### Phase 0 — Repo & Foundations
**Deliverables:** Repo scaffold, tooling, config, CI checks

**Tasks:**
- Create `.env.example`, `requirements.txt`
- Add logging setup and structured config
- Add pre-commit hooks (ruff/black, optional)

---

### Phase 1 — PDF Scraping + Raw Storage [COMPLETE]

**Data Source**
- **URL:** https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx
- **Downloaded:** 510 Aviation accident report PDFs (as of 2026-01-06)
- **Date range:** 1966 to present

**Results:**
- 510 PDFs successfully downloaded to NAS
- 97% have titles, 96% have accident dates, 100% have report numbers
- All metadata stored in `reports` table

#### robots.txt Compliance
Checked: https://www.ntsb.gov/robots.txt
- `/investigations/AccidentReports/` is **ALLOWED** (not in Disallow list)
- No `Crawl-delay` specified, so we implement our own
- Blocked paths (not relevant to us): `/_layouts/`, `/_vti_bin/`, `/Lists/`, `/Search/`, etc.

#### Rate Limiting Requirements
**CRITICAL: Do not DDOS the NTSB website!**

| Setting | Value | Purpose |
|---------|-------|---------|
| `SCRAPER_REQUEST_DELAY` | 2.0s | Between any requests |
| `SCRAPER_DOWNLOAD_DELAY` | 3.0s | Between PDF downloads |
| `SCRAPER_PAGE_DELAY` | 2.0s | Between pagination clicks |

With 781 PDFs and ~79 pages:
- Estimated minimum runtime: ~45-60 minutes
- Use `rate_limit()` with jitter to vary timing
- Run during off-peak hours if possible

#### Page Selectors
The mode dropdown is in the page source:
```html
<select name="year_select" id="mode_dropdown">
    <option value="Aviation">Aviation</option>
    <option value="HazMat">Hazardous Materials</option>
    <option value="Highway">Highway</option>
    <option value="Marine">Marine</option>
    <option value="Pipeline">Pipeline</option>
    <option value="Railroad">Railroad</option>
</select>
```

**Selenium locator:**
```python
from selenium.webdriver.common.by import By

MODE_DROPDOWN = (By.ID, "mode_dropdown")
select_dropdown_by_value(driver, MODE_DROPDOWN, "Aviation")
```

#### Scraping Workflow
```python
from scraper.config import BrowserConfig
from scraper.browser import chrome
from scraper.actions import go_to, select_dropdown_by_value, click
from scraper.waits import rate_limit
from scraper.download import wait_for_new_download, move_and_rename

config = BrowserConfig()

with chrome(config) as driver:
    # 1. Navigate to reports page
    go_to(driver, NTSB_URL)
    rate_limit(config.request_delay_sec)

    # 2. Select Aviation from dropdown (CRITICAL)
    select_dropdown_by_value(driver, MODE_DROPDOWN, "Aviation")
    rate_limit(config.page_delay_sec)  # Wait for page to update

    # 3. For each page:
    while has_more_pages:
        # Extract metadata from all 10 reports on page
        for report in reports_on_page:
            # Extract: title, location, dates, report_number
            # Find PDF icon link, extract URL and filename
            # Download PDF
            click(driver, pdf_link)
            downloaded = wait_for_new_download(config)
            move_and_rename(downloaded, NAS_PATH, filename)
            rate_limit(config.download_delay_sec)  # RESPECT THE SERVER!
            # Save metadata to SQLite

        # Click next page
        click(driver, NEXT_BUTTON)
        rate_limit(config.page_delay_sec)
```

**Metadata to Capture (from webpage, NOT PDF)**
| Field | Example |
|-------|---------|
| title | "Crash of Pan American World Airways Boeing 727" |
| location | "Berlin, State not available" |
| accident_date | 11/14/1966 |
| report_date | 6/4/1968 |
| report_number | AAR-68-AH |
| pdf_url | Full URL to PDF file |
| filename | Original PDF filename (e.g., AIR2507.pdf) |

**Database Schema (reports table)**
```sql
CREATE TABLE reports (
    id INTEGER PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    title TEXT,
    location TEXT,
    accident_date DATE,
    report_date DATE,
    report_number TEXT,
    pdf_url TEXT,
    local_path TEXT,
    sha256 TEXT,
    downloaded_at TIMESTAMP,
    status TEXT
);
```

**Acceptance Criteria**
- Can ingest all 781 Aviation PDFs reproducibly
- All metadata captured from webpage and stored in SQLite
- Re-run does not re-download duplicates
- Errors logged and recoverable (resume from failure point)
- **Rate limiting enforced** - no more than 1 request per 2 seconds

---

### Phase 2 — Metadata Extraction + Relational Model

**Deliverables:** Metadata tables keyed by `report_id`, fields usable for filtering

**Tasks:**
- Parse metadata from filename/HTML/embedded PDF metadata
- Create canonical `report_id`
- Store in SQLite

**Acceptance Criteria**
- At least 3–5 metadata fields reliably populated for majority of docs
- Can query "reports between dates", "by aircraft make/model", etc.

---

### Phase 3 — Text Extraction + OCR with Quality Metrics

**Per-Page Workflow**
1. Attempt embedded text extraction using pypdf
2. Compute quality heuristics:
   - `char_count` - total characters extracted
   - `alphabetic_ratio` - % of alphabetic characters
   - `garbage_ratio` - % of garbage/unrecognized characters
3. If below threshold or empty → run OCR (pytesseract + pdf2image)

**Quality Thresholds:** TBD - research best practices during implementation

**Database Schema (pages table)**
```sql
CREATE TABLE pages (
    id INTEGER PRIMARY KEY,
    report_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    text TEXT,
    text_source TEXT,  -- 'embedded' or 'ocr'
    char_count INTEGER,
    alphabetic_ratio REAL,
    garbage_ratio REAL,
    FOREIGN KEY (report_id) REFERENCES reports(filename),
    UNIQUE (report_id, page_number)
);
```

**Acceptance Criteria**
- Text is readable and attributable to pages
- Quality metrics computed for every page
- OCR failures are logged and recoverable

---

### Phase 4 — Chunking (Search-Ready Text) [COMPLETE]

**Three-Pass Pipeline**
```
Pass 0: JSON files → pages.jsonl (consolidated, deduplicated, ordered)
Pass 1: pages.jsonl → documents.jsonl (per-report full text)
Pass 2: documents.jsonl → chunks.jsonl (search-ready segments)
```

**CLI Commands**
```bash
# Run full pipeline
python -m extraction.processing.chunk all

# Run individual passes
python -m extraction.processing.chunk pages      # Pass 0
python -m extraction.processing.chunk documents  # Pass 1
python -m extraction.processing.chunk chunks     # Pass 2
```

**Parameters**
- **Token limits:** 500-700 tokens per chunk (target 600)
- **Overlap:** 20% between consecutive chunks (~120 tokens)
- **Tokenizer:** tiktoken cl100k_base
- **Strategy:** Section-aware chunking with fallback

**Output Files (JSONL format, one line per record)**
| File | Records | Size |
|------|---------|------|
| `pages.jsonl` | 30,602 | ~69 MB |
| `documents.jsonl` | 510 | ~68 MB |
| `chunks.jsonl` | 28,321 | ~85 MB |

**Results**
- 28,321 chunks from 510 documents
- Avg 534 tokens/chunk, median 658 tokens
- Token distribution: 33% under 500, 35% in range, 32% over 700
- Section detection: 95% pattern match, 5% paragraph fallback, <1% no structure
- Footnotes appended to 1,264 chunks

**Database Schema (chunks table)**
```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    report_id TEXT NOT NULL,
    chunk_sequence INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    page_list_json TEXT,
    char_start INTEGER,
    char_end INTEGER,
    section_name TEXT,
    section_number TEXT,
    section_detection_method TEXT,
    chunk_text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    overlap_tokens INTEGER DEFAULT 0,
    text_source TEXT,
    page_sources_json TEXT,
    source_quality_json TEXT,
    has_footnotes INTEGER DEFAULT 0,
    footnotes_json TEXT,
    quality_flags_json TEXT,
    jsonl_path TEXT NOT NULL,
    pipeline_version TEXT,
    run_id INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (report_id) REFERENCES reports(filename),
    UNIQUE (report_id, chunk_sequence)
);
```

**Acceptance Criteria** (all met)
- Chunks are coherent (not mid-sentence fragments)
- Each chunk targets 500-700 tokens with 20% overlap
- Each chunk has traceability back to report + page range
- Section metadata populated where headers detected
- Footnotes appended to referencing chunks
- Pipeline is idempotent and resumable

---

### Phase 5 — Embeddings + Vector Database + Benchmark [IN PROGRESS]

**Embedding Models**
| Model | Dimensions | Collection Name | Purpose |
|-------|------------|-----------------|---------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | `riskradar_minilm` | Baseline - general purpose |
| `NASA-AIML/MIKA_Custom_IR` | 768 | `riskradar_mika` | Domain IR - aerospace/aviation specialized |

**Implementation Complete:**
- `embeddings/` module with CLI, model wrapper, Parquet storage, Qdrant upload
- `eval/` module with professional benchmark framework
- SQLite run tracking tables (embedding_runs, qdrant_upload_runs)
- Secrets management via `.env` and `.streamlit/secrets.toml`

**CLI Commands**
```bash
# Embedding generation
python -m embeddings.cli embed minilm        # MiniLM only
python -m embeddings.cli embed mika          # MIKA only
python -m embeddings.cli embed both          # Both models

# Qdrant upload
python -m embeddings.cli upload minilm
python -m embeddings.cli upload both

# Full pipeline
python -m embeddings.cli all                 # Embed + upload both

# Verification
python -m embeddings.cli verify minilm
python -m embeddings.cli stats

# Benchmarking
python -m eval.benchmark run                 # Run both models
python -m eval.benchmark run -m minilm       # Single model
python -m eval.benchmark report              # Generate report
python -m eval.benchmark validate            # Validate queries
```

**Vector Payload Schema (Qdrant)**
| Field | Type | Notes |
|-------|------|-------|
| `chunk_id` | string | Unique chunk identifier |
| `report_id` | string | Links to reports table |
| `chunk_sequence` | int | Order within document |
| `page_start` | int | Starting page number |
| `page_end` | int | Ending page number |
| `section_name` | string | Detected section name |
| `token_count` | int | Token count |
| `text_source` | string | `embedded`, `ocr`, or `mixed` |
| `accident_date` | string | From webpage scraping |
| `report_date` | string | From webpage scraping |
| `location` | string | From webpage scraping |
| `title` | string | Report title |

**Benchmark Framework (50 queries)**

| Category | Count | Difficulty | Purpose |
|----------|-------|------------|---------|
| Incident Lookup | 10 | Easy | Known accidents (Alaska 261, ValuJet 592, etc.) |
| Conceptual Queries | 12 | Medium-Hard | Technical concepts (CRM, CFIT, fatigue) |
| Section Queries | 10 | Medium | Structural retrieval (PROBABLE CAUSE, etc.) |
| Comparative Queries | 8 | Hard | Analytical patterns |
| Aircraft Queries | 6 | Medium | Aircraft-type filtering |
| Phase Queries | 4 | Medium | Flight phase specific |

**Benchmark Metrics**
- MRR (Mean Reciprocal Rank)
- Hit@K (K = 1, 3, 5, 10, 20)
- Precision@K, Recall@K
- nDCG@10 (Normalized DCG)
- Section Accuracy (for section queries)
- Latency (embed + search time)
- Statistical tests (paired t-test, Wilcoxon, bootstrap CI)

**Output Files**
```
eval/
├── gold_queries.yaml          # 50 stratified test queries
├── benchmark.py               # Benchmark runner
├── benchmark_report.md        # Generated comparison report
└── results/
    ├── benchmark_minilm_*.json
    ├── benchmark_minilm_*.parquet  # Streamlit-ready
    ├── benchmark_mika_*.json
    └── benchmark_mika_*.parquet
```

**Acceptance Criteria**
- Both models successfully embed all 28,321 chunks
- Both Qdrant collections queryable
- Benchmark report with statistical comparison
- Clear winner recommendation for Streamlit app

---

### Phase 6 — Hierarchical Taxonomy + Weak Signals

**Taxonomy Structure**
- File: `taxonomy.yaml`
- Size: 20-30 nodes maximum
- Depth: 2-3 levels
- Human-in-the-loop curation

**Scoring Method**
1. **Weighted keyword rules** - pattern matching
2. **Embedding similarity** - to seed examples per leaf node
3. **Soft scores** - 0-1 range
4. **Roll-up** - deterministic aggregation to parent nodes

**Database Schema**
```sql
CREATE TABLE taxonomy_scores (
    report_id TEXT,
    node_path TEXT,
    score REAL,
    evidence_chunk_ids_json TEXT,
    PRIMARY KEY (report_id, node_path)
);

CREATE TABLE taxonomy_trends (
    node_path TEXT,
    period TEXT,
    prevalence REAL,
    mean_score REAL,
    n_reports INTEGER,
    PRIMARY KEY (node_path, period)
);
```

**Acceptance Criteria**
- Scores look plausible and are evidence-backed
- Scores roll up correctly to parent nodes

---

### Phase 7 — Trend + Prevalence Analytics

**Tasks:**
- Define prevalence: % of reports where node_score >= threshold
- Build aggregates by month/quarter/year
- Populate `taxonomy_trends` table

**Acceptance Criteria**
- Can answer: "which themes increased over time?"
- Charts reproducible from `taxonomy_trends` table

---

### Phase 8 — Streamlit Application

**Three Pages:**

**Page 1: Semantic Search**
- Query box for natural language search
- Metadata filters (date, aircraft, location, report type)
- Results list with relevance ranking
- Best match excerpt + source citation

**Page 2: Cause Map Explorer**
- Browse taxonomy tree
- Per-node: description, top reports, evidence snippets, trend chart

**Page 3: Analysis Dashboard**
- Time series visualizations
- Theme prevalence over time
- Export capabilities (CSV downloads)

**Model Comparison:** App allows switching between MiniLM and MIKA collections

**Acceptance Criteria**
- Hiring manager can run app locally with one command
- All three pages functional and polished
- Secrets properly managed (no hardcoded keys)

---

## Future Roadmap (Post-MVP)

- **User PDF upload:** Allow users to upload their own NTSB PDF and find similar content via semantic search

---

## Security & Secrets Management

### CRITICAL: Never Commit Secrets

The following files contain secrets and are gitignored - **NEVER commit them**:

| File | Purpose | Status |
|------|---------|--------|
| `.env` | Environment variables for CLI/pipelines | Gitignored |
| `.streamlit/secrets.toml` | Streamlit app secrets | Gitignored |

### Setting Up Credentials

**For CLI and pipeline usage** - Edit `.env`:
```bash
QDRANT_URL=https://your-cluster.region.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_api_key_here
```

**For Streamlit app** - Edit `.streamlit/secrets.toml`:
```toml
QDRANT_URL = "https://your-cluster.region.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your_api_key_here"
```

### Getting Qdrant Credentials

1. Go to https://cloud.qdrant.io/
2. Create account (free tier is sufficient)
3. Create a cluster
4. Go to **API Keys** tab → Create new key
5. Copy cluster URL and API key

### Security Best Practices

1. **Never hardcode secrets** in Python files
2. **Never log API keys** - only log URLs (without keys)
3. **Use environment variables** or Streamlit secrets
4. **Rotate keys** if accidentally exposed
5. **Check before commits**: `git status` should never show `.env` or `secrets.toml`

### Verifying Gitignore Protection

```powershell
# These should return nothing (files are ignored)
git status --ignored | Select-String ".env"
git status --ignored | Select-String "secrets.toml"

# If files appear in git status, they are NOT protected - fix immediately!
```

### Template Files (Safe to Commit)

| File | Purpose |
|------|---------|
| `.env.example` | Template showing required variables (no real values) |

---

## Development Guidelines

1. **Scraping scripts** go in a dedicated module (e.g., `src/ingestion/`)
2. **PDF storage** always targets the NAS path
3. **Deduplication** by file hash (sha256) before downloading
4. **Metadata tracking** in SQLite
5. **Error handling** with retries for network failures
6. **Logging** for tracking progress and failures
7. **Secrets** via `.env` (CLI) or `.streamlit/secrets.toml` (app) - NEVER hardcode

---

## Key Files & Directories

### Core Code
| Path | Purpose |
|------|---------|
| `riskradar/config.py` | Shared configuration (DB_PATH, NAS_PATH, etc.) |
| `sqlite/` | Database schema, connection, queries |
| `sqlite/riskradar.db` | **SQLite database** (reports, pages, runs) |
| `scraper/` | Selenium-based web scraping library |
| `extraction/` | PDF text extraction pipeline |

### Extraction Pipeline (Phase 3)
| Path | Purpose |
|------|---------|
| `extraction/processing/extract.py` | Extraction pipeline orchestration |
| `extraction/processing/pdf_reader.py` | Embedded text extraction (pymupdf) |
| `extraction/processing/ocr.py` | OCR with confidence scoring (pytesseract) |
| `extraction/processing/quality.py` | Quality thresholds and metrics |
| `extraction/processing/analytics.py` | Quality reporting queries |
| `extraction/json_data/passed/` | Pages that passed quality checks |
| `extraction/json_data/ocr_retry/` | OCR re-extraction results |

### Chunking Pipeline (Phase 4)
| Path | Purpose |
|------|---------|
| `extraction/processing/chunk.py` | Chunking pipeline CLI entry point |
| `extraction/processing/consolidate_pages.py` | Pass 0: JSON → pages.jsonl |
| `extraction/processing/consolidate.py` | Pass 1: pages → documents.jsonl |
| `extraction/processing/section_detect.py` | Section header detection |
| `extraction/processing/toc_detect.py` | TOC page detection |
| `extraction/processing/footnote_parse.py` | Footnote extraction |
| `extraction/processing/tokenizer.py` | tiktoken cl100k wrapper |
| `extraction/json_data/pages.jsonl` | Consolidated pages (30K lines) |
| `extraction/json_data/documents.jsonl` | Full documents (510 lines) |
| `extraction/json_data/chunks.jsonl` | Search-ready chunks (28K lines) |

### Analytics Engine (DuckDB)
| Path | Purpose |
|------|---------|
| `analytics/convert.py` | JSONL → Parquet conversion |
| `analytics/views.py` | Pre-built analytical views |
| `analytics/cli.py` | Interactive SQL shell |
| `analytics/data/pages.parquet` | Pages data (19.5 MB) |
| `analytics/data/documents.parquet` | Documents data (18.7 MB) |
| `analytics/data/chunks.parquet` | Chunks data (18.5 MB) |

**Usage:**
```bash
python -m analytics.convert    # Convert JSONL to Parquet (one-time)
python -m analytics.cli        # Launch interactive SQL shell
python -m analytics.cli --query "SELECT * FROM data_summary;"
```

**Available Views:** `data_summary`, `extraction_quality`, `chunks_by_section`, `token_distribution`, `chunks_enriched`, `timeline_by_decade`, `timeline_by_year`

### Embedding Pipeline (Phase 5)
| Path | Purpose |
|------|---------|
| `embeddings/__init__.py` | Package marker |
| `embeddings/config.py` | Model registry, paths, batch sizes |
| `embeddings/models.py` | Model wrapper with dimension validation |
| `embeddings/storage.py` | Parquet read/write for embeddings |
| `embeddings/embed.py` | Embedding generation pipeline |
| `embeddings/upload.py` | Qdrant upload with retry logic |
| `embeddings/cli.py` | CLI entry point |
| `embeddings_data/` | Local Parquet embeddings (gitignored) |

**Usage:**
```bash
python -m embeddings.cli embed both     # Generate embeddings
python -m embeddings.cli upload both    # Upload to Qdrant
python -m embeddings.cli all            # Full pipeline
python -m embeddings.cli stats          # Show statistics
```

### Benchmark Framework (Phase 5)
| Path | Purpose |
|------|---------|
| `eval/__init__.py` | Package marker |
| `eval/gold_queries.yaml` | 50 stratified test queries with ground truth |
| `eval/benchmark.py` | Benchmark runner with metrics and stats |
| `eval/benchmark_report.md` | Generated comparison report |
| `eval/results/` | JSON and Parquet benchmark outputs |

**Usage:**
```bash
python -m eval.benchmark run            # Benchmark both models
python -m eval.benchmark run -m minilm  # Single model
python -m eval.benchmark report         # Generate comparison report
python -m eval.benchmark validate       # Validate query definitions
```

### Documentation
| Path | Purpose |
|------|---------|
| `README.md` | Public project summary |
| `CLAUDE.md` | This file - Claude Code context |
| `PHASE5_PLAN.md` | Phase 5 implementation plan |
| `extraction/README.md` | Extraction pipeline documentation |
| `analytics/README.md` | DuckDB analytics engine docs |
| `eval/README.md` | Benchmark framework documentation |
| `scraper/readme.md` | Scraper library API docs |

### Logs
| Path | Purpose |
|------|---------|
| `logs/` | All pipeline logs (gitignored) |
| `logs/extract_initial_*.log` | Phase 3 Pass 1 (embedded text) logs |
| `logs/extract_ocr_retry_*.log` | Phase 3 Pass 2 (OCR) logs |
| `logs/chunk_all_*.log` | Phase 4 chunking pipeline logs |
| `logs/embed_embed_*.log` | Phase 5 embedding generation logs |
| `logs/embed_upload_*.log` | Phase 5 Qdrant upload logs |
