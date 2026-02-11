# RiskRADAR - Implementation Archive

This document contains historical implementation details, completed phase documentation, and database schemas for reference. For active project context, see `CLAUDE.md`.

---

## Table of Contents

1. [Phase 0 - Foundations](#phase-0--foundations)
2. [Phase 1 - PDF Scraping](#phase-1--pdf-scraping--raw-storage)
3. [Phase 2 - Metadata](#phase-2--metadata-extraction)
4. [Phase 3 - Text Extraction](#phase-3--text-extraction--ocr)
5. [Phase 4 - Chunking](#phase-4--chunking)
6. [Phase 5 - Embeddings](#phase-5--embeddings--vector-database)
7. [Phase 5 v2 Improvement Plan](#phase-5-v2-improvement-plan)
8. [Phase 6A - Taxonomy Classification](#phase-6a--taxonomy-classification)
9. [Phase 6A-Sub - L2 Classification](#phase-6a-sub--l2-classification)
10. [Phase 7-8 - Future Phases](#phase-7--trend-analytics)
11. [Database Schemas](#database-schemas)
12. [Scraper Library Reference](#scraper-library-reference)

---

## Goals & Non-Goals

### Goals
- Scrape PDFs and manage raw file storage with lineage
- Extract and store rich metadata tied to each file
- OCR and text extraction with quality tracking
- Chunk documents reliably for retrieval + analysis
- Generate embeddings and build a vector search index
- Produce an interpretable "hierarchical cause map" (weakly supervised / rule + embedding signals)
- Ship a Streamlit application that supports search + insights

### Non-Goals
- Training a deep supervised model with large human-labeled datasets
- Predicting future accidents or making causal claims
- Building a multi-tenant SaaS (auth/billing/etc.) for production
- Perfect extraction of every field from every report

---

## Phase 0 — Foundations

**Status:** Complete

**Deliverables:** Repo scaffold, tooling, config, CI checks

**Tasks:**
- Create `.env.example`, `requirements.txt`
- Add logging setup and structured config
- Add pre-commit hooks (ruff/black, optional)

---

## Phase 1 — PDF Scraping + Raw Storage

**Status:** Complete (2026-01-06)

**Data Source**
- **URL:** https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx
- **Downloaded:** 510 Aviation accident report PDFs
- **Date range:** 1966 to present

**Results:**
- 510 PDFs successfully downloaded to NAS
- 97% have titles, 96% have accident dates, 100% have report numbers
- All metadata stored in `reports` table

### robots.txt Compliance
Checked: https://www.ntsb.gov/robots.txt
- `/investigations/AccidentReports/` is **ALLOWED** (not in Disallow list)
- No `Crawl-delay` specified, so we implement our own

### Rate Limiting
| Setting | Value | Purpose |
|---------|-------|---------|
| `SCRAPER_REQUEST_DELAY` | 2.0s | Between any requests |
| `SCRAPER_DOWNLOAD_DELAY` | 3.0s | Between PDF downloads |
| `SCRAPER_PAGE_DELAY` | 2.0s | Between pagination clicks |

### Page Selectors
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

### Metadata Captured
| Field | Example |
|-------|---------|
| title | "Crash of Pan American World Airways Boeing 727" |
| location | "Berlin, State not available" |
| accident_date | 11/14/1966 |
| report_date | 6/4/1968 |
| report_number | AAR-68-AH |
| pdf_url | Full URL to PDF file |
| filename | Original PDF filename (e.g., AIR2507.pdf) |

---

## Phase 2 — Metadata Extraction

**Status:** Complete (metadata captured during scraping)

**Deliverables:** Metadata tables keyed by `report_id`, fields usable for filtering

**Acceptance Criteria:**
- At least 3–5 metadata fields reliably populated for majority of docs
- Can query "reports between dates", "by aircraft make/model", etc.

---

## Phase 3 — Text Extraction + OCR

**Status:** Complete

**Per-Page Workflow:**
1. Attempt embedded text extraction using pymupdf
2. Compute quality heuristics:
   - `char_count` - total characters extracted
   - `alphabetic_ratio` - % of alphabetic characters
   - `garbage_ratio` - % of garbage/unrecognized characters
3. If below threshold or empty → run OCR (pytesseract + pdf2image)

**Results:**
- 30,602 pages extracted
- 14,000 pages with embedded text
- 16,000 pages required OCR

**Key Files:**
| Path | Purpose |
|------|---------|
| `extraction/processing/extract.py` | Extraction pipeline orchestration |
| `extraction/processing/pdf_reader.py` | Embedded text extraction (pymupdf) |
| `extraction/processing/ocr.py` | OCR with confidence scoring (pytesseract) |
| `extraction/processing/quality.py` | Quality thresholds and metrics |
| `extraction/json_data/passed/` | Pages that passed quality checks |
| `extraction/json_data/ocr_retry/` | OCR re-extraction results |

---

## Phase 4 — Chunking

**Status:** Complete (v2)

**Three-Pass Pipeline:**
```
Pass 0: JSON files → pages.jsonl (consolidated, deduplicated, ordered)
Pass 1: pages.jsonl → documents.jsonl (per-report full text)
Pass 2: documents.jsonl → chunks.jsonl (search-ready segments)
```

**CLI Commands:**
```bash
python -m extraction.processing.chunk all
python -m extraction.processing.chunk pages      # Pass 0
python -m extraction.processing.chunk documents  # Pass 1
python -m extraction.processing.chunk chunks     # Pass 2
```

**v2 Parameters:**
| Parameter | Value |
|-----------|-------|
| Minimum tokens | 400 |
| Maximum tokens | 800 |
| Target tokens | 600 |
| Overlap | 25% |
| Tokenizer | tiktoken cl100k_base |

**Output Files:**
| File | Records | Size |
|------|---------|------|
| `pages.jsonl` | 30,602 | ~69 MB |
| `documents.jsonl` | 510 | ~68 MB |
| `chunks_v2.jsonl` | 24,766 | ~90 MB |

**Results:**
- 24,766 chunks from 510 documents
- Avg 632 tokens/chunk, median 640 tokens
- Token distribution: 2.3% under 400, 95.6% in range (400-800), 2.1% over 800
- Section detection: 95% pattern match, 5% paragraph fallback
- Footnotes appended to 1,297 chunks

**Key Files:**
| Path | Purpose |
|------|---------|
| `extraction/processing/chunk.py` | CLI entry point |
| `extraction/processing/consolidate_pages.py` | Pass 0: JSON → pages.jsonl |
| `extraction/processing/consolidate.py` | Pass 1: pages → documents.jsonl |
| `extraction/processing/section_detect.py` | Section header detection |
| `extraction/processing/toc_detect.py` | TOC page detection |
| `extraction/processing/footnote_parse.py` | Footnote extraction |
| `extraction/processing/tokenizer.py` | tiktoken cl100k wrapper |

---

## Phase 5 — Embeddings + Vector Database

**Status:** Complete (v2)

**Embedding Models:**
| Model | Dimensions | Collection Name |
|-------|------------|-----------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | `riskradar_minilm` |
| `NASA-AIML/MIKA_Custom_IR` | 768 | `riskradar_mika` |

**CLI Commands:**
```bash
python -m embeddings.cli embed minilm
python -m embeddings.cli embed mika
python -m embeddings.cli embed both
python -m embeddings.cli upload both
python -m embeddings.cli all
python -m embeddings.cli verify minilm
python -m embeddings.cli stats
python -m embeddings.cli enrich both --l1-run 1 --l2-run 1
```

**Benchmark Framework (50 queries):**
| Category | Count | Difficulty |
|----------|-------|------------|
| Incident Lookup | 10 | Easy |
| Conceptual Queries | 12 | Medium-Hard |
| Section Queries | 10 | Medium |
| Comparative Queries | 8 | Hard |
| Aircraft Queries | 6 | Medium |
| Phase Queries | 4 | Medium |

**Benchmark Metrics:**
- MRR (Mean Reciprocal Rank)
- Hit@K (K = 1, 3, 5, 10, 20)
- Precision@K, Recall@K
- nDCG@10
- Section Accuracy
- Latency
- Statistical tests (paired t-test, Wilcoxon, bootstrap CI)

**v1 Benchmark Results (2026-01-18):**
| Model | MRR | Hit@10 | Semantic Precision | False Positive Rate |
|-------|-----|--------|-------------------|---------------------|
| MiniLM | 0.669 | 94.9% | **75.5%** | **16.4%** |
| MIKA | 0.788 | 94.9% | 60.0% | 34.5% |

**Key Finding:** Human evaluation showed MiniLM outperforms MIKA despite MIKA's better automated metrics. MIKA has 2x the false positive rate.

**Key Files:**
| Path | Purpose |
|------|---------|
| `embeddings/config.py` | Model registry, paths, batch sizes |
| `embeddings/models.py` | Model wrapper with dimension validation |
| `embeddings/storage.py` | Parquet read/write for embeddings |
| `embeddings/embed.py` | Embedding generation pipeline |
| `embeddings/upload.py` | Qdrant upload with retry logic |
| `embeddings/cli.py` | CLI entry point |
| `eval/gold_queries.yaml` | 50 stratified test queries |
| `eval/benchmark.py` | Benchmark runner |

---

## Phase 5 v2 Improvement Plan

### Problems Identified in v1

1. **Hierarchical section bug:** "1. The Accident" and "1.1 History of Flight" treated as siblings
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
```

**Qdrant Collections:**
| Version | MiniLM Collection | MIKA Collection |
|---------|-------------------|-----------------|
| v1 | `riskradar_minilm_v1` | `riskradar_mika_v1` |
| v2 | `riskradar_minilm_v2` | `riskradar_mika_v2` |

### Implementation Stages (All Complete)

#### Stage 0: Version Current Artifacts
- [x] Rename `chunks.jsonl` → `chunks_v1.jsonl`
- [x] Backup v1 artifacts
- [x] Chunking config updated

#### Stage 1: Chunking Fixes
- [x] Hierarchical section merging
- [x] 400 token minimum with forward borrowing
- [x] Section prefix [SECTION_NAME]
- [x] 25% overlap across section boundaries
- [x] Protected sentence splitting
- [x] Unit tests passing (25 tests)

#### Stage 2: Re-Process Everything
- [x] 24,766 chunks generated (95.6% in range)
- [x] Chunks under 400: 566 (2.3%) - major improvement from 8.8%
- [x] MiniLM embeddings (56.3 MB)
- [x] MIKA embeddings (112.1 MB)
- [x] Both collections uploaded (24,766 vectors each)

### Rollback Strategy

```python
# In config.py - single line rollback
ACTIVE_VERSION = "v1"

# Or via environment variable
RISKRADAR_VERSION=v1 python -m streamlit run app.py
```

---

## Phase 6A — Taxonomy Classification

**Status:** Complete

**Pivot from BERTopic:** Initial experiments with BERTopic topic modeling produced poor results - 76 topics discovered mostly captured document structure (headers, boilerplate) rather than meaningful safety factors.

### PROBABLE CAUSE Section Consistency
| Section Type | Reports | Coverage |
|--------------|---------|----------|
| PROBABLE CAUSE | 155 | 30.4% |
| CONCLUSIONS | 255 | 50.0% |
| FINDINGS | 278 | 54.5% |
| ANALYSIS | 358 | **70.2%** |
| **Any cause section** | 411 | **80.6%** |

### L1 Classification Approach
1. Filter chunks to causal sections (PROBABLE CAUSE, ANALYSIS, CONCLUSIONS, FINDINGS)
2. Load pre-computed MIKA embeddings (768-dim)
3. Compute category embeddings from seed phrases (averaged)
4. Cosine similarity matching to 27 CICTT categories
5. Aggregate chunk assignments to report level with weighted scoring

### Results (Run 1)
- 453 reports classified
- 5,806 causal chunks analyzed
- 6,555 chunk-category assignments
- 1,736 report-category assignments
- 27 categories used

### CLI Commands
```bash
python -m taxonomy.cli map
python -m taxonomy.cli categories
python -m taxonomy.cli stats
python -m taxonomy.cli review
```

---

## Phase 6A-Sub — L2 Classification

**Status:** Complete

### Two-Level Hierarchy
```
Level 1: CICTT Categories (27 categories)
  └─ Level 2: Industry-standard subcategories
       - LOC-I subcategories from IATA LOC-I Analysis
       - CFIT subcategories from IATA CFIT Analysis
       - HFACS for human factors (cross-cutting)
       - Technical subcategories from FAA/EASA
```

### L2-Enabled Categories
| L1 Code | L1 Name | Subcategories |
|---------|---------|---------------|
| LOC-I | Loss of Control - Inflight | STALL, UPSET, SD, ENV, SYS, LOAD |
| CFIT | Controlled Flight Into Terrain | NAV, SA, VIS, TAWS, PROC |
| SCF-PP | System/Component Failure - Powerplant | ENG, FUEL, PROP, FIRE |
| SCF-NP | System/Component Failure - Non-Powerplant | FLT, HYD, ELEC, STRUCT, GEAR |
| ICE | Icing | STRUCT, INDUCT, PITOT |
| FUEL | Fuel Related | EXHAUST, STARVE, CONTAM |
| WSTRW | Windshear/Thunderstorm | MICRO, TSTORM |
| HFACS | Human Factors (cross-cutting) | SKILL, DECISION, PERCEPTUAL, VIOLATION |

### Results (Run 1)
- 1,478 chunks processed
- 2,446 chunk-L2 assignments
- 1,106 report-L2 assignments
- 32 subcategories used
- Processing time: 11 seconds

### Configuration
| Parameter | L1 Value | L2 Value |
|-----------|----------|----------|
| Min similarity | 0.40 | 0.35 |
| Top-k per chunk | 3 | 2 |
| Max per report | 5 | 3 per parent |
| Min combined confidence | - | 0.20 |

### CLI Commands
```bash
python -m taxonomy.cli classify
python -m taxonomy.cli classify --l1-only
python -m taxonomy.cli subcategories
python -m taxonomy.cli export-review --sample-size 50
python -m taxonomy.cli import-review FILE
python -m taxonomy.cli review-stats
```

### Key Files
| File | Purpose |
|------|---------|
| `taxonomy/subcategories.py` | 32 subcategory definitions with seed phrases |
| `taxonomy/hierarchical_mapper.py` | Two-pass classification logic |
| `taxonomy/pipeline.py` | End-to-end orchestration |
| `taxonomy/config.py` | L1 and L2 thresholds and settings |

### References
- CICTT Aviation Occurrence Categories v4.7
- IATA (2015, 2019). LOC-I Analysis Reports
- IATA (2018). CFIT Analysis Report
- Shappell & Wiegmann (2000). HFACS DOT/FAA/AM-00/7
- EASA (2024). LOC-I Prevention Guidance

---

## Phase 6A - Qdrant Payload Enrichment

**Status:** Complete

**Enrichment Results:**
- minilm collection: 24,766 points updated
- mika collection: 24,766 points updated
- Reports with taxonomy: 418
- Reports without taxonomy: 92 (unclassified)
- Processing time: ~74 minutes

**New Payload Fields:**
```json
{
  "l1_categories": ["LOC-I", "SCF-NP", "CFIT"],
  "l2_subcategories": ["LOC-I-STALL", "CFIT-NAV"],
  "pdf_url": "https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR0201.pdf"
}
```

---

## Phase 7 — Trend Analytics

**Status:** Planned

**Prevalence Definition:** % of reports where cause score >= 10%

**Aggregation Periods:**
- By year (1966-2026)
- By decade (1960s, 1970s, ...)
- By quarter (optional)

**Expected Patterns:**
- Human factors prevalence over time
- Technology-related causes by era (automation issues post-1990s)
- Weather-related trends by decade

---

## Phase 8 — Streamlit Application

**Status:** Planned

**Three Pages:**

**Page 1: Semantic Search**
- Query box for natural language search
- Taxonomy filters (dropdown/checkboxes for L1/L2 categories)
- Metadata filters (date range, location, aircraft type)
- Results list with relevance ranking
- Best match excerpt + source citation
- Cause attribution badges on results

**Page 2: Cause Map Explorer**
- Interactive tree view of 3-level taxonomy
- Per-node display: description, keywords, top 10 reports, evidence snippets, trend chart
- Click-through to view full reports
- Compare MiniLM vs MIKA assignments

**Page 3: Analysis Dashboard**
- Time series: Cause prevalence by decade
- Stacked area chart: Cause distribution over time
- Heatmap: Cause co-occurrence matrix
- Top causes by aircraft type
- Export capabilities (CSV downloads)

**Module Structure:**
```
app/
├── __init__.py
├── main.py
├── pages/
│   ├── search.py
│   ├── taxonomy.py
│   └── dashboard.py
├── components/
│   ├── taxonomy_tree.py
│   ├── cause_badges.py
│   └── trend_chart.py
└── utils/
    ├── qdrant_client.py
    ├── taxonomy_queries.py
    └── trend_queries.py
```

---

## Database Schemas

### reports table
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

### pages table
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

### chunks table
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

### taxonomy_trends table (Phase 7)
```sql
CREATE TABLE taxonomy_trends (
    id INTEGER PRIMARY KEY,
    node_code TEXT NOT NULL,
    period TEXT NOT NULL,
    period_type TEXT NOT NULL,
    prevalence REAL NOT NULL,
    mean_percentage REAL NOT NULL,
    n_reports INTEGER NOT NULL,
    n_with_cause INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    taxonomy_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (node_code, period, model_name, taxonomy_version)
);
```

### Taxonomy Tables (Schema v5)
- `taxonomy_runs` - Classification run tracking
- `taxonomy_chunk_l1` - Chunk to L1 assignments
- `taxonomy_report_l1` - Report-level L1 categories
- `taxonomy_chunk_l2` - Chunk to L2 assignments
- `taxonomy_report_l2` - Report-level L2 subcategories
- `taxonomy_reviews` - Human review decisions
- `taxonomy_errors` - Error logging

---

## Scraper Library Reference

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
| `SCRAPER_HEADLESS` | `1` | Headless mode |
| `SCRAPER_DOWNLOADS_DIR` | `.scraper_tmp_downloads` | Temp download directory |
| `SCRAPER_DOWNLOAD_TIMEOUT` | `120` | Download timeout (seconds) |
| `HTTP_USER_AGENT` | `RiskRADARBot/1.0` | Browser user agent |
| `SCRAPER_REQUEST_DELAY` | `2.0` | Delay between requests |
| `SCRAPER_DOWNLOAD_DELAY` | `3.0` | Delay between downloads |
| `SCRAPER_PAGE_DELAY` | `2.0` | Delay between page navigations |

### Scraping Workflow
```python
from scraper.config import BrowserConfig
from scraper.browser import chrome
from scraper.actions import go_to, select_dropdown_by_value, click
from scraper.waits import rate_limit
from scraper.download import wait_for_new_download, move_and_rename

config = BrowserConfig()

with chrome(config) as driver:
    go_to(driver, NTSB_URL)
    rate_limit(config.request_delay_sec)

    select_dropdown_by_value(driver, MODE_DROPDOWN, "Aviation")
    rate_limit(config.page_delay_sec)

    while has_more_pages:
        for report in reports_on_page:
            click(driver, pdf_link)
            downloaded = wait_for_new_download(config)
            move_and_rename(downloaded, NAS_PATH, filename)
            rate_limit(config.download_delay_sec)

        click(driver, NEXT_BUTTON)
        rate_limit(config.page_delay_sec)
```

---

## Analytics Engine (DuckDB)

**Usage:**
```bash
python -m analytics.convert    # Convert JSONL to Parquet (one-time)
python -m analytics.cli        # Launch interactive SQL shell
python -m analytics.cli --query "SELECT * FROM data_summary;"
```

**Available Views:**
- `data_summary`
- `extraction_quality`
- `chunks_by_section`
- `token_distribution`
- `chunks_enriched`
- `timeline_by_decade`
- `timeline_by_year`

**Key Files:**
| Path | Purpose |
|------|---------|
| `analytics/convert.py` | JSONL → Parquet conversion |
| `analytics/views.py` | Pre-built analytical views |
| `analytics/cli.py` | Interactive SQL shell |
| `analytics/data/pages.parquet` | Pages data (19.5 MB) |
| `analytics/data/documents.parquet` | Documents data (18.7 MB) |
| `analytics/data/chunks.parquet` | Chunks data (18.5 MB) |
