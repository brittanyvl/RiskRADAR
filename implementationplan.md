# NTSB Aviation Semantic Search + Safety Taxonomy (Portfolio Project)

A portfolio-grade, end-to-end data + ML project that turns unstructured NTSB aviation accident PDFs into searchable, structured, explainable safety insights using embeddings, a vector database, and a lightweight hierarchical taxonomy.

## Goals

Demonstrate ability to:
- Scrape PDFs and manage raw file storage with lineage
- Extract and store rich metadata tied to each file
- OCR and text extraction with quality tracking
- Chunk documents reliably for retrieval + analysis
- Generate embeddings and build a vector search index
- Produce an interpretable “hierarchical cause map” (weakly supervised / rule + embedding signals)
- Ship a simple application (Streamlit) that supports search + insights

## Non-Goals

- Training a deep supervised model with large human-labeled datasets
- Predicting future accidents or making causal claims
- Building a multi-tenant SaaS (auth/billing/etc.) for production
- Perfect extraction of every field from every report (focus on robust pipeline + measurable quality)

---

## Proposed End Product

### App 1: Semantic Search
- Search NTSB reports by meaning (not just keywords)
- Filter by metadata (date, aircraft, location, report type)
- Show “best match excerpt” and source citation

### App 2: Safety Taxonomy Explorer
- A small, manually-defined cause hierarchy (20–30 nodes)
- Probabilistic scoring per node for each report (evidence-backed)
- Trend views over time (theme prevalence and drift)

---

## High-Level Architecture

1. **Ingestion**
   - Discover report URLs → download PDFs → store raw files
   - Compute file hash for dedupe, store file metadata

2. **Metadata**
   - Extract/derive: report id, dates, location, aircraft, report type
   - Persist in relational tables for filtering and analysis

3. **Text Extraction**
   - Extract embedded text when available
   - OCR scanned pages (with confidence/quality metrics)

4. **Chunking**
   - Section-aware chunking where possible
   - Token/length-based chunks with overlap
   - Chunk-level metadata (page range, section guess, quality flags)

5. **Embeddings + Vector Index**
   - Embed each chunk
   - Store embeddings in a vector DB (with chunk metadata)
   - Support metadata filters + semantic retrieval

6. **ML / Analytics Layer**
   - Build hierarchical taxonomy
   - Generate weak signals (rules + embedding similarity to seed examples)
   - Roll scores up the taxonomy tree
   - Produce report-level cause distributions and trend metrics

7. **Application**
   - Streamlit UI for search + taxonomy explorer
   - Exportable artifacts (CSV summaries, trend charts)

---

## Implementation Phases (Milestones)

### Phase 0 — Repo & Foundations (Day 1)
**Deliverables**
- Repo scaffold + tooling
- Minimal CLI entrypoints
- Config + secrets strategy
- CI checks (lint/test)

**Tasks**
- Create `.env.example`, `pyproject.toml` or `requirements.txt`
- Add Makefile / task runner commands (optional)
- Add logging setup and structured config (`yaml`/`toml`)
- Add pre-commit hooks (ruff/black, optional)

---

### Phase 1 — PDF Scraping + Raw Storage (MVP Ingestion)
**Deliverables**
- Repeatable downloader that stores PDFs and captures provenance
- Metadata captured from webpage and stored in SQLite
- Dedupe by filename or hash
- Persistent storage layout on NAS

**Data Source**
- URL: https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx
- Total: 781 Aviation accident report PDFs
- Date range: 1966 to present
- PDF URL pattern: `https://www.ntsb.gov/investigations/AccidentReports/Reports/{FILENAME}.pdf`

**Page Interaction Requirements**
1. Navigate to reports page
2. Click "Aviation" filter button (CRITICAL - must filter before scraping)
3. Page displays 10 reports per page after filtering
4. Paginate through ~79 pages to collect all 781 reports
5. Target PDF icon links (not text/HTML links)

**Metadata Extraction (from webpage, NOT from PDF)**
Each report entry on the page contains metadata that must be captured during scraping:
| Field | Example |
|-------|---------|
| title | "Crash of Pan American World Airways Boeing 727" |
| location | "Berlin, State not available" |
| accident_date | 11/14/1966 |
| report_date | 6/4/1968 |
| report_number | AAR-68-AH |
| pdf_url | Full URL to PDF file |
| filename | Original PDF filename (e.g., AIR2507.pdf) |

This metadata is only available on the webpage and must be captured while scraping to link it with the downloaded files.

**Tasks**
- Implement Selenium scraper using `scraper/` library
- Click Aviation filter, wait for page update
- For each report on page:
  - Extract metadata (title, location, accident_date, report_date, report_number)
  - Extract PDF URL and filename from PDF icon link
  - Download PDF (keep original filename, do not rename)
  - Move to NAS
  - Insert metadata row into SQLite
- Handle pagination (next page button)
- Implement resume capability (skip already-downloaded by filename)
- Add delays between downloads to respect rate limits

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

**Storage**
- PDFs stored on NAS (not in project directory due to laptop space constraints)
- Path: `\\TRUENAS\Photos\RiskRADAR\{original_filename}`
- Keep original filenames - do not rename
- Temp downloads: `.scraper_tmp_downloads/` (intermediate only)

**Acceptance Criteria**
- Can ingest all 781 Aviation PDFs reproducibly
- All metadata captured from webpage and stored in SQLite
- Re-run does not re-download duplicates (checks by filename)
- Scraper handles pagination automatically
- Errors logged and recoverable (resume from failure point)

---

### Phase 2 — Metadata Extraction + Relational Model
**Deliverables**
- Metadata table(s) keyed by `report_id`
- Basic fields usable for filtering in app

**Tasks**
- Parse metadata from filename/HTML/embedded PDF metadata where available
- Create canonical `report_id`
- Store in DB (SQLite initially, Postgres later)

**Acceptance Criteria**
- At least 3–5 metadata fields reliably populated for majority of docs
- Can query “reports between dates”, “by aircraft make/model”, etc.

---

### Phase 3 — Text Extraction + OCR with Quality Metrics
**Deliverables**
- Page-level text extraction results
- OCR pipeline for scanned PDFs
- Quality tracking with heuristics

**Per-Page Workflow**
1. Attempt embedded text extraction using pypdf
2. Compute quality heuristics:
   - `char_count` - total characters extracted
   - `alphabetic_ratio` - % of alphabetic characters
   - `garbage_ratio` - % of garbage/unrecognized characters
3. If below threshold or empty → run OCR on that page
4. Persist per-page record with all metrics

**Quality Thresholds**
TBD - research best practices during implementation

**Libraries**
- PDF parsing: pypdf (preferred for simplicity)
- OCR: pytesseract + pdf2image
- Optional cleanup: opencv-python (only if OCR quality becomes a problem)

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

**Tasks**
- Implement page-by-page text extraction
- Compute quality heuristics per page
- Apply OCR fallback when quality is low
- Persist to SQLite pages table
- Log extraction method and quality for each page

**Acceptance Criteria**
- For sampled PDFs, text is readable and attributable to pages
- Quality metrics computed for every page
- OCR failures are logged and recoverable
- Each page has lineage to report_id

---

### Phase 4 — Chunking (Search-Ready Text)
**Deliverables**
- Chunk dataset with chunk-level metadata
- Chunking strategy documented and testable

**Chunking Parameters**
- **Token limits:** 400-700 tokens per chunk
- **Overlap:** 20% between consecutive chunks
- **Strategy:** Section-aware chunking preferred

**Tasks**
- Implement chunking pipeline:
  - Section detection (heuristic headings)
  - Paragraph preservation
  - Overlap windowing (20%)
  - Token counting (400-700 target range)
- Store chunks to SQLite with full metadata

**Database Schema (chunks table)**
```sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section_guess TEXT,
    chunk_text TEXT,
    chunk_len INTEGER,  -- token count
    text_source TEXT,   -- 'embedded' or 'ocr'
    quality_flags TEXT,
    FOREIGN KEY (report_id) REFERENCES reports(filename)
);
```

**Acceptance Criteria**
- Chunks are coherent (not mid-sentence fragments)
- Each chunk is 400-700 tokens with 20% overlap
- Each chunk has traceability back to report + page range
- Section boundaries respected where detectable

---

### Phase 5 — Embeddings + Vector Database + Benchmark
**Deliverables**
- Embeddings generated for all chunks using two models
- Qdrant Cloud populated with two collections (one per model)
- Benchmark comparing both models
- `eval/benchmark_report.md` with results

**Embedding Models to Benchmark (Locked)**
| Model | Purpose |
|-------|---------|
| `sentence-transformers/all-MiniLM-L6-v2` | Baseline - general purpose |
| `NASA-AIML/MIKA_Custom_IR` | Domain IR - aerospace/aviation specialized |

**Benchmark Contract**
- Both models embed the exact same chunk set
- Both are indexed into separate Qdrant Cloud collections
- App allows switching between collections/models to compare results
- Final production app uses winning model

**Vector Database: Qdrant Cloud**
- Deployment: Qdrant Cloud (not local)
- Secrets: Managed via Streamlit secrets manager
- Collections: Two separate collections (one per model)

**Vector Payload Schema**
Each point stores:
| Field | Type | Notes |
|-------|------|-------|
| `chunk_id` | string | Unique chunk identifier |
| `report_id` | string | Links to reports table |
| `text` | string | Optional - can keep only in SQLite |
| `page_start` | int | Starting page number |
| `page_end` | int | Ending page number |
| `section_guess` | string | Detected section name |
| `accident_date` | string | From webpage scraping |
| `report_date` | string | From webpage scraping |
| `aircraft_make` | string | Nullable |
| `aircraft_model` | string | Nullable |
| `location` | string | Nullable |
| `text_source` | string | `embedded` or `ocr` |
| `quality_flags` | list/string | Quality indicators |

**Benchmarking Requirements**
Create `eval/` folder with:

A) Search relevance benchmark:
- `eval/gold_queries.yaml` (20-30 queries)
- For each model, store: top-k results, human judgment notes

B) Performance benchmark:
- Embed throughput (chunks/sec)
- Qdrant indexing time
- Query latency (rough timing)
- Storage footprint (vector count, dimensions)

**Tasks**
- Set up Qdrant Cloud account and collections
- Implement embedding pipeline for both models
- Batch embed all chunks with both models
- Index into separate Qdrant collections
- Implement vector search with metadata filters
- Create `eval/gold_queries.yaml` with 20-30 test queries
- Run benchmark and document results
- Produce `eval/benchmark_report.md`

**Acceptance Criteria**
- Both models successfully embed all chunks
- Both Qdrant collections queryable
- Benchmark report comparing MiniLM vs MIKA
- Can run semantic search and retrieve relevant chunks
- Roll-up retrieval from chunk → report is implemented (top-k)

---

### Phase 6 — Hierarchical Taxonomy + Weak Signals (Blended #1 + #2)
**Deliverables**
- Cause hierarchy (small, curated) in `taxonomy.yaml`
- Per-report probabilistic scoring across the hierarchy
- Evidence excerpts supporting each score
- Human-in-the-loop taxonomy building process

**Taxonomy Structure**
- File: `taxonomy.yaml`
- Size: 20-30 nodes maximum
- Depth: 2-3 levels
- Human-in-the-loop curation

**Scoring Method**
1. **Weighted keyword rules** - pattern matching against chunk text
2. **Embedding similarity** - compare chunks to seed examples per leaf node
3. **Soft scores** - produce 0-1 range for each leaf
4. **Roll-up** - deterministic aggregation to parent nodes

**Database Schema**
```sql
CREATE TABLE taxonomy_scores (
    report_id TEXT,
    node_path TEXT,
    score REAL,
    evidence_chunk_ids_json TEXT,
    PRIMARY KEY (report_id, node_path),
    FOREIGN KEY (report_id) REFERENCES reports(filename)
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

**Tasks**
- Define taxonomy structure (20-30 nodes, depth 2-3)
- Create `taxonomy.yaml` with node definitions
- Create seed phrases/examples per leaf node
- Build weak signal scoring:
  - Weighted keyword/rule matches
  - Embedding similarity to seed examples
- Implement score roll-up to parents
- Store per-report scores in `taxonomy_scores` table
- Compute trend aggregates in `taxonomy_trends` table

**Acceptance Criteria**
- For a sample set, scores look plausible and are evidence-backed
- "Uncertain" and "mixed cause" cases are handled gracefully
- Scores roll up correctly to parent nodes
- Evidence chunk IDs link back to source text

---

### Phase 7 — Trend + Prevalence Analytics
**Deliverables**
- Time-based prevalence metrics per taxonomy node
- Aggregated data in `taxonomy_trends` table
- Simple drift / change detection (optional)

**Data Source**
Uses `taxonomy_trends` table from Phase 6:
```sql
-- taxonomy_trends(node_path, period, prevalence, mean_score, n_reports)
```

**Tasks**
- Define prevalence metrics:
  - % of reports where node_score >= threshold
  - Mean node_score per time period
- Build aggregates by month/quarter/year
- Populate `taxonomy_trends` table
- Produce summary tables and charts for app

**Acceptance Criteria**
- Can answer: "which themes increased over time?"
- Charts are reproducible from `taxonomy_trends` table
- Trends computed for all taxonomy nodes

---

### Phase 8 — Streamlit Application (Portfolio UI)
**Deliverables**
- Streamlit app with three pages:
  - Semantic Search
  - Cause Map Explorer
  - Analysis Dashboard

**Secrets Management**
- Qdrant Cloud credentials via Streamlit secrets manager
- No hardcoded API keys

**Page 1: Semantic Search**
- Query box for natural language search
- Metadata filters (date, aircraft, location, report type)
- Results list with relevance ranking
- Best match excerpt + source citation
- Link to full report

**Page 2: Cause Map Explorer**
- Browse taxonomy tree (hierarchical navigation)
- Per-node display:
  - Description (from `taxonomy.yaml`)
  - Top reports for that node
  - Evidence snippets from chunks
  - Trend chart over time (from `taxonomy_trends`)

**Page 3: Analysis Dashboard**
- Time series visualizations
- Theme prevalence over time
- Insights and trends
- Export capabilities (CSV downloads)

**Model Comparison (Benchmark Phase)**
During benchmark evaluation, app allows switching between:
- MiniLM collection
- MIKA collection
to compare search results side-by-side

**Tasks**
- Set up Streamlit project structure
- Implement Page 1: Semantic Search
- Implement Page 2: Cause Map Explorer
- Implement Page 3: Analysis Dashboard
- Configure Streamlit secrets for Qdrant Cloud
- Add CSV export functionality
- Add model/collection switching for benchmark comparison

**Acceptance Criteria**
- Hiring manager can run app locally with one command
- App clearly showcases pipeline outputs and interpretability
- All three pages functional and polished
- Secrets properly managed (no hardcoded keys)

---

### Future Roadmap (Post-MVP)
- **User PDF upload:** Allow users to upload their own NTSB PDF and find similar content via semantic search (explore later)

---

## Tech Stack (Locked)

- **Language:** Python 3.9+
- **Scraping:** Selenium + webdriver-manager
- **PDF parsing:** pypdf
- **OCR:** pytesseract + pdf2image (optional: opencv-python for cleanup)
- **Storage:** Local filesystem (NAS) + SQLite
- **Vector DB:** Qdrant Cloud
- **Embeddings:** sentence-transformers
  - Baseline: `sentence-transformers/all-MiniLM-L6-v2`
  - Domain IR: `NASA-AIML/MIKA_Custom_IR`
- **App:** Streamlit + secrets manager
- **Packaging:** pip

---

## Repo Structure (Suggested)

