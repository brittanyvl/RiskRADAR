# extraction - PDF Text Extraction & Chunking Pipeline

Text extraction (Phase 3) and semantic chunking (Phase 4) pipelines for RiskRADAR.

---

## Table of Contents

- [Overview](#overview)
- [Role in Pipeline](#role-in-pipeline)
- [Directory Structure](#directory-structure)
- [Phase 3: Text Extraction](#phase-3-text-extraction)
  - [Pass 1: Initial Extraction](#pass-1-initial-extraction)
  - [Pass 2: OCR Retry](#pass-2-ocr-retry)
  - [Quality Thresholds](#quality-thresholds)
  - [OCR Confidence](#ocr-confidence)
- [Phase 4: Chunking](#phase-4-chunking)
  - [Three-Pass Pipeline](#three-pass-pipeline)
  - [CLI Commands](#cli-commands)
  - [Chunking Parameters (v2)](#chunking-parameters-v2)
  - [Section Detection](#section-detection)
  - [Footnote Handling](#footnote-handling)
- [Output Files](#output-files)
  - [Phase 3 JSON Schema](#phase-3-json-schema)
  - [Chunk JSONL Schema](#chunk-jsonl-schema)
- [Database Tables](#database-tables)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [See Also](#see-also)

---

## Overview

The `extraction` module provides:

- **Multi-pass PDF text extraction** with quality metrics and OCR fallback
- **Section-aware chunking** with header detection and footnote linking
- **Token-controlled chunk sizes** (400-800 tokens) optimized for embedding models
- **Full lineage tracking** from source PDF page to final chunk
- **Quality gates** at every stage to ensure data integrity

This is where raw PDFs become search-ready text chunks. The extraction quality directly impacts downstream embedding and retrieval quality.

---

## Role in Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Phase 3: Text Extraction                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    510 PDFs ──────► pymupdf ──────► Quality Gate ──────► OCR fallback   │
│    (NAS)              │                   │                    │         │
│                       ▼                   ▼                    ▼         │
│                 embedded text        threshold?          pytesseract     │
│                 (47% pages)         char_count          (53% pages)      │
│                       │             alpha_ratio              │           │
│                       │             garbage_ratio            │           │
│                       └───────────────────┬──────────────────┘           │
│                                           ▼                              │
│                                     30,602 pages                         │
│                                    (json_data/)                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        Phase 4: Chunking                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    pages.jsonl ────► documents.jsonl ────► chunks.jsonl                 │
│    (30,602)              (510)               (24,766)                    │
│        │                   │                    │                        │
│        ▼                   ▼                    ▼                        │
│    consolidated       full text           search-ready                   │
│    deduplicated       per report          with sections                  │
│                                           and footnotes                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
extraction/
├── processing/               # Pipeline modules
│   ├── extract.py           # Multi-pass extraction orchestration
│   ├── pdf_reader.py        # Embedded text extraction (pymupdf)
│   ├── ocr.py               # OCR processing (pytesseract)
│   ├── quality.py           # Quality assessment and thresholds
│   ├── analytics.py         # Quality reporting queries
│   ├── chunk.py             # Chunking pipeline CLI entry point
│   ├── consolidate_pages.py # Pass 0: JSON → pages.jsonl
│   ├── consolidate.py       # Pass 1: pages → documents.jsonl
│   ├── section_detect.py    # Section header detection
│   ├── toc_detect.py        # TOC page detection
│   ├── footnote_parse.py    # Footnote extraction
│   └── tokenizer.py         # tiktoken cl100k wrapper
├── scripts/                  # Utility scripts
│   ├── verify_ocr.py        # Verify OCR dependencies
│   ├── install_tesseract.ps1 # Automated Tesseract install (Windows)
│   └── validate_extraction.py # Pipeline validation
├── json_data/                # Pipeline outputs (GITIGNORED)
│   ├── passed/              # Pages that passed quality checks
│   ├── failed/              # Pages that failed (need OCR)
│   ├── ocr_retry/           # OCR re-extraction results
│   ├── pages.jsonl          # Consolidated pages
│   ├── documents.jsonl      # Full documents
│   └── chunks.jsonl         # Search-ready chunks
└── temp/                     # Temporary workspace (deleted after use)
```

---

## Phase 3: Text Extraction

### Pass 1: Initial Extraction

Extracts embedded text from all PDFs using pymupdf. Fast (~35ms/page) but only works for PDFs with embedded text.

```bash
# Process all reports
python -m extraction.processing.extract initial

# Process limited batch (for testing)
python -m extraction.processing.extract initial 10
```

**Output:** JSON files in `json_data/passed/` or `json_data/failed/` based on quality gate.

### Pass 2: OCR Retry

Runs OCR on pages that failed quality checks. Slower (~7-10s/page at 300 DPI) but handles scanned documents.

```bash
python -m extraction.processing.extract ocr
```

**Output:** JSON files in `json_data/ocr_retry/` with confidence scores.

### Full Pipeline

Runs both passes sequentially:

```bash
python -m extraction.processing.extract all
```

**Note:** Use `python` (not `py`) when venv is activated. Requires NAS access for PDF files.

### Quality Thresholds

Pages must meet ALL criteria to pass:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| `char_count` | ≥ 50 | Page has meaningful content |
| `alphabetic_ratio` | ≥ 0.50 | More letters than garbage characters |
| `garbage_ratio` | ≤ 0.15 | Low proportion of unrecognized characters |
| `word_count` | ≥ 10 | At least some words extracted |

Pages failing any threshold are sent to OCR retry.

### OCR Confidence

OCR extractions include word-level confidence scores:

| Range | Quality | Action |
|-------|---------|--------|
| ≥ 80 | Good | Use as-is |
| 60-80 | Acceptable | Use with caution |
| < 60 | Poor | Flagged for review |

```bash
# View quality analytics
python -m extraction.processing.analytics
```

---

## Phase 4: Chunking

### Three-Pass Pipeline

```
Pass 0: JSON files → pages.jsonl (consolidated, deduplicated, ordered)
Pass 1: pages.jsonl → documents.jsonl (per-report full text)
Pass 2: documents.jsonl → chunks.jsonl (search-ready segments)
```

Each pass is independently runnable and resumable.

### CLI Commands

```bash
# Run full pipeline (all three passes)
python -m extraction.processing.chunk all

# Run individual passes
python -m extraction.processing.chunk pages      # Pass 0 only
python -m extraction.processing.chunk documents  # Pass 1 only
python -m extraction.processing.chunk chunks     # Pass 2 only

# Limit number of reports (for testing)
python -m extraction.processing.chunk all --limit 10
```

### Chunking Parameters (v2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Token min | 400 | Minimum tokens per chunk |
| Token target | 600 | Target tokens per chunk |
| Token max | 800 | Maximum tokens per chunk |
| Overlap | 25% | Overlap between consecutive chunks (~150 tokens) |
| Tokenizer | cl100k_base | tiktoken encoding (GPT-4 compatible) |
| Section prefix | Yes | Chunks prefixed with `[SECTION_NAME]` for context |

> **v1 vs v2:** v1 used 500-700 token range with 20% overlap and no section prefixes. v2 expanded to 400-800 with 25% overlap and section prefixes for better retrieval. This improved MRR from 0.788 to 0.816 (+3.5%).

### Section Detection

The pipeline detects NTSB section headers using regex patterns:

| Pattern Type | Example |
|--------------|---------|
| Numbered section | `1.8 METEOROLOGICAL INFORMATION` |
| Standalone header | `SYNOPSIS` |
| Letter subsection | `(a) Findings` |
| Spaced decimal | `1. 8 Aids to Navigation` (OCR artifacts) |

**Detection stats (v2):**
- 99.2% pattern match
- 0.8% paragraph fallback
- <0.1% no structure

When no sections detected, falls back to paragraph-based chunking.

### Footnote Handling

NTSB reports heavily use footnotes. The pipeline:

1. Detects footnote markers in text (e.g., "1/", "2/")
2. Extracts footnote definitions from page bottoms
3. Appends relevant footnotes to chunks that reference them

**v2 stats:** 1,297 chunks (5.2%) have footnotes appended.

---

## Output Files

### Phase 3 JSON Schema

Each page is stored as `{report_id}/{report_base}_page_{number:04d}.json`:

```json
{
  "report_id": "AIR2507.pdf",
  "page_number": 0,
  "extraction_method": "embedded",
  "extraction_pass": "initial",
  "text": "...",
  "quality_metrics": {
    "char_count": 1523,
    "alphabetic_ratio": 0.78,
    "garbage_ratio": 0.02,
    "word_count": 245,
    "passes_threshold": true
  },
  "ocr_confidence": null,
  "metadata": {
    "extracted_at": "2026-01-04T12:34:56",
    "extraction_time_ms": 45,
    "pdf_source": "\\\\TRUENAS\\Photos\\RiskRADAR\\AIR2507.pdf"
  }
}
```

### Chunk JSONL Schema

```json
{
  "chunk_id": "AAR7005.pdf_chunk_0003",
  "report_id": "AAR7005.pdf",
  "chunk_sequence": 3,
  "page_start": 5,
  "page_end": 6,
  "char_start": 8500,
  "char_end": 10200,
  "section_name": "METEOROLOGICAL INFORMATION",
  "section_number": "1.8",
  "section_detection_method": "pattern_match",
  "chunk_text": "...",
  "token_count": 623,
  "text_source": "embedded",
  "has_footnotes": true,
  "footnotes": [{"marker": "1/", "text": "All times are Pacific standard"}],
  "created_at": "2026-01-11T16:40:00"
}
```

### Output File Sizes (v2)

| File | Records | Size | Description |
|------|---------|------|-------------|
| `pages.jsonl` | 30,602 | ~69 MB | One line per page, deduplicated |
| `documents.jsonl` | 510 | ~68 MB | One line per document, full text |
| `chunks.jsonl` | 24,766 | ~85 MB | One line per chunk, search-ready |

---

## Database Tables

### Phase 3 Tables

| Table | Purpose |
|-------|---------|
| `pages` | Per-page extraction status and quality metrics |
| `extraction_runs` | Extraction pipeline execution history |
| `extraction_errors` | Extraction error details with stack traces |

### Phase 4 Tables

| Table | Purpose |
|-------|---------|
| `documents` | Consolidated document metadata |
| `chunks` | Search-ready chunks with lineage |
| `chunking_runs` | Chunking pipeline execution history |
| `chunking_errors` | Chunking error details with stack traces |

```bash
# Check run history
sqlite3 sqlite/riskradar.db "SELECT * FROM extraction_runs ORDER BY id;"
sqlite3 sqlite/riskradar.db "SELECT * FROM chunking_runs ORDER BY id;"
```

---

## Performance

### Phase 3 (Extraction)

| Operation | Time | Notes |
|-----------|------|-------|
| Embedded text | ~35ms/page | Fast, uses pymupdf |
| OCR with confidence | ~7-10s/page | 300 DPI, pytesseract |
| Batch of 10 reports (~450 pages) | ~1.5 hours | Mostly OCR time |

### Phase 4 (Chunking)

| Operation | Time | Notes |
|-----------|------|-------|
| Full pipeline | ~70 seconds | All 510 reports, 30K pages |
| Pass 0 (consolidate pages) | ~7 seconds | |
| Pass 1 (build documents) | ~25 seconds | |
| Pass 2 (chunk documents) | ~35 seconds | |

---

## Troubleshooting

**Resume interrupted run:**
The pipeline automatically resumes - it checks which pages are already extracted.

**Re-process specific report:**
```sql
-- Delete existing pages for a report
DELETE FROM pages WHERE report_id = 'AAR7008.pdf';
```
Then delete the corresponding JSON files and re-run extraction.

**Check run history:**
```bash
sqlite3 sqlite/riskradar.db "SELECT * FROM extraction_runs ORDER BY id;"
```

**NAS connection issues:**
See CLAUDE.md for NAS troubleshooting steps.

**OCR dependency missing:**
```bash
python -m extraction.scripts.verify_ocr
```

---

## Limitations

1. **Tesseract Required**: OCR pass requires Tesseract 4.0+ installed on the system. Windows users can run `install_tesseract.ps1`.

2. **NAS Dependency**: PDFs must be accessible via the configured NAS path. No local PDF storage.

3. **Memory for Large PDFs**: Some 100+ page reports may require significant memory during OCR. Consider processing in batches.

4. **OCR Quality Varies**: Historical documents (1960s-1980s) often have lower OCR confidence due to scan quality.

5. **No Multi-Language**: OCR configured for English only. Reports with non-English content may have lower quality.

6. **Single-Threaded**: Extraction runs single-threaded for predictability. Could be parallelized for speed.

---

## See Also

- [Main README](../README.md) - Project overview
- [sqlite/README.md](../sqlite/README.md) - Database schema (pages, chunks tables)
- [embeddings/README.md](../embeddings/README.md) - Next step: generating embeddings
- [analytics/README.md](../analytics/README.md) - DuckDB queries on chunks data
- [PORTFOLIO.md](../PORTFOLIO.md) - Chunking evolution (v1 vs v2)
