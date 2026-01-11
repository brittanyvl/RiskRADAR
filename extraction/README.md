# PDF Text Extraction & Chunking Pipeline

This directory contains the multi-pass PDF text extraction (Phase 3) and chunking (Phase 4) pipelines for RiskRADAR.

## Directory Structure

- `processing/` - Pipeline modules
  - **Phase 3 (Extraction):**
    - `quality.py` - Quality assessment and thresholds
    - `pdf_reader.py` - Embedded text extraction (pymupdf)
    - `ocr.py` - OCR processing with confidence scores (pytesseract)
    - `extract.py` - Multi-pass extraction orchestration
    - `analytics.py` - Quality reporting and analytics queries
  - **Phase 4 (Chunking):**
    - `chunk.py` - Chunking pipeline CLI entry point
    - `consolidate_pages.py` - Pass 0: JSON → pages.jsonl
    - `consolidate.py` - Pass 1: pages → documents.jsonl
    - `section_detect.py` - Section header detection
    - `toc_detect.py` - TOC page detection
    - `footnote_parse.py` - Footnote extraction
    - `tokenizer.py` - tiktoken cl100k wrapper
- `scripts/` - Extraction-specific scripts
  - `verify_ocr.py` - Verify OCR dependencies installed
  - `install_tesseract.ps1` - Automated Tesseract installation (Windows)
  - `validate_extraction.py` - Pipeline validation script
- `json_data/` - Pipeline outputs (GITIGNORED)
  - **Phase 3 outputs:**
    - `passed/` - Pages that passed quality checks (per-page JSON)
    - `failed/` - Pages that failed embedded extraction (need OCR)
    - `ocr_retry/` - OCR re-extraction results (per-page JSON)
  - **Phase 4 outputs:**
    - `pages.jsonl` - Consolidated pages (one line per page)
    - `documents.jsonl` - Full documents (one line per document)
    - `chunks.jsonl` - Search-ready chunks (one line per chunk)
- `temp/` - Temporary workspace (deleted after quality gate)

## Phase 3: Extraction Pipeline

### Pass 1: Initial Extraction
Extracts embedded text from all PDFs using pymupdf.

```bash
# Process all reports
py -m extraction.processing.extract initial

# Process limited batch (for testing)
py -m extraction.processing.extract initial 10
```

### Pass 2: OCR Retry
Runs OCR on pages that failed quality checks.

```bash
py -m extraction.processing.extract ocr
```

### Full Pipeline
Runs both passes sequentially.

```bash
py -m extraction.processing.extract all
```

**Note:** On Windows, use `py` instead of `python`. The pipeline requires NAS access for PDF files.

---

## Phase 4: Chunking Pipeline

Transforms extracted pages into search-ready chunks for embedding.

### Three-Pass Pipeline

```
Pass 0: JSON files → pages.jsonl (consolidated, deduplicated, ordered)
Pass 1: pages.jsonl → documents.jsonl (per-report full text)
Pass 2: documents.jsonl → chunks.jsonl (search-ready segments)
```

### CLI Commands

```bash
# Run full pipeline (all three passes)
py -m extraction.processing.chunk all

# Run individual passes
py -m extraction.processing.chunk pages      # Pass 0 only
py -m extraction.processing.chunk documents  # Pass 1 only
py -m extraction.processing.chunk chunks     # Pass 2 only

# Limit number of reports (for testing)
py -m extraction.processing.chunk all --limit 10
```

### Chunking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Token target | 600 | Target tokens per chunk |
| Token min | 500 | Minimum tokens (soft limit) |
| Token max | 700 | Maximum tokens (soft limit) |
| Overlap | 20% | Overlap between consecutive chunks (~120 tokens) |
| Tokenizer | cl100k_base | tiktoken encoding |

### Output Files

| File | Records | Size | Description |
|------|---------|------|-------------|
| `pages.jsonl` | 30,602 | ~69 MB | One line per page, deduplicated |
| `documents.jsonl` | 510 | ~68 MB | One line per document, full text |
| `chunks.jsonl` | 28,321 | ~85 MB | One line per chunk, search-ready |

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

### Section Detection

The pipeline detects NTSB section headers using regex patterns:

| Pattern | Example |
|---------|---------|
| Numbered section | `1.8 METEOROLOGICAL INFORMATION` |
| Standalone header | `SYNOPSIS` |
| Letter subsection | `(a) Findings` |
| Spaced decimal | `1. 8 Aids to Navigation` |

When no sections detected, falls back to paragraph-based chunking.

### Analytics

The pipeline logs comprehensive analytics at completion:

```
CHUNK ANALYTICS REPORT
Total chunks: 28,321
Total documents: 510
Total tokens: 15,124,726
Avg tokens/chunk: 534.0
Median tokens/chunk: 658.0
Token distribution:
  Under 500: 9,419 (33.3%)
  In range (500-700): 9,863 (34.8%)
  Over 700: 9,039 (31.9%)
Text source distribution: {'embedded': 12977, 'mixed': 452, 'ocr': 14892}
Section detection: {'pattern_match': 26977, 'paragraph_fallback': 1334}
Chunks with footnotes: 1264
```

---

## Phase 3 JSON Schema

Each page is stored as `{report_id}/{report_base}_page_{number:04d}.json`:

Example: `AAR7008.pdf/AAR7008_page_0001.json`

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

## Quality Thresholds

Pages must meet ALL criteria to pass:
- `char_count >= 50`
- `alphabetic_ratio >= 0.50`
- `garbage_ratio <= 0.15`
- `word_count >= 10`

## OCR Confidence

OCR extractions include word-level confidence scores:
- **Good:** mean_confidence >= 80
- **Acceptable:** 60-80
- **Poor:** < 60 (flagged for review)

## Analytics

View quality statistics:

```bash
py -m extraction.processing.analytics
```

Or query the database directly:
```bash
sqlite3 sqlite/riskradar.db "SELECT status, COUNT(*) FROM pages GROUP BY status;"
```

## Database

Extraction and chunking data is stored in `sqlite/riskradar.db`:

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

## Performance

### Phase 3 (Extraction)
- Embedded text: ~35ms/page
- OCR with confidence: ~7-10 seconds/page (300 DPI)
- Batch of 10 reports (~450 pages): ~1.5 hours for OCR

### Phase 4 (Chunking)
- Full pipeline (510 reports, 30K pages): ~70 seconds
- Pass 0 (consolidate pages): ~7 seconds
- Pass 1 (build documents): ~25 seconds
- Pass 2 (chunk documents): ~35 seconds
