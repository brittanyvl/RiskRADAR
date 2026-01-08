# PDF Text Extraction Pipeline

This directory contains the multi-pass PDF text extraction pipeline for RiskRADAR.

## Directory Structure

- `processing/` - Extraction processing modules
  - `quality.py` - Quality assessment and thresholds
  - `pdf_reader.py` - Embedded text extraction (pymupdf)
  - `ocr.py` - OCR processing with confidence scores (pytesseract)
  - `extract.py` - Multi-pass pipeline orchestration
  - `analytics.py` - Quality reporting and analytics queries
- `scripts/` - Extraction-specific scripts
  - `verify_ocr.py` - Verify OCR dependencies installed
  - `install_tesseract.ps1` - Automated Tesseract installation (Windows)
  - `validate_extraction.py` - Pipeline validation script
- `json_data/` - Extraction results (GITIGNORED)
  - `passed/` - Pages that passed quality checks
  - `failed/` - Pages that failed embedded extraction (need OCR)
  - `ocr_retry/` - OCR re-extraction results
- `temp/` - Temporary workspace (deleted after quality gate)

## Pipeline Workflow

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

## JSON Schema

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

Extraction data is stored in `sqlite/riskradar.db`:

| Table | Purpose |
|-------|---------|
| `pages` | Per-page extraction status and quality metrics |
| `extraction_runs` | Pipeline execution history |
| `extraction_errors` | Error details with stack traces |

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

- Embedded text: ~35ms/page
- OCR with confidence: ~7-10 seconds/page (300 DPI)
- Batch of 10 reports (~450 pages): ~1.5 hours for OCR
