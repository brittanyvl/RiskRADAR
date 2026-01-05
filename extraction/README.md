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
riskradar extract initial
```

### Pass 2: OCR Retry
Runs OCR on pages that failed quality checks.

```bash
riskradar extract ocr-retry
```

### Full Pipeline
Runs both passes sequentially.

```bash
riskradar extract all
```

## JSON Schema

Each page is stored as `{report_id}/page_{number:04d}.json`:

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
riskradar analytics quality          # Overall summary
riskradar analytics by-decade        # Per-decade breakdown
riskradar analytics low-confidence   # Pages needing review
```

## Troubleshooting

**Resume interrupted run:**
```bash
riskradar extract all --resume
```

**Re-process specific report:**
```bash
# Delete from database and JSON files
# Then re-run extraction
```

**Check run history:**
```bash
riskradar analytics runs
```

## Database Tables

- `pages` - Per-page extraction tracking (JSON path, quality metrics, status)
- `extraction_runs` - Pipeline execution history
- `extraction_errors` - Error log with stack traces

## Performance

- Embedded text: ~35ms/page
- OCR with confidence: ~4 seconds/page
- Full corpus estimate: ~28 hours
