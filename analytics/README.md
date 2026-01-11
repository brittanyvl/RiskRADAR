# RiskRADAR Analytics Engine

DuckDB-based analytics layer for querying pages, documents, and chunks data with SQL.

## Quick Start

```bash
# 1. Install DuckDB (if not already installed)
pip install duckdb

# 2. Convert JSONL to Parquet (one-time)
py -m analytics.convert

# 3. Launch interactive SQL shell
py -m analytics.cli
```

## CLI Usage

### Interactive Shell
```bash
py -m analytics.cli
```

This opens an interactive SQL prompt with:
- All Parquet tables loaded (pages, documents, chunks)
- SQLite database attached for report metadata
- Pre-built analytical views registered

### Single Query
```bash
py -m analytics.cli --query "SELECT COUNT(*) FROM chunks"
```

### Shell Commands
| Command | Description |
|---------|-------------|
| `.help` | Show help |
| `.exit` | Exit shell |
| `.tables` | List tables |
| `.views` | List available views |
| `.schema TABLE` | Show table schema |

## Available Views

| View | Description |
|------|-------------|
| `data_summary` | Overall counts (pages, documents, chunks, tokens) |
| `extraction_quality` | Quality metrics by extraction source |
| `chunks_by_section` | Chunk distribution across sections |
| `token_distribution` | Token count buckets |
| `chunks_per_report` | Chunks per report |
| `section_detection_stats` | Section detection method stats |
| `text_source_distribution` | Embedded vs OCR distribution |
| `chunks_enriched` | Chunks joined with report metadata |
| `timeline_by_decade` | Chunks by decade |
| `timeline_by_year` | Chunks by year |

## Example Queries

### Data Overview
```sql
SELECT * FROM data_summary;
```

### Section Analysis
```sql
SELECT * FROM chunks_by_section LIMIT 10;
```

### Full-Text Search
```sql
SELECT chunk_id, report_id, section_name, LEFT(chunk_text, 200) as preview
FROM chunks
WHERE chunk_text ILIKE '%engine failure%'
LIMIT 10;
```

### Join with Report Metadata
```sql
SELECT
    c.section_name,
    r.title,
    r.accident_date,
    r.location
FROM chunks c
JOIN sqlite.reports r ON c.report_id = r.filename
WHERE r.accident_date > '2000-01-01'
LIMIT 20;
```

### Timeline Analysis
```sql
SELECT * FROM timeline_by_decade;
```

### Low Quality OCR Pages
```sql
SELECT report_id, page_number, ocr_confidence
FROM pages
WHERE source = 'ocr' AND ocr_confidence < 70
ORDER BY ocr_confidence
LIMIT 20;
```

### Chunks from Specific Report
```sql
SELECT chunk_sequence, section_name, token_count, LEFT(chunk_text, 100) as preview
FROM chunks
WHERE report_id = 'AAR7005.pdf'
ORDER BY chunk_sequence;
```

### Reports with Most Chunks
```sql
SELECT * FROM chunks_per_report LIMIT 10;
```

## Refreshing Data

If you re-run the chunking pipeline, regenerate the Parquet files:

```bash
py -m analytics.convert
```

## Data Files

| File | Records | Description |
|------|---------|-------------|
| `data/pages.parquet` | 30,602 | Page-level extraction results |
| `data/documents.parquet` | 510 | Full document text |
| `data/chunks.parquet` | 28,321 | Search-ready chunks |

## Architecture

```
JSONL Files                    Parquet Files              DuckDB
(extraction/json_data/)   ->   (analytics/data/)    ->   (in-memory)
                                                              |
                                                              v
                                                    SQLite attachment
                                                    (sqlite/riskradar.db)
```

The analytics engine:
1. Converts JSONL to Parquet (columnar, compressed)
2. Loads Parquet into DuckDB in-memory tables
3. Attaches SQLite for report metadata joins
4. Registers pre-built views for common queries
