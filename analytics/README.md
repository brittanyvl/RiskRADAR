# analytics - DuckDB Analytics Engine

DuckDB-based analytics layer for querying pages, documents, and chunks data with SQL.

---

## Table of Contents

- [Overview](#overview)
- [Role in Pipeline](#role-in-pipeline)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
  - [Interactive Shell](#interactive-shell)
  - [Single Query](#single-query)
  - [Shell Commands](#shell-commands)
- [Available Views](#available-views)
- [Example Queries](#example-queries)
  - [Data Overview](#data-overview)
  - [Section Analysis](#section-analysis)
  - [Full-Text Search](#full-text-search)
  - [Join with Report Metadata](#join-with-report-metadata)
  - [Timeline Analysis](#timeline-analysis)
  - [Quality Analysis](#quality-analysis)
- [Data Files](#data-files)
- [Architecture](#architecture)
- [Refreshing Data](#refreshing-data)
- [Benchmark Integration](#benchmark-integration)
- [Limitations](#limitations)
- [See Also](#see-also)

---

## Overview

The `analytics` module provides:

- **JSONL to Parquet conversion** for columnar, compressed data storage
- **DuckDB in-memory SQL** for fast ad-hoc queries on extraction data
- **SQLite attachment** for joining with report metadata
- **Pre-built analytical views** for common query patterns
- **Interactive SQL shell** with tab completion

This is the exploration layer for the pipeline output. Use it to investigate chunk quality, section distribution, and text content before embedding.

---

## Role in Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Analytics Engine                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    extraction/json_data/          analytics/data/            DuckDB     │
│    ────────────────────          ───────────────          ──────────    │
│    pages.jsonl      ─────►       pages.parquet   ─────►   in-memory    │
│    documents.jsonl  ─────►       documents.parquet ───►   tables       │
│    chunks.jsonl     ─────►       chunks.parquet  ─────►                 │
│                                                                 │        │
│                                                                 ▼        │
│                                                        ┌──────────────┐ │
│                                                        │ sqlite/      │ │
│                                                        │ riskradar.db │ │
│                                                        │ (attached)   │ │
│                                                        └──────────────┘ │
│                                                                          │
│    Benchmark framework (eval/) also reads chunks.parquet for             │
│    signal-based ground truth verification.                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install DuckDB (if not already installed)
pip install duckdb

# 2. Convert JSONL to Parquet (one-time)
python -m analytics.convert

# 3. Launch interactive SQL shell
python -m analytics.cli
```

---

## CLI Usage

### Interactive Shell

```bash
python -m analytics.cli
```

This opens an interactive SQL prompt with:
- All Parquet tables loaded (pages, documents, chunks)
- SQLite database attached for report metadata
- Pre-built analytical views registered

### Single Query

```bash
python -m analytics.cli --query "SELECT COUNT(*) FROM chunks"
```

### Shell Commands

| Command | Description |
|---------|-------------|
| `.help` | Show help |
| `.exit` | Exit shell |
| `.tables` | List tables |
| `.views` | List available views |
| `.schema TABLE` | Show table schema |

---

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
| `timeline_by_decade` | Chunks by decade (accident date) |
| `timeline_by_year` | Chunks by year (accident date) |

---

## Example Queries

### Data Overview

```sql
-- Summary statistics
SELECT * FROM data_summary;
```

Output:
```
┌────────┬───────────┬────────┬─────────────┐
│ pages  │ documents │ chunks │ total_tokens│
├────────┼───────────┼────────┼─────────────┤
│ 30602  │ 510       │ 24766  │ 15663076    │
└────────┴───────────┴────────┴─────────────┘
```

### Section Analysis

```sql
-- Chunks by section name
SELECT * FROM chunks_by_section LIMIT 10;

-- Token distribution across sections
SELECT section_name,
       COUNT(*) as chunk_count,
       AVG(token_count) as avg_tokens,
       SUM(token_count) as total_tokens
FROM chunks
WHERE section_name IS NOT NULL
GROUP BY section_name
ORDER BY chunk_count DESC
LIMIT 15;
```

### Full-Text Search

```sql
-- Search for specific content
SELECT chunk_id, report_id, section_name,
       LEFT(chunk_text, 200) as preview
FROM chunks
WHERE chunk_text ILIKE '%engine failure%'
LIMIT 10;

-- Find chunks mentioning specific aircraft
SELECT chunk_id, report_id, section_name
FROM chunks
WHERE chunk_text ILIKE '%Boeing 737%'
  AND chunk_text ILIKE '%rudder%';
```

### Join with Report Metadata

```sql
-- Join chunks with report titles and dates
SELECT
    c.section_name,
    r.title,
    r.accident_date,
    r.location
FROM chunks c
JOIN sqlite.reports r ON c.report_id = r.filename
WHERE r.accident_date > '2000-01-01'
LIMIT 20;

-- Chunks from specific accident
SELECT chunk_sequence, section_name, token_count,
       LEFT(chunk_text, 100) as preview
FROM chunks
WHERE report_id = 'AAR0201.pdf'
ORDER BY chunk_sequence;
```

### Timeline Analysis

```sql
-- Chunks by decade
SELECT * FROM timeline_by_decade;

-- Chunks by year (for trending)
SELECT * FROM timeline_by_year
WHERE year >= 2000
ORDER BY year;
```

### Quality Analysis

```sql
-- Low quality OCR pages
SELECT report_id, page_number, ocr_confidence
FROM pages
WHERE source = 'ocr' AND ocr_confidence < 70
ORDER BY ocr_confidence
LIMIT 20;

-- Token distribution buckets
SELECT * FROM token_distribution;

-- Reports with most chunks
SELECT * FROM chunks_per_report LIMIT 10;
```

---

## Data Files

| File | Records | Size | Description |
|------|---------|------|-------------|
| `data/pages.parquet` | 30,602 | ~19.5 MB | Page-level extraction results |
| `data/documents.parquet` | 510 | ~18.7 MB | Full document text |
| `data/chunks.parquet` | 24,766 | ~18.5 MB | Search-ready chunks with text |

Parquet benefits over JSONL:
- **Columnar storage**: Only reads columns needed for query
- **Compression**: ~3x smaller than JSONL
- **Fast loading**: DuckDB reads Parquet directly

---

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

---

## Refreshing Data

If you re-run the chunking pipeline, regenerate the Parquet files:

```bash
python -m analytics.convert
```

This reads the latest JSONL files and overwrites the Parquet files.

---

## Benchmark Integration

The benchmark framework (Phase 5) uses `chunks.parquet` for ground truth verification:

```sql
-- Verify expected chunks exist for a benchmark query
SELECT COUNT(*) FROM chunks
WHERE chunk_text ILIKE '%hydraulic system failure%'
  AND section_name ILIKE '%probable cause%';
```

Benchmark results are also exported to Parquet for Streamlit visualization:
- `eval/results/*_results.parquet` - Per-model benchmark results

See [eval/README.md](../eval/README.md) for benchmark documentation.

---

## Limitations

1. **In-Memory Only**: DuckDB loads all data into RAM. Large datasets may require more memory.

2. **No Persistence**: Queries run against in-memory tables. No persistent DuckDB database file.

3. **Read-Only**: Analytics layer is read-only. Data modifications require re-running pipelines.

4. **Manual Refresh**: Parquet files must be manually regenerated after pipeline changes.

5. **Case-Insensitive Search**: `ILIKE` is slow for large tables. Consider indexing for production use.

6. **Single-User**: Interactive shell designed for single-user exploration, not concurrent access.

---

## See Also

- [Main README](../README.md) - Project overview
- [extraction/README.md](../extraction/README.md) - Source data: JSONL files
- [eval/README.md](../eval/README.md) - Benchmark uses chunks.parquet
- [sqlite/README.md](../sqlite/README.md) - Report metadata for joins
