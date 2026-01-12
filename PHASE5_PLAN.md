# RiskRADAR Phase 5: Text Embedding Pipeline - Implementation Plan

## Executive Summary

This document outlines the implementation plan for Phase 5 (Embeddings + Vector Database + Benchmark) of the RiskRADAR project. The pipeline will:

1. Generate embeddings for 28,321 text chunks using two models:
   - **Baseline:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) - general purpose
   - **Domain-specific:** `NASA-AIML/MIKA_Custom_IR` (768 dimensions) - aerospace/aviation specialized (BERT-base architecture)

2. Store embeddings locally (Parquet) before uploading to Qdrant Cloud

3. Benchmark both models for search relevance to determine the winner for the Streamlit app

---

## Requirements Summary

Based on your input:

| Requirement | Decision |
|-------------|----------|
| Secrets management | Both env vars and Streamlit secrets (fallback chain) |
| Benchmark approach | Manual gold queries (20-30 test queries with expected results) |
| Storage strategy | Local Parquet first, then bulk upload to Qdrant |
| Collection naming | `riskradar_minilm` and `riskradar_mika` |
| Qdrant setup | Include setup instructions (need account) |
| Hardware | CPU-only (batch size 16-32) |
| MIKA fallback | Fail loudly if model can't load |

---

## Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      chunks.jsonl (28,321 chunks)                       │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │  MiniLM Embeddings    │   │   MIKA Embeddings     │
        │  (384 dimensions)     │   │   (768 dimensions)    │
        └───────────┬───────────┘   └───────────┬───────────┘
                    │                           │
                    ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │ minilm_embeddings.pqt │   │ mika_embeddings.pqt   │
        │   (Local Parquet)     │   │   (Local Parquet)     │
        └───────────┬───────────┘   └───────────┬───────────┘
                    │                           │
                    ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │   riskradar_minilm    │   │    riskradar_mika     │
        │   (Qdrant Cloud)      │   │    (Qdrant Cloud)     │
        └───────────────────────┘   └───────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │     Benchmark Suite       │
                    │  (20-30 gold queries)     │
                    └───────────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  benchmark_report.md      │
                    │  (Winner recommendation)  │
                    └───────────────────────────┘
```

### Module Structure

```
riskRADAR/
├── embeddings/                    # NEW - Embedding pipeline
│   ├── __init__.py
│   ├── config.py                  # Model registry, paths, batch sizes
│   ├── models.py                  # Model wrapper, dimension detection
│   ├── storage.py                 # Parquet read/write
│   ├── embed.py                   # Embedding generation pipeline
│   ├── upload.py                  # Qdrant upload pipeline
│   ├── cli.py                     # CLI entry point
│   └── README.md                  # Pipeline documentation
│
├── eval/                          # NEW - Benchmarking
│   ├── __init__.py
│   ├── gold_queries.yaml          # 20-30 test queries
│   ├── metrics.py                 # Precision, Recall, MRR, nDCG
│   ├── benchmark.py               # Benchmark runner
│   └── results/                   # Benchmark outputs
│       ├── minilm_results.json
│       ├── mika_results.json
│       └── benchmark_report.md
│
├── embeddings_data/               # NEW - Output directory (gitignored)
│   ├── minilm_embeddings.parquet
│   └── mika_embeddings.parquet
│
├── sqlite/
│   ├── schema.py                  # MODIFY - Add Phase 5 tables
│   └── queries.py                 # MODIFY - Add Phase 5 queries
│
├── riskradar/
│   └── config.py                  # MODIFY - Add Qdrant config
│
├── requirements.txt               # MODIFY - Uncomment dependencies
├── .env.example                   # MODIFY - Add Qdrant vars
└── .gitignore                     # MODIFY - Add embeddings_data/
```

---

## CLI Interface

```bash
# Embedding generation
python -m embeddings.cli embed minilm        # Embed with MiniLM only
python -m embeddings.cli embed mika          # Embed with MIKA only
python -m embeddings.cli embed both          # Embed with both models

# Upload to Qdrant
python -m embeddings.cli upload minilm       # Upload MiniLM to Qdrant
python -m embeddings.cli upload mika         # Upload MIKA to Qdrant
python -m embeddings.cli upload both         # Upload both

# Full pipeline
python -m embeddings.cli all                 # Embed + upload both models

# Verification and stats
python -m embeddings.cli verify minilm       # Verify Qdrant collection
python -m embeddings.cli stats               # Show embedding statistics

# Benchmarking
python -m embeddings.cli benchmark           # Run full benchmark suite
python -m embeddings.cli benchmark --model minilm  # Benchmark single model

# Options
--limit N        Limit chunks (for testing)
--batch-size N   Override batch size
--verbose        Debug logging
--skip-upload    Generate embeddings only, don't upload
```

---

## Database Schema (Phase 5 Additions)

### embedding_runs Table
```sql
CREATE TABLE IF NOT EXISTS embedding_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,           -- 'minilm' or 'mika'
    model_id TEXT NOT NULL,             -- HuggingFace model ID
    run_type TEXT NOT NULL CHECK(run_type IN ('full', 'resume', 'incremental')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),

    -- Stats
    total_chunks INTEGER DEFAULT 0,
    embedding_dimension INTEGER,
    embeddings_generated INTEGER DEFAULT 0,
    total_time_sec REAL,
    embeddings_per_sec REAL,

    -- Output
    parquet_path TEXT,
    parquet_size_mb REAL,

    -- Error tracking
    error_count INTEGER DEFAULT 0,
    config_json TEXT
);
```

### qdrant_upload_runs Table
```sql
CREATE TABLE IF NOT EXISTS qdrant_upload_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    embedding_run_id INTEGER,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),

    -- Stats
    total_vectors INTEGER DEFAULT 0,
    uploaded_vectors INTEGER DEFAULT 0,
    failed_vectors INTEGER DEFAULT 0,
    batches_uploaded INTEGER DEFAULT 0,
    total_time_sec REAL,

    -- Error tracking
    error_count INTEGER DEFAULT 0,
    qdrant_url TEXT NOT NULL,
    config_json TEXT,

    FOREIGN KEY (embedding_run_id) REFERENCES embedding_runs(id) ON DELETE SET NULL
);
```

### embedding_errors Table
```sql
CREATE TABLE IF NOT EXISTS embedding_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    chunk_id TEXT,
    error_type TEXT NOT NULL,
    error_message TEXT,
    stack_trace TEXT,
    created_at TEXT NOT NULL,

    FOREIGN KEY (run_id) REFERENCES embedding_runs(id) ON DELETE CASCADE
);
```

---

## Qdrant Cloud Setup Instructions

### 1. Create Qdrant Cloud Account
1. Go to https://cloud.qdrant.io/
2. Sign up with email/Google/GitHub
3. Free tier includes 1GB storage (~1M vectors) - sufficient for 56K vectors (28K x 2 models)

### 2. Create a Cluster
1. Click "Create Cluster"
2. Select region closest to you (e.g., us-east-1)
3. Name: `riskradar-production`
4. Free tier: Starter (sufficient)

### 3. Get API Credentials
1. Go to "API Keys" in dashboard
2. Create new API key with full access
3. Copy the cluster URL (looks like: `https://xxx-xxx.us-east-1.aws.cloud.qdrant.io:6333`)
4. Copy the API key

### 4. Configure Environment
Create `.env` file in project root:
```bash
QDRANT_URL=https://your-cluster.us-east-1.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_api_key_here
```

For Streamlit deployment, add to `.streamlit/secrets.toml`:
```toml
[default]
QDRANT_URL = "https://your-cluster.us-east-1.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your_api_key_here"
```

---

## Parquet Schema

Each Parquet file will contain:

| Column | Type | Description |
|--------|------|-------------|
| chunk_id | string | Unique chunk identifier |
| embedding | float[] | Embedding vector (384 or D dimensions) |
| report_id | string | Links to reports table |
| chunk_sequence | int | Order within document |
| page_start | int | Starting page in PDF |
| page_end | int | Ending page in PDF |
| section_name | string | Detected section header |
| token_count | int | Token count |
| text_source | string | 'embedded', 'ocr', or 'mixed' |

Metadata stored in Parquet file metadata:
- model_name
- model_id
- embedding_dimension
- created_at
- pipeline_version

---

## Qdrant Vector Payload Schema

Each vector in Qdrant will include this payload (for filtering):

```json
{
    "chunk_id": "AAR6903.pdf_chunk_0001",
    "report_id": "AAR6903.pdf",
    "page_start": 5,
    "page_end": 6,
    "section_name": "HISTORY OF FLIGHT",
    "accident_date": "1969-03-15",
    "report_date": "1970-01-20",
    "location": "New York, NY",
    "text_source": "embedded",
    "quality_flags": []
}
```

---

## Benchmark Evaluation Plan

### Gold Queries File Format (eval/gold_queries.yaml)

```yaml
queries:
  - id: "crew_fatigue"
    query: "accidents caused by pilot fatigue or crew rest issues"
    expected_themes: ["fatigue", "crew rest", "duty time"]
    expected_report_ids: ["AAR9901.pdf", "AAR0502.pdf"]  # Known relevant reports

  - id: "icing_conditions"
    query: "aircraft icing and anti-ice system failures"
    expected_themes: ["icing", "anti-ice", "de-ice", "freezing"]

  # ... 20-30 total queries
```

### Metrics to Compute

| Metric | Description |
|--------|-------------|
| Precision@k | % of top-k results that are relevant |
| Recall@k | % of relevant docs found in top-k |
| MRR | Mean Reciprocal Rank of first relevant result |
| nDCG@k | Normalized Discounted Cumulative Gain |
| Latency | Average query response time |

### Benchmark Report Structure

```markdown
# RiskRADAR Embedding Model Benchmark Report

## Summary
| Metric | MiniLM | MIKA | Winner |
|--------|--------|------|--------|
| Precision@10 | 0.72 | 0.81 | MIKA |
| Recall@10 | 0.65 | 0.78 | MIKA |
| MRR | 0.82 | 0.89 | MIKA |
| Avg Latency (ms) | 45 | 62 | MiniLM |

## Recommendation
Based on benchmark results, **MIKA** is recommended for production use
due to superior relevance metrics in the aviation domain.

## Per-Query Analysis
[Detailed breakdown by query...]
```

---

## Implementation Phases

### Phase 5A: Foundation (Local Embedding)
**Goal:** Generate embeddings and save to Parquet

**Tasks:**
1. Update `requirements.txt` - uncomment sentence-transformers, qdrant-client, add pyarrow
2. Create `embeddings/config.py` - model registry, batch sizes, paths
3. Create `embeddings/models.py` - model wrapper with dimension detection
4. Create `embeddings/storage.py` - Parquet read/write operations
5. Create `embeddings/embed.py` - embedding generation pipeline
6. Create `embeddings/cli.py` - CLI with `embed` command
7. Test with `--limit 100` on both models

**Acceptance Criteria:**
- `python -m embeddings.cli embed minilm --limit 100` succeeds
- `python -m embeddings.cli embed mika --limit 100` succeeds
- Parquet files created with correct dimensions
- MIKA fails loudly if model unavailable

---

### Phase 5B: Database Integration
**Goal:** Track embedding runs in SQLite

**Tasks:**
1. Update `sqlite/schema.py` - add Phase 5 tables, bump version to 4
2. Update `sqlite/queries.py` - add CRUD for embedding_runs, qdrant_upload_runs
3. Integrate run tracking into `embeddings/embed.py`
4. Test: Verify embedding_runs table populated after embed

**Acceptance Criteria:**
- Each embed command creates a run record
- Stats (time, dimension, count) correctly recorded
- Error logging functional

---

### Phase 5C: Qdrant Upload
**Goal:** Upload embeddings to Qdrant Cloud

**Tasks:**
1. Update `riskradar/config.py` - add `get_qdrant_config()` function
2. Update `.env.example` - add Qdrant variables
3. Create `embeddings/upload.py` - Qdrant upload with retry logic
4. Add `upload` and `verify` commands to CLI
5. Test with small batch, then full upload

**Acceptance Criteria:**
- `python -m embeddings.cli upload minilm` succeeds
- Collection created with correct dimension
- `verify` confirms vector count matches
- qdrant_upload_runs table populated

---

### Phase 5D: Full Production Run
**Goal:** Complete embedding + upload for both models

**Tasks:**
1. Run full embed pipeline: `python -m embeddings.cli embed both`
2. Run full upload: `python -m embeddings.cli upload both`
3. Verify both Qdrant collections
4. Document results

**Acceptance Criteria:**
- 28,321 embeddings per model
- Both Qdrant collections verified
- No errors in logs

---

### Phase 5E: Benchmarking
**Goal:** Compare models and determine winner

**Tasks:**
1. Create `eval/gold_queries.yaml` - 20-30 test queries
2. Create `eval/metrics.py` - relevance metric calculations
3. Create `eval/benchmark.py` - benchmark runner
4. Add `benchmark` command to CLI
5. Generate comparison report

**Acceptance Criteria:**
- Both models benchmarked on identical queries
- Metrics comparable (precision, recall, MRR, nDCG)
- Clear recommendation in benchmark_report.md

---

### Phase 5F: Documentation
**Goal:** Complete Phase 5 documentation

**Tasks:**
1. Write `embeddings/README.md` with usage examples
2. Update `CLAUDE.md` with Phase 5 completion status
3. Finalize `eval/results/benchmark_report.md`

**Acceptance Criteria:**
- All documentation complete
- CLI usage examples included
- Benchmark results professionally formatted

---

## Performance Estimates (CPU-only)

| Operation | Estimate |
|-----------|----------|
| MiniLM embedding (28K chunks) | 50-100 minutes |
| MIKA embedding (28K chunks) | 60-150 minutes |
| Qdrant upload per model | 1-2 minutes |
| Benchmark (30 queries x 2 models) | 5-10 minutes |
| **Total Phase 5** | 2-4 hours |

## Storage Estimates

| File | Size |
|------|------|
| minilm_embeddings.parquet | ~45 MB (28K × 384 dims) |
| mika_embeddings.parquet | ~90 MB (28K × 768 dims) |
| Qdrant free tier usage | ~10% of 1GB quota |

---

## Error Handling Strategy

### Model Loading (MIKA)
- **Behavior:** Fail loudly with clear error message
- **Reason:** MIKA is the domain-specific model - it must work

### Embedding Batch Errors
- **Behavior:** Log error, skip chunk, continue pipeline
- **Reason:** One bad chunk shouldn't stop 28K embeddings

### Qdrant Upload Errors
- **Behavior:** Retry 3x with exponential backoff, then fail
- **Reason:** Network issues are transient

### Resume Capability
- **Behavior:** Load existing Parquet, embed only missing chunks
- **Reason:** Don't re-embed if pipeline interrupted

---

## Files Summary

### New Files (9 files, ~1,400 lines)

| File | Lines (est) | Purpose |
|------|-------------|---------|
| `embeddings/__init__.py` | 10 | Package marker |
| `embeddings/config.py` | 80 | Model configs, paths |
| `embeddings/models.py` | 150 | Model wrapper |
| `embeddings/storage.py` | 150 | Parquet I/O |
| `embeddings/embed.py` | 250 | Embedding pipeline |
| `embeddings/upload.py` | 200 | Qdrant upload |
| `embeddings/cli.py` | 350 | CLI entry point |
| `embeddings/README.md` | 150 | Documentation |
| `eval/gold_queries.yaml` | 100 | Test queries |

### Modified Files (6 files, ~350 lines)

| File | Changes |
|------|---------|
| `requirements.txt` | Uncomment dependencies |
| `riskradar/config.py` | Add Qdrant config |
| `.env.example` | Add Qdrant vars |
| `sqlite/schema.py` | Phase 5 tables |
| `sqlite/queries.py` | Phase 5 queries |
| `.gitignore` | Add embeddings_data/ |

---

## Success Criteria

Phase 5 is complete when:

1. **Embeddings Generated**
   - 28,321 MiniLM embeddings (384 dim) in Parquet
   - 28,321 MIKA embeddings (D dim) in Parquet

2. **Qdrant Collections**
   - `riskradar_minilm` with 28,321 vectors
   - `riskradar_mika` with 28,321 vectors

3. **Benchmark Complete**
   - 20-30 gold queries evaluated
   - Metrics computed for both models
   - Winner recommendation documented

4. **Documentation**
   - CLI usage documented
   - Benchmark report complete
   - CLAUDE.md updated

---

## Next Steps After Approval

Once you approve this plan, I will:

1. Create the `embeddings/` module structure
2. Implement the embedding pipeline (embed command first)
3. Add database schema and run tracking
4. Implement Qdrant upload
5. Create benchmarking infrastructure
6. Run full pipeline and generate benchmark report

Ready for your approval!
