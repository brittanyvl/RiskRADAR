# RiskRADAR Phase 5: Text Embedding Pipeline - Implementation Plan

**Status: IMPLEMENTED (2026-01-13)**

## Executive Summary

This document outlines the implementation plan for Phase 5 (Embeddings + Vector Database + Benchmark) of the RiskRADAR project. The pipeline:

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

**Status: IMPLEMENTED with 50 stratified queries**

### Query Categories (50 total)

| Category | Count | Difficulty | Purpose |
|----------|-------|------------|---------|
| Incident Lookup | 10 | Easy | Known accidents with specific report IDs |
| Conceptual Queries | 12 | Medium-Hard | Technical concepts requiring semantic understanding |
| Section Queries | 10 | Medium | Queries targeting specific report sections |
| Comparative Queries | 8 | Hard | Analytical queries about patterns |
| Aircraft Queries | 6 | Medium | Aircraft-type specific searches |
| Phase Queries | 4 | Medium | Flight phase specific searches |

### Ground Truth Methodology

All ground truth established via SQL queries against `chunks.parquet`, NOT human judgment:
- **Incident queries:** Report IDs verified from NTSB metadata
- **Conceptual queries:** Term co-occurrence verified via LIKE patterns
- **Section queries:** Section names from chunk metadata

### Metrics Computed

| Metric | Description |
|--------|-------------|
| MRR | Mean Reciprocal Rank - position of first relevant result |
| Hit@K | K = 1, 3, 5, 10, 20 - at least one relevant in top K |
| Precision@K | K = 5, 10 - fraction of top K that are relevant |
| Recall@K | K = 5, 10, 20 - fraction of relevant found in top K |
| nDCG@10 | Normalized Discounted Cumulative Gain |
| Section Accuracy | For section queries - fraction from expected sections |
| Latency | Embed time + search time per query |

### Statistical Tests

- **Paired t-test** - parametric comparison between models
- **Wilcoxon signed-rank** - non-parametric alternative
- **Bootstrap 95% CI** - confidence intervals (n=1000)
- **Win/Loss/Tie analysis** - per-query comparison

### CLI Commands

```bash
python -m eval.benchmark run              # Benchmark both models
python -m eval.benchmark run -m minilm    # Single model
python -m eval.benchmark report           # Generate comparison report
python -m eval.benchmark validate         # Validate query definitions
```

### Output Files

```
eval/
├── gold_queries.yaml              # 50 stratified test queries
├── benchmark.py                   # Benchmark runner
├── benchmark_report.md            # Generated comparison report
└── results/
    ├── benchmark_minilm_*.json    # Full JSON results
    ├── benchmark_minilm_*.parquet # Streamlit-ready DataFrame
    ├── benchmark_mika_*.json
    └── benchmark_mika_*.parquet
```

---

## Implementation Phases

### Phase 5A: Foundation (Local Embedding) - COMPLETE
**Goal:** Generate embeddings and save to Parquet

**Completed:**
- `requirements.txt` updated with sentence-transformers, qdrant-client, pyarrow
- `embeddings/config.py` - model registry with MiniLM and MIKA configs
- `embeddings/models.py` - model wrapper with dimension validation
- `embeddings/storage.py` - Parquet read/write operations
- `embeddings/embed.py` - embedding generation pipeline
- `embeddings/cli.py` - full CLI with all commands

---

### Phase 5B: Database Integration - COMPLETE
**Goal:** Track embedding runs in SQLite

**Completed:**
- `sqlite/schema.py` updated to v4 with embedding_runs, qdrant_upload_runs, embedding_errors tables
- `sqlite/queries.py` with CRUD for all Phase 5 tables
- Run tracking integrated into embed.py and upload.py

---

### Phase 5C: Qdrant Upload - COMPLETE
**Goal:** Upload embeddings to Qdrant Cloud

**Completed:**
- `riskradar/config.py` with `get_qdrant_config()` function
- `.env.example` with Qdrant variables
- `embeddings/upload.py` with retry logic and batch uploads
- `upload`, `verify`, and `stats` commands in CLI
- Secrets management via .env and .streamlit/secrets.toml

---

### Phase 5D: Full Production Run - PENDING
**Goal:** Complete embedding + upload for both models

**To Run:**
```bash
python -m embeddings.cli embed both      # Generate embeddings
python -m embeddings.cli upload both     # Upload to Qdrant
python -m embeddings.cli verify both     # Verify collections
```

---

### Phase 5E: Benchmarking - COMPLETE
**Goal:** Compare models and determine winner

**Completed:**
- `eval/gold_queries.yaml` - 50 stratified test queries with verifiable ground truth
- `eval/benchmark.py` - professional benchmark runner with:
  - 6 query categories (incident, conceptual, section, comparative, aircraft, phase)
  - Full IR metrics (MRR, Hit@K, P@K, R@K, nDCG@10, Section Accuracy)
  - Latency tracking (embed + search time)
  - Statistical tests (paired t-test, Wilcoxon, bootstrap CI)
  - Win/loss/tie analysis
- JSON and Parquet output for Streamlit visualization

**To Run:**
```bash
python -m eval.benchmark run             # Benchmark both models
python -m eval.benchmark report          # Generate comparison report
```

---

### Phase 5F: Documentation - COMPLETE
**Goal:** Complete Phase 5 documentation

**Completed:**
- `README.md` updated with Phase 5 status and commands
- `CLAUDE.md` updated with full Phase 5 details
- `PHASE5_PLAN.md` (this file) marked as implemented
- `eval/README.md` created for benchmark documentation

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
