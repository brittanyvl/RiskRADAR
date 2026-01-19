# embeddings - Embedding Generation & Vector Upload (Phase 5)

Embedding generation pipeline and Qdrant Cloud integration for RiskRADAR semantic search.

---

## Table of Contents

- [Overview](#overview)
- [Role in Pipeline](#role-in-pipeline)
- [Model Registry](#model-registry)
- [Prerequisites](#prerequisites)
- [CLI Commands](#cli-commands)
  - [Embedding Generation](#embedding-generation)
  - [Qdrant Upload](#qdrant-upload)
  - [Verification](#verification)
- [Output Files](#output-files)
- [Qdrant Payload Schema](#qdrant-payload-schema)
- [Pipeline Configuration](#pipeline-configuration)
- [Environment Variables](#environment-variables)
- [Directory Structure](#directory-structure)
- [Database Tables](#database-tables)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [See Also](#see-also)

---

## Overview

The `embeddings` module provides:

- **Dual-model embedding generation** comparing general-purpose vs domain-specific models
- **Parquet storage** for local embedding persistence and analytics
- **Qdrant Cloud integration** with retry logic and batch uploading
- **Rich payload schema** for metadata filtering during search
- **Run tracking** in SQLite for reproducibility

This module bridges the gap between text chunks and searchable vectors, enabling semantic search over the 24,766 chunks in the corpus.

---

## Role in Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Phase 5: Embeddings                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    chunks.jsonl ──────► sentence-transformers ──────► Qdrant Cloud      │
│    (24,766)                     │                          │             │
│                                 ▼                          ▼             │
│                        ┌───────────────┐          ┌───────────────┐     │
│                        │    MiniLM     │          │ riskradar_    │     │
│                        │  384 dims     │          │   minilm      │     │
│                        │  ~15 min      │          │ (24,766 vecs) │     │
│                        └───────────────┘          └───────────────┘     │
│                        ┌───────────────┐          ┌───────────────┐     │
│                        │     MIKA      │          │ riskradar_    │     │
│                        │  768 dims     │          │   mika        │     │
│                        │  ~3.5 hours   │          │ (24,766 vecs) │     │
│                        └───────────────┘          └───────────────┘     │
│                                 │                                        │
│                                 ▼                                        │
│                        embeddings_data/*.parquet                        │
│                        (local backup)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Model Registry

| Model | HuggingFace ID | Dimensions | Collection | Batch Size |
|-------|----------------|------------|------------|------------|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 | `riskradar_minilm` | 32 |
| MIKA | `NASA-AIML/MIKA_Custom_IR` | 768 | `riskradar_mika` | 16 |

**Model comparison:**

| Aspect | MiniLM | MIKA |
|--------|--------|------|
| **Purpose** | General-purpose baseline | Domain-specific (aviation) |
| **Training data** | Web text | NASA aerospace/aviation documents |
| **Embedding time** | ~15 minutes | ~3.5 hours |
| **Storage** | ~56 MB | ~112 MB |
| **MRR (benchmark)** | 0.704 | 0.816 |
| **Semantic Lift** | +28.2% | +38.6% |

**Recommendation:** MIKA achieves 38.6% semantic lift over keyword baseline, demonstrating the value of domain-specific embeddings for specialized corpora.

---

## Prerequisites

Before running embedding commands:

1. **Chunks must exist:**
   ```bash
   # Verify chunks.jsonl exists
   ls extraction/json_data/chunks.jsonl

   # If not, run chunking pipeline
   python -m extraction.processing.chunk all
   ```

2. **Qdrant credentials configured (for upload):**
   ```bash
   # Copy template and edit
   cp .env.example .env
   # Add QDRANT_URL and QDRANT_API_KEY
   ```

---

## CLI Commands

### Embedding Generation

```bash
# Generate embeddings for specific model
python -m embeddings.cli embed minilm    # MiniLM only (~15 min)
python -m embeddings.cli embed mika      # MIKA only (~3.5 hours)
python -m embeddings.cli embed both      # Both models

# With version tag (for benchmarking different chunk versions)
python -m embeddings.cli embed both -V v2
```

### Qdrant Upload

```bash
# Upload to Qdrant Cloud
python -m embeddings.cli upload minilm
python -m embeddings.cli upload mika
python -m embeddings.cli upload both

# Delete and recreate collection first
python -m embeddings.cli upload both --recreate

# With version tag
python -m embeddings.cli upload both -V v2
```

### Verification

```bash
# Verify collection vector counts
python -m embeddings.cli verify minilm
python -m embeddings.cli verify mika
python -m embeddings.cli verify both

# Show embedding statistics
python -m embeddings.cli stats
```

### Full Pipeline

```bash
# Embed + upload for both models
python -m embeddings.cli all

# With options
python -m embeddings.cli all --recreate -V v2
```

---

## Output Files

Embeddings are stored locally in Parquet format:

| File | Size | Contents |
|------|------|----------|
| `embeddings_data/minilm_embeddings.parquet` | ~56 MB | 24,766 x 384 vectors + metadata |
| `embeddings_data/mika_embeddings.parquet` | ~112 MB | 24,766 x 768 vectors + metadata |

Parquet schema:
```
chunk_id: string
report_id: string
embedding: list<float>  # 384 or 768 dimensions
token_count: int
section_name: string
created_at: timestamp
```

---

## Qdrant Payload Schema

Each vector in Qdrant has the following payload for filtering:

```json
{
  "chunk_id": "AAR0201.pdf_chunk_0042",
  "report_id": "AAR0201.pdf",
  "chunk_sequence": 42,
  "page_start": 15,
  "page_end": 16,
  "section_name": "ANALYSIS",
  "token_count": 623,
  "text_source": "embedded",
  "accident_date": "2000-01-31",
  "report_date": "2002-12-30",
  "location": "Point Mugu, California",
  "title": "Loss of Control and Impact with Pacific Ocean..."
}
```

**Note:** `chunk_text` is NOT stored in Qdrant to save storage. For signal-based evaluation, the benchmark loads text from `analytics/data/chunks.parquet`.

---

## Pipeline Configuration

```python
# embeddings/config.py
PIPELINE_CONFIG = {
    "version": "5.0.0",
    "distance_metric": "cosine",
    "upload_batch_size": 100,
    "retry_attempts": 3,
    "retry_delay_sec": 2.0,
}
```

---

## Environment Variables

Required for upload operations:

```bash
# .env file
QDRANT_URL=https://your-cluster.region.cloud.qdrant.io:6333
QDRANT_API_KEY=your_api_key_here
```

**Getting Qdrant credentials:**
1. Create account at https://cloud.qdrant.io/
2. Create a cluster (free tier sufficient)
3. Go to **API Keys** tab and create a new key
4. Copy cluster URL and API key to `.env`

---

## Directory Structure

```
embeddings/
├── __init__.py      # Package marker
├── config.py        # Model registry and paths
├── models.py        # Model wrapper with dimension validation
├── storage.py       # Parquet read/write for embeddings
├── embed.py         # Embedding generation pipeline
├── upload.py        # Qdrant upload with retry logic
└── cli.py           # CLI entry point
```

---

## Database Tables

Run tracking is stored in SQLite:

| Table | Purpose |
|-------|---------|
| `embedding_runs` | Embedding generation run history |
| `qdrant_upload_runs` | Qdrant upload run history |
| `embedding_errors` | Error log with details |

```sql
-- View recent embedding runs
SELECT * FROM embedding_runs ORDER BY id DESC LIMIT 5;

-- View upload status
SELECT * FROM qdrant_upload_runs ORDER BY id DESC LIMIT 5;
```

---

## Troubleshooting

**Qdrant connection fails:**
- Verify `.env` has correct `QDRANT_URL` and `QDRANT_API_KEY`
- Check Qdrant Cloud dashboard for cluster status
- Ensure network allows outbound HTTPS

**Dimension mismatch:**
- The pipeline validates embedding dimensions match expected values
- MiniLM must produce 384, MIKA must produce 768
- If mismatch, model may have been updated - check HuggingFace

**Resume interrupted upload:**
- Use `--recreate` flag to start fresh
- Without flag, upload resumes from existing vectors

**Memory issues with MIKA:**
- MIKA uses batch_size=16 (vs 32 for MiniLM) to manage memory
- Process takes ~3.5 hours on CPU
- Consider running overnight or on a machine with more RAM

**"Chunks file not found" error:**
- Run the chunking pipeline first:
  ```bash
  python -m extraction.processing.chunk all
  ```

---

## Limitations

1. **CPU-Only**: Embedding generation runs on CPU. GPU support would significantly speed up MIKA.

2. **No Incremental Updates**: Currently regenerates all embeddings. No delta processing for new chunks.

3. **Single Collection per Model**: Each model gets one Qdrant collection. No multi-version support within a collection.

4. **Network Dependent**: Upload requires stable internet connection. Large batches may timeout on slow connections.

5. **No Chunk Text in Qdrant**: To save storage, chunk text is not uploaded. Requires local Parquet files for text retrieval.

6. **Model Download on First Run**: First embedding run downloads models from HuggingFace (~500MB for MIKA).

---

## See Also

- [Main README](../README.md) - Project overview
- [extraction/README.md](../extraction/README.md) - Input: chunks.jsonl
- [eval/README.md](../eval/README.md) - Benchmarking embedded collections
- [analytics/README.md](../analytics/README.md) - chunks.parquet for text retrieval
- [riskradar/README.md](../riskradar/README.md) - Qdrant credential configuration
