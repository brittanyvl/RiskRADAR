# RiskRADAR

**R**etrieval and **D**iscovery of **A**viation **A**ccident **R**eports

An end-to-end data engineering and machine learning pipeline that transforms unstructured NTSB aviation accident PDFs into searchable, semantically-indexed safety insights using modern NLP techniques, vector databases, and rigorous evaluation methodology.

---

## Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Verify Setup](#verify-setup)
  - [Run Pipeline](#run-pipeline)
- [Requirements](#requirements)
- [Module Documentation](#module-documentation)
- [Data Pipeline](#data-pipeline)
  - [Phase 1-2: Web Scraping](#phase-1-2-web-scraping)
  - [Phase 3: Text Extraction](#phase-3-text-extraction)
  - [Phase 4: Chunking](#phase-4-chunking)
  - [Phase 5: Embeddings](#phase-5-embeddings)
- [Evaluation Framework](#evaluation-framework)
- [Taxonomy & Cause Attribution](#taxonomy--cause-attribution-phase-6-8)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [License](#license)

---

## Overview

RiskRADAR demonstrates a complete production-grade pipeline for processing unstructured documents into a semantic search system. The project:

1. **Scrapes** 510 aviation accident reports from the NTSB website
2. **Extracts** text using embedded extraction and OCR with quality metrics
3. **Chunks** documents into semantically coherent segments with section awareness
4. **Embeds** chunks using both general-purpose and domain-specific models
5. **Indexes** vectors in Qdrant Cloud for similarity search
6. **Evaluates** retrieval quality using a rigorous 50-query benchmark with human review

This project serves as a portfolio piece demonstrating skills in data engineering, NLP, information retrieval, and ML evaluation methodology.

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 - Scraping | **Complete** | 510 PDFs downloaded from NTSB |
| Phase 2 - Metadata | **Complete** | Report metadata extracted and stored |
| Phase 3 - Text Extraction | **Complete** | 30,602 pages extracted (14K embedded + 16K OCR) |
| Phase 4 - Chunking | **Complete** | 24,766 search-ready chunks (v2: 400-800 tokens) |
| Phase 5 - Embeddings | **Complete** | Dual-model embeddings with benchmark evaluation |
| Phase 6A - Discovery | **Planned** | BERTopic topic discovery from cause sections |
| Phase 6A-Review | **Planned** | **GATE 1**: Human topic review & approval |
| Phase 6B - Taxonomy Build | **Planned** | CICTT + discovered topics → 3-level hierarchy |
| Phase 6B-Review | **Planned** | **GATE 2**: Human taxonomy review & seed selection |
| Phase 6C - Scoring | **Planned** | Multi-signal cause attribution (percentage allocation) |
| Phase 6C-Review | **Planned** | **GATE 3**: Human score review (50 reports) |
| Phase 7 - Trends | Planned | Prevalence analytics by time period |
| Phase 8 - Streamlit App | Planned | Search + Cause Map Explorer interface |

---

## Key Results

### Benchmark Performance (v2)

| Model | MRR | Hit@10 | Semantic Precision | Semantic Lift |
|-------|-----|--------|-------------------|---------------|
| MiniLM | 0.704 | 100% | 92.7% | +28.2% |
| **MIKA** | **0.816** | **100%** | **97.1%** | **+38.6%** |

**Recommendation:** MIKA (NASA's aviation-domain model) achieves 38.6% semantic lift over keyword baseline, demonstrating the value of domain-specific embeddings for specialized corpora.

### v1 to v2 Improvements

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| MRR (MIKA) | 0.788 | 0.816 | +3.5% |
| Hit@10 | 94.9% | 100% | +5.1% |
| Chunks in target range | 35% | 95.6% | +60.6% |

The v2 chunking strategy (400-800 tokens with section prefixes and 25% overlap) dramatically improved both chunk quality and retrieval performance.

---

## Architecture

```
                                 RiskRADAR Architecture

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   NTSB      │     │   Extract   │     │   Chunk     │     │   Embed     │
    │   Website   │────▶│   + OCR     │────▶│   + Index   │────▶│   + Upload  │
    │  (510 PDFs) │     │  (30K pages)│     │ (24K chunks)│     │  (Qdrant)   │
    └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │                   │
          ▼                   ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   SQLite    │     │   JSON/     │     │   Parquet   │     │   Vector    │
    │  (metadata) │     │   JSONL     │     │  (analytics)│     │   Search    │
    └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Git
- Tesseract OCR (for PDF processing)
- Qdrant Cloud account (free tier sufficient)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/riskRADAR.git
cd riskRADAR

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
python -m pip install -r requirements.txt
python -m pip install -e ./scraper
```

### Environment Setup

Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Required for Phase 5 (Embeddings) - get from https://cloud.qdrant.io/
QDRANT_URL=https://your-cluster-id.region.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_api_key_here

# Required for PDF access - path to your PDF storage
# Default assumes NAS at \\TRUENAS\Photos\RiskRADAR
# RISKRADAR_NAS_PATH=\\TRUENAS\Photos\RiskRADAR

# Optional - adjust if needed
# RISKRADAR_DB_PATH=sqlite/riskradar.db
# RISKRADAR_LOG_DIR=logs
```

**Getting Qdrant credentials:**
1. Create account at https://cloud.qdrant.io/
2. Create a cluster (free tier is sufficient)
3. Go to **API Keys** tab → Create new key
4. Copy the cluster URL and API key to `.env`

### Verify Setup

```bash
python -m scripts.verify_setup
```

### Run Pipeline

The pipeline has 5 phases that must be run in order:

```bash
# Phase 1-2: Scrape PDFs from NTSB (only if PDFs not already downloaded) and loads metadata into SQLite database. 
# See scraper/readme.md for details - this downloads 510 PDFs to NAS
python -m scraper.ntsb_scraper  # Takes ~1 hour with rate limiting

# Phase 3: Extract text from PDFs (requires NAS access to PDFs)
python -m extraction.processing.extract all

# Phase 4: Chunk documents into search-ready segments
python -m extraction.processing.chunk all

# Phase 5: Generate embeddings and upload to Qdrant
python -m embeddings.cli all

# Run benchmark evaluation
python -m eval.benchmark run
```

**Note:** Phase 1-2 (scraping) has already been completed and the 510 PDFs are stored on the NAS in my configuration. New users need NAS access or other storage configured in their environment to run extraction.

---

## Requirements

### System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.9 or higher |
| RAM | 8GB minimum (16GB recommended for MIKA) |
| Storage | ~2GB for embeddings + source data |
| Tesseract | 4.0+ for OCR processing |
| Poppler | For PDF to image conversion |

### Python Dependencies

Core dependencies (see `requirements.txt` for full list):

- **Data Processing:** pandas, numpy, pyarrow
- **PDF Extraction:** pymupdf, pytesseract, pdf2image
- **Embeddings:** sentence-transformers, torch
- **Vector Database:** qdrant-client
- **Analytics:** duckdb
- **Web Scraping:** selenium, webdriver-manager
- **Tokenization:** tiktoken

### External Services

| Service | Purpose | Tier |
|---------|---------|------|
| Qdrant Cloud | Vector database hosting | Free tier (1M vectors) |
| HuggingFace | Model downloads | Free |

---

## Module Documentation

Each module has detailed documentation covering usage, API reference, and examples:

| Module | Description | Documentation |
|--------|-------------|---------------|
| **riskradar** | Core configuration and shared utilities | [riskradar/README.md](riskradar/README.md) |
| **sqlite** | Database schema and query layer | [sqlite/README.md](sqlite/README.md) |
| **scraper** | Selenium-based web scraping library | [scraper/readme.md](scraper/readme.md) |
| **extraction** | PDF extraction and chunking pipeline | [extraction/README.md](extraction/README.md) |
| **embeddings** | Embedding generation and Qdrant upload | [embeddings/README.md](embeddings/README.md) |
| **eval** | Benchmark framework and evaluation | [eval/README.md](eval/README.md) |
| **taxonomy** | Cause attribution and taxonomy scoring | [taxonomy/README.md](taxonomy/README.md) |
| **analytics** | DuckDB analytics and SQL interface | [analytics/README.md](analytics/README.md) |
| **app** | Streamlit search and explorer interface | [app/README.md](app/README.md) |

---

## Data Pipeline

### Phase 1-2: Web Scraping

```bash
# Only needed if PDFs are not already downloaded
python -m scraper.ntsb_scraper
```

- Scrapes 510 aviation accident reports from [NTSB website](https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx)
- Downloads PDFs to configured NAS storage path
- Extracts metadata (title, location, dates, report number) from web pages
- Stores metadata in SQLite `reports` table
- Rate limited (2-3s delays) to respect robots.txt
- **Runtime:** ~1 hour with rate limiting

See [scraper/readme.md](scraper/readme.md) for detailed documentation.

**Note:** This phase has been completed. The 510 PDFs are stored on NAS and metadata is in the database.

### Phase 3: Text Extraction

```bash
python -m extraction.processing.extract initial  # Embedded text
python -m extraction.processing.extract ocr      # OCR for failed pages
```

- Extracts text from 510 PDFs (30,602 pages)
- Quality metrics: alphabetic ratio, garbage ratio, word count
- Automatic OCR fallback for scanned/image PDFs
- Per-page JSON output with full lineage
- **Requires:** NAS access to PDF files

### Phase 4: Chunking

```bash
python -m extraction.processing.chunk all
```

- Three-pass pipeline: pages → documents → chunks
- Section-aware chunking with header detection
- Token range: 400-800 (target 600) with 25% overlap
- Footnote extraction and linking
- Output: 24,766 search-ready chunks

### Phase 5: Embeddings

```bash
python -m embeddings.cli embed both   # Generate embeddings (~3.5 hours)
python -m embeddings.cli upload both  # Upload to Qdrant
python -m embeddings.cli verify both  # Verify collections
```

Two embedding models for comparison:

| Model | Dimensions | Purpose |
|-------|------------|---------|
| MiniLM | 384 | General-purpose baseline |
| MIKA | 768 | NASA aviation-domain model |

---

## Evaluation Framework

The benchmark framework evaluates retrieval quality across 50 stratified queries:

| Category | Count | Purpose |
|----------|-------|---------|
| Incident Lookup | 10 | Known accidents (Alaska 261, ValuJet 592) |
| Conceptual Queries | 12 | Technical concepts (CRM, CFIT, fatigue) |
| Section Queries | 10 | Structural retrieval (PROBABLE CAUSE) |
| Comparative Queries | 8 | Analytical patterns |
| Aircraft Queries | 6 | Aircraft-type specific |
| Phase Queries | 4 | Flight phase specific |

### Metrics

- **MRR** (Mean Reciprocal Rank): Primary ranking metric
- **Hit@K**: Recall at K=1,3,5,10,20
- **nDCG@10**: Normalized discounted cumulative gain
- **Semantic Precision**: Human-judged relevance
- **Semantic Lift**: Improvement over keyword baseline

### Run Evaluation

```bash
python -m eval.benchmark run              # Automated evaluation
python -m eval.benchmark export-review    # Export for human review
python -m eval.benchmark import-review    # Import completed reviews
python -m eval.benchmark final-report     # Generate final report
python -m eval.benchmark version-compare  # Compare v1 vs v2
```

See [eval/README.md](eval/README.md) for complete methodology documentation.

---

## Taxonomy & Cause Attribution (Phase 6-8)

### Overview

Phase 6-8 builds a hierarchical accident cause taxonomy and assigns multi-cause attribution to each report:

```
Phase 6A: Topic Discovery (BERTopic)
    │
    ▼
Phase 6B: Taxonomy Building (CICTT + Discovered Topics)
    │
    ▼
Phase 6C: Multi-Signal Scoring (Keywords + Embeddings)
    │
    ▼
Phase 7: Trend Analytics (Prevalence over Time)
    │
    ▼
Phase 8: Streamlit App (Search + Cause Map Explorer)
```

### Critical Finding: Section Consistency

Analysis revealed PROBABLE CAUSE sections only exist in 30.4% of reports:

| Section | Coverage |
|---------|----------|
| PROBABLE CAUSE | 30.4% |
| CONCLUSIONS | 50.0% |
| FINDINGS | 54.5% |
| ANALYSIS | **70.2%** |
| Any cause section | **80.6%** |

**Solution:** Hybrid approach - use cause sections when available, fallback to full document.

### Taxonomy Structure

3-level hierarchy using CICTT (NTSB official categories) as foundation:

```
Level 1: CICTT Categories (15 official categories)
  └─ Level 2: Topic Groups (BERTopic-discovered)
      └─ Level 3: Specific Themes (Sub-topics)
```

**CICTT Level 1 Categories:**
- LOC-I (Loss of Control - Inflight)
- CFIT (Controlled Flight Into Terrain)
- SCF-PP/SCF-NP (System/Component Failure)
- ICE, UIMC, FUEL, FIRE, etc.

### Multi-Cause Attribution

Reports can have multiple causes with percentage allocation:
- Percentages sum to 100%
- Evidence-backed (linked to specific chunks)
- Roll-up: L3 → L2 → L1

### Human-in-the-Loop Review

Three explicit approval gates ensure quality:

| Gate | Phase | Review Process |
|------|-------|----------------|
| **Gate 1** | After 6A | Export topics to CSV, review/rename/merge, import decisions |
| **Gate 2** | After 6B | Export taxonomy YAML, review structure, select seed examples |
| **Gate 3** | After 6C | Export 50 scored reports, validate cause assignments |

Pipeline pauses at each gate until you explicitly approve to proceed.

### CLI Commands (Phase 6)

```bash
python -m taxonomy.cli discover          # BERTopic topic discovery
python -m taxonomy.cli export-topics     # Export for Gate 1 review
python -m taxonomy.cli approve-gate 1    # Approve to proceed

python -m taxonomy.cli map               # Map topics to CICTT
python -m taxonomy.cli export-taxonomy   # Export for Gate 2 review
python -m taxonomy.cli approve-gate 2    # Approve to proceed

python -m taxonomy.cli score --model both  # Score all reports
python -m taxonomy.cli export-review     # Export for Gate 3 review
python -m taxonomy.cli approve-gate 3    # Approve to proceed

python -m taxonomy.cli trends            # Compute prevalence trends
```

See CLAUDE.md for detailed implementation plan.

---

## Project Structure

```
riskRADAR/
├── riskradar/           # Core configuration module
│   └── config.py        # Paths, Qdrant settings, environment
├── sqlite/              # Database layer
│   ├── schema.py        # Table definitions (14+ tables)
│   ├── connection.py    # Connection management
│   └── queries.py       # Common SQL operations
├── scraper/             # Web scraping library
│   ├── browser.py       # Chrome driver management
│   ├── actions.py       # Page interactions
│   └── download.py      # File download handling
├── extraction/          # PDF processing pipeline
│   ├── processing/      # Extraction and chunking modules
│   └── json_data/       # Pipeline outputs (gitignored)
├── embeddings/          # Embedding pipeline
│   ├── models.py        # Model wrapper
│   ├── embed.py         # Generation pipeline
│   └── upload.py        # Qdrant integration
├── eval/                # Benchmark framework
│   ├── benchmark.py     # Evaluation runner
│   ├── gold_queries.yaml # 50 test queries
│   └── results/         # Benchmark outputs
├── taxonomy/            # Cause attribution pipeline (Phase 6)
│   ├── taxonomy.yaml    # 3-level taxonomy definition
│   ├── discover.py      # BERTopic topic discovery
│   ├── scorer.py        # Multi-signal scoring
│   └── cli.py           # CLI entry point
├── app/                 # Streamlit application (Phase 8)
│   ├── main.py          # Entry point
│   └── pages/           # Search, taxonomy, dashboard
├── analytics/           # DuckDB analytics
│   ├── cli.py           # Interactive SQL shell
│   └── views.py         # Pre-built analytical views
├── scripts/             # Utility scripts
│   └── verify_setup.py  # Environment verification
├── .env.example         # Environment template
├── requirements.txt     # Python dependencies
├── CLAUDE.md            # Development context and phase plans
├── PORTFOLIO.md         # Project narrative and learnings
└── README.md            # This file
```

---

## Technologies

### Core Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Language | Python 3.9+ | Primary development |
| Database | SQLite | Metadata and run tracking |
| Analytics | DuckDB + Parquet | Ad-hoc SQL queries |
| Vector DB | Qdrant Cloud | Similarity search |
| Embeddings | sentence-transformers | Text vectorization |

### Models

| Model | Source | Dimensions |
|-------|--------|------------|
| all-MiniLM-L6-v2 | sentence-transformers | 384 |
| MIKA_Custom_IR | NASA-AIML | 768 |

### PDF Processing

| Tool | Purpose |
|------|---------|
| pymupdf | Embedded text extraction |
| pytesseract | OCR processing |
| pdf2image | PDF to image conversion |

---

## Contributing

This is a portfolio project. Issues and suggestions are welcome.

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- **NTSB** for making aviation accident reports publicly available
- **NASA-AIML** for the MIKA domain-specific embedding model
- **Qdrant** for the vector database platform

---

*For detailed project narrative, technical challenges, and learnings, see [PORTFOLIO.md](PORTFOLIO.md).*
