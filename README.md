# RiskRADAR

**R**etrieval and **D**iscovery of **A**viation **A**ccident **R**eports

An end-to-end data + ML project that transforms unstructured NTSB aviation accident PDFs into searchable, structured, explainable safety insights using embeddings, a vector database, and a hierarchical taxonomy.

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 - Scraping | Complete | 510 PDFs downloaded from NTSB |
| Phase 2 - Metadata | Complete | Report metadata extracted |
| Phase 3 - Text Extraction | Complete | 30,602 pages extracted (14K embedded + 16K OCR) |
| Phase 4 - Chunking | Complete | 28,321 search-ready chunks created |
| Phase 5 - Embeddings | **In Progress** | Embedding pipeline built, benchmark ready |
| Phase 6-8 - App | Not Started | |

### Phase 5 Status (Current)

- Embedding pipeline implemented for MiniLM (384d) and MIKA (768d)
- Qdrant Cloud integration complete
- Professional benchmark framework with 50 stratified queries
- Awaiting embedding generation and benchmark execution

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd RiskRADAR
python -m venv venv
venv\Scripts\activate  # Windows (use `source venv/bin/activate` on Linux/Mac)
python -m pip install -r requirements.txt
python -m pip install -e ./scraper

# Run extraction pipeline (requires NAS access)
python -m extraction.processing.extract initial --limit 10
python -m extraction.processing.extract ocr

# Run chunking pipeline (Phase 4)
python -m extraction.processing.chunk all

# Setup analytics (DuckDB)
python -m analytics.convert    # Convert JSONL to Parquet
python -m analytics.cli        # Launch interactive SQL shell

# Phase 5: Embeddings (requires Qdrant credentials in .env)
python -m embeddings.cli embed both      # Generate embeddings
python -m embeddings.cli upload both     # Upload to Qdrant
python -m eval.benchmark run             # Run benchmark
```

**Note:** Always use `python` (not `py`) when the venv is activated. See [CLAUDE.md](CLAUDE.md) for details.

## Directory Structure

```
RiskRADAR/
├── riskradar/           # Core configuration
│   └── config.py        # DB_PATH, NAS_PATH, Qdrant settings
├── sqlite/              # Database layer
│   ├── riskradar.db     # SQLite database (510 reports)
│   ├── schema.py        # Table definitions (v4)
│   └── queries.py       # Common queries
├── scraper/             # Web scraping library
├── extraction/          # PDF text extraction + chunking
│   ├── processing/      # Pipeline modules
│   │   ├── extract.py   # Text extraction pipeline
│   │   ├── chunk.py     # Chunking pipeline (Phase 4)
│   │   └── ...          # Supporting modules
│   └── json_data/       # Pipeline outputs
│       ├── passed/      # Embedded text pages
│       ├── ocr_retry/   # OCR processed pages
│       ├── pages.jsonl  # Consolidated pages
│       ├── documents.jsonl  # Full document text
│       └── chunks.jsonl # Search-ready chunks
├── embeddings/          # Embedding pipeline (Phase 5)
│   ├── config.py        # Model registry
│   ├── models.py        # Model wrapper
│   ├── embed.py         # Embedding generation
│   ├── upload.py        # Qdrant upload
│   └── cli.py           # CLI entry point
├── eval/                # Benchmark framework (Phase 5)
│   ├── gold_queries.yaml  # 50 stratified test queries
│   ├── benchmark.py     # Benchmark runner
│   └── results/         # Benchmark outputs
├── embeddings_data/     # Local embeddings (gitignored)
├── analytics/           # DuckDB analytics engine
│   ├── convert.py       # JSONL to Parquet conversion
│   ├── cli.py           # Interactive SQL shell
│   └── data/            # Parquet files (gitignored)
├── logs/                # Execution logs
├── CLAUDE.md            # Detailed project context
├── PHASE5_PLAN.md       # Phase 5 implementation plan
└── README.md            # This file
```

## Key Technologies

- **Scraping:** Selenium + webdriver-manager
- **PDF Parsing:** pymupdf (embedded text)
- **OCR:** pytesseract + pdf2image
- **Database:** SQLite
- **Analytics:** DuckDB + Parquet
- **Vector DB:** Qdrant Cloud
- **Embeddings:** sentence-transformers
  - Baseline: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
  - Domain: `NASA-AIML/MIKA_Custom_IR` (768 dimensions)
- **App:** Streamlit (planned)

## Phase 5 Embedding Pipeline

### CLI Commands

```bash
# Embedding generation
python -m embeddings.cli embed minilm    # MiniLM only
python -m embeddings.cli embed mika      # MIKA only
python -m embeddings.cli embed both      # Both models

# Qdrant upload
python -m embeddings.cli upload minilm
python -m embeddings.cli upload both

# Full pipeline
python -m embeddings.cli all             # Embed + upload both

# Verification
python -m embeddings.cli verify minilm
python -m embeddings.cli stats
```

### Benchmark Framework

```bash
# Run benchmark on both models
python -m eval.benchmark run

# Run single model
python -m eval.benchmark run -m minilm

# Generate comparison report
python -m eval.benchmark report

# Validate query definitions
python -m eval.benchmark validate
```

The benchmark evaluates 50 queries across 6 categories:
- Incident Lookup (10) - Known accidents with specific report IDs
- Conceptual Queries (12) - Technical concepts requiring semantic understanding
- Section Queries (10) - Queries targeting specific report sections
- Comparative Queries (8) - Analytical queries about patterns
- Aircraft Queries (6) - Aircraft-type specific searches
- Phase Queries (4) - Flight phase specific searches

## Data Source

- **URL:** https://www.ntsb.gov/investigations/AccidentReports/
- **PDFs:** Stored on NAS at `\\TRUENAS\Photos\RiskRADAR`
- **Database:** `sqlite/riskradar.db`
- **Vector DB:** Qdrant Cloud (riskradar_minilm, riskradar_mika collections)

## Documentation

- [CLAUDE.md](CLAUDE.md) - Detailed project context and phase specifications
- [PHASE5_PLAN.md](PHASE5_PLAN.md) - Phase 5 embedding pipeline plan
- [extraction/README.md](extraction/README.md) - Text extraction pipeline docs
- [analytics/README.md](analytics/README.md) - DuckDB analytics engine docs
- [eval/README.md](eval/README.md) - Benchmark framework docs
- [scraper/readme.md](scraper/readme.md) - Web scraping library docs
