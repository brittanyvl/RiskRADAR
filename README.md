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
| Phase 5 - Embeddings | Not Started | |
| Phase 6-8 - App | Not Started | |

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
```

**Note:** Always use `python` (not `py`) when the venv is activated. See [CLAUDE.md](CLAUDE.md) for details.

## Directory Structure

```
RiskRADAR/
├── riskradar/           # Core configuration
│   └── config.py        # DB_PATH, NAS_PATH settings
├── sqlite/              # Database layer
│   ├── riskradar.db     # SQLite database (510 reports)
│   ├── schema.py        # Table definitions
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
├── analytics/           # DuckDB analytics engine
│   ├── convert.py       # JSONL to Parquet conversion
│   ├── cli.py           # Interactive SQL shell
│   └── data/            # Parquet files (gitignored)
├── logs/                # Execution logs
├── CLAUDE.md            # Detailed project context
└── README.md            # This file
```

## Key Technologies

- **Scraping:** Selenium + webdriver-manager
- **PDF Parsing:** pymupdf (embedded text)
- **OCR:** pytesseract + pdf2image
- **Database:** SQLite
- **Analytics:** DuckDB + Parquet
- **Vector DB:** Qdrant Cloud (planned)
- **Embeddings:** sentence-transformers (planned)
- **App:** Streamlit (planned)

## Data Source

- **URL:** https://www.ntsb.gov/investigations/AccidentReports/
- **PDFs:** Stored on NAS at `\\TRUENAS\Photos\RiskRADAR`
- **Database:** `sqlite/riskradar.db`

## Documentation

- [CLAUDE.md](CLAUDE.md) - Detailed project context and phase specifications
- [extraction/README.md](extraction/README.md) - Text extraction pipeline docs
- [analytics/README.md](analytics/README.md) - DuckDB analytics engine docs
- [scraper/readme.md](scraper/readme.md) - Web scraping library docs
