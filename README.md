# RiskRADAR

**R**etrieval and **D**iscovery of **A**viation **A**ccident **R**eports

An end-to-end data + ML project that transforms unstructured NTSB aviation accident PDFs into searchable, structured, explainable safety insights using embeddings, a vector database, and a hierarchical taxonomy.

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 - Scraping | Complete | 510 PDFs downloaded from NTSB |
| Phase 2 - Metadata | Complete | Report metadata extracted |
| Phase 3 - Text Extraction | In Progress | OCR pipeline running |
| Phase 4 - Chunking | Not Started | |
| Phase 5 - Embeddings | Not Started | |
| Phase 6-8 - App | Not Started | |

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd RiskRADAR
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -e ./scraper

# Run extraction pipeline (requires NAS access)
py -m extraction.processing.extract initial --limit 10
py -m extraction.processing.extract ocr
```

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
├── extraction/          # PDF text extraction
│   ├── processing/      # Pipeline modules
│   └── json_data/       # Extraction results
├── logs/                # Execution logs
├── CLAUDE.md            # Detailed project context
└── README.md            # This file
```

## Key Technologies

- **Scraping:** Selenium + webdriver-manager
- **PDF Parsing:** pymupdf (embedded text)
- **OCR:** pytesseract + pdf2image
- **Database:** SQLite
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
- [scraper/readme.md](scraper/readme.md) - Web scraping library docs
