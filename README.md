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
- [References & Citations](#references--citations)

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
| Phase 6A - L1 Classification | **Complete** | 453 reports → 27 CICTT categories |
| Phase 6A-Sub - L2 Classification | **Complete** | 1,106 report-L2 assignments → 32 subcategories |
| Phase 6A - Qdrant Enrichment | **Complete** | Payloads enriched with taxonomy + PDF URLs |
| Phase 6A-Review | **Pending** | Human review of L1+L2 classifications |
| Phase 6C - Scoring | Planned | Multi-signal cause attribution (percentage allocation) |
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

Phase 6-8 builds a hierarchical accident cause taxonomy and assigns multi-cause attribution to each report.

**Key Learning:** Initial BERTopic unsupervised topic discovery produced 76 topics, but human review revealed they captured document structure (headers, boilerplate) rather than meaningful safety factors. This led to a pivot to industry-standard taxonomies.

```
Phase 6A: CICTT Level 1 Classification (Complete - 453 reports → 27 categories)
    │
    ▼
Phase 6A-Sub: Level 2 Sub-Categorization (Complete - 1,106 assignments → 32 subcategories)
    │
    ▼
Qdrant Enrichment (Complete - payloads enriched with l1_categories, l2_subcategories, pdf_url)
    │
    ▼
Phase 6A-Review: Human Taxonomy Review (GATE - Pending)
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

### Taxonomy Structure

2-level hybrid hierarchy using industry standards:

```
Level 1: CICTT Categories (27 categories from CAST/ICAO)
    │
    └─ Level 2: Sub-Categories (48 total, targeted by category type)
        ├── LOC-I → Industry-standard sub-causes (IATA/EASA research)
        ├── CFIT → Industry-standard sub-causes (IATA/SKYbrary)
        ├── SCF-PP/SCF-NP → Technical sub-systems
        └── Human-causal categories → HFACS Unsafe Acts
```

### Level 1: CICTT Categories (Complete)

453 reports classified to 27 CICTT occurrence categories using embedding similarity.

**Key Categories:**
| Code | Name | Description |
|------|------|-------------|
| LOC-I | Loss of Control - Inflight | Stalls, spins, spatial disorientation |
| CFIT | Controlled Flight Into Terrain | Impact while aircraft under control |
| SCF-PP | System Failure - Powerplant | Engine, propeller, rotor failures |
| SCF-NP | System Failure - Non-Powerplant | Flight controls, hydraulics, electrical |
| RE | Runway Excursion | Overruns, veer-offs |
| UIMC | VFR into IMC | Inadvertent instrument conditions |

### Level 2: Sub-Categories (Complete)

Targeted sub-categorization using domain-specific frameworks.

**L2 Classification Results:**
- 1,478 chunks processed (from L2-enabled categories)
- 2,446 chunk-L2 assignments generated
- 1,106 report-L2 assignments
- 32 subcategories used

**Sub-category frameworks:**

| L1 Category | Sub-Category Source | Sub-Categories |
|-------------|---------------------|----------------|
| **LOC-I** | IATA/EASA research | STALL, UPSET, SD (Spatial Disorientation), ENV, SYS, LOAD |
| **CFIT** | IATA/SKYbrary | NAV (Navigation), SA (Situational Awareness), VIS, TAWS, PROC |
| **SCF-PP** | Technical sub-systems | ENG, FUEL, PROP, FIRE |
| **SCF-NP** | Technical sub-systems | FLT (Flight Controls), HYD, ELEC, STRUCT, GEAR |
| **Human-causal** (7 categories) | HFACS Level 1 | SKILL, DECISION, PERCEPTUAL, VIOLATION |
| **Other** (13 categories) | None | Remain flat (low volume or already specific) |

### Classification Method

Two-pass embedding-based classification:

```
Pass 1: Chunk → L1 CICTT (cosine similarity to CICTT seed phrases)
Pass 2: Chunk + L1 → L2 Sub-category (cosine similarity to L2 seed phrases)
Combined Confidence: L1_confidence × L2_confidence
```

Uses NASA MIKA embeddings (768 dimensions) for domain-specific semantic matching.

### Human-in-the-Loop Review

| Gate | Phase | Review Process |
|------|-------|----------------|
| **Gate 1** | After 6A | ✅ Complete - CICTT L1 classification approved |
| **Gate 2** | After 6A-Sub | Export 50 sub-category assignments, validate accuracy |
| **Gate 3** | After 6C | Export scored reports, validate multi-cause attribution |

### CLI Commands (Phase 6)

```bash
# Level 1 CICTT Classification (Complete)
python -m taxonomy.cli map               # Map reports to CICTT categories
python -m taxonomy.cli review            # Export HTML for human review
python -m taxonomy.cli stats             # Show classification statistics

# Level 2 Sub-Categorization (In Progress)
python -m taxonomy.cli classify          # Hierarchical L1+L2 classification
python -m taxonomy.cli export-review     # Export for Gate 2 review
python -m taxonomy.cli import-review     # Import completed reviews
python -m taxonomy.cli review-stats      # Show approval/rejection rates

# Trend Analytics (Phase 7)
python -m taxonomy.cli trends            # Compute prevalence over time

# Qdrant Payload Enrichment (adds taxonomy to vector payloads)
python -m embeddings.cli enrich both     # Adds l1_categories, l2_subcategories, pdf_url
```

### Qdrant Enrichment (Complete)

Vector payloads in Qdrant have been enriched with taxonomy data for category-filtered search:

| Field | Type | Example |
|-------|------|---------|
| `l1_categories` | list[string] | `["LOC-I", "SCF-NP"]` |
| `l2_subcategories` | list[string] | `["LOC-I-STALL", "CFIT-NAV"]` |
| `pdf_url` | string | `https://www.ntsb.gov/.../AAR0201.pdf` |

This enables:
- **Category-filtered semantic search** in Streamlit
- **Direct PDF linking** to original NTSB reports
- **Faceted navigation** by taxonomy categories

See [CLAUDE.md](CLAUDE.md) for detailed implementation plan and [portfolio.md](portfolio.md) for the BERTopic → CICTT pivot narrative.

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

## References & Citations

This project incorporates research and standards from multiple aviation safety organizations. Below are the primary sources used in designing the taxonomy and classification systems.

### Aviation Taxonomy Standards

#### CICTT (CAST/ICAO Common Taxonomy Team)

The primary taxonomy framework used for Level 1 occurrence classification:

- **NTSB. (2013).** *Aviation Occurrence Categories: Definitions and Usage Notes (Version 4.6)*. National Transportation Safety Board.
  - URL: https://www.ntsb.gov/safety/data/Documents/datafiles/OccurrenceCategoryDefinitions.pdf
  - Used for: 28 CICTT occurrence category definitions

- **CAST-ICAO. (2017).** *CICTT Aviation Occurrence Categories (Version 4.7)*. Commercial Aviation Safety Team / International Civil Aviation Organization.
  - URL: https://www.cast-safety.org/
  - Used for: Category keywords and classification guidance

- **SKYbrary. (2024).** *CAST/ICAO Common Taxonomy Team (CICTT)*. SKYbrary Aviation Safety.
  - URL: https://skybrary.aero/articles/casticao-common-taxonomy-team-cictt
  - Used for: Taxonomy overview and implementation guidance

#### HFACS (Human Factors Analysis and Classification System)

Used for Level 2 human factors sub-categorization:

- **Shappell, S.A. & Wiegmann, D.A. (2000).** *The Human Factors Analysis and Classification System (HFACS)*. DOT/FAA/AM-00/7. Federal Aviation Administration.
  - URL: https://rosap.ntl.bts.gov/view/dot/21482
  - Used for: 4-level HFACS framework (Unsafe Acts, Preconditions, Unsafe Supervision, Organizational Influences)

- **Wiegmann, D.A. & Shappell, S.A. (2003).** *A Human Error Approach to Aviation Accident Analysis*. Ashgate Publishing.
  - Used for: HFACS category definitions and application methodology

- **SKYbrary. (2024).** *Human Factors Analysis and Classification System (HFACS)*. SKYbrary Aviation Safety.
  - URL: https://skybrary.aero/articles/human-factors-analysis-and-classification-system-hfacs
  - Used for: HFACS Level 1 sub-categories (Skill-Based, Decision, Perceptual Errors; Violations)

- **HFACS, Inc. (2024).** *The HFACS Framework*.
  - URL: https://www.hfacs.com/hfacs-framework.html
  - Used for: Framework structure and category hierarchy

#### ECCAIRS/ADREP

European occurrence reporting taxonomy (referenced for comparison):

- **ICAO. (2024).** *ADREP Taxonomy*. International Civil Aviation Organization.
  - URL: https://www.icao.int/safety/airnavigation/AIG/Pages/Taxonomy.aspx
  - Used for: Understanding international aviation taxonomy standards

- **EASA. (2024).** *ECCAIRS 2 Taxonomy Browser*. European Union Aviation Safety Agency.
  - URL: https://aviationreporting.eu/en/taxonomy-browser
  - Used for: Multi-level taxonomy structure reference

### Loss of Control In-Flight (LOC-I) Research

Used for LOC-I Level 2 sub-categorization:

- **IATA. (2015).** *Loss of Control In-Flight Accident Analysis Report (1st Edition)*. International Air Transport Association.
  - URL: https://flightsafety.org/wp-content/uploads/2017/07/IATA-LOC-I-1st-Ed-2015.pdf
  - Used for: LOC-I contributing factors taxonomy (Environmental, Pilot-Induced, Systems-Induced)

- **IATA. (2019).** *Loss of Control In-Flight Accident Analysis Report (2019 Edition)*. International Air Transport Association.
  - URL: https://www.iata.org/contentassets/b6eb2adc248c484192101edd1ed36015/loc-i_2019.pdf
  - Used for: Updated LOC-I statistics and Threat Error Management (TEM) framework

- **EASA. (2024).** *Loss of Control (LOC-I)*. European Union Aviation Safety Agency.
  - URL: https://www.easa.europa.eu/en/domains/general-aviation/flying-safely/loss-of-control
  - Used for: LOC-I causes (stall, upset, spatial disorientation, environmental, automation)

- **NASA. (2016).** *Aircraft Loss of Control: Problem Analysis for the Development and Validation of Technology Solutions*. NASA Technical Reports Server.
  - URL: https://ntrs.nasa.gov/api/citations/20160007744/downloads/20160007744.pdf
  - Used for: Three major contributing factor categories (adverse onboard conditions, external hazards, abnormal flight conditions)

- **Flight Safety Foundation. (2024).** *Loss of Control-In Flight (LOC-I) Archives*.
  - URL: https://flightsafety.org/safety-issue/loc-i/
  - Used for: LOC-I safety research and prevention strategies

### Controlled Flight Into Terrain (CFIT) Research

Used for CFIT Level 2 sub-categorization:

- **IATA. (2018).** *Controlled Flight Into Terrain Accident Analysis Report (2008-2017 Data)*. International Air Transport Association.
  - URL: https://www.iata.org/contentassets/06377898f60c46028a4dd38f13f979ad/cfit-report.pdf
  - Used for: CFIT contributing factors (navigation error, situational awareness, visibility)

- **SKYbrary. (2024).** *Controlled Flight Into Terrain (CFIT)*. SKYbrary Aviation Safety.
  - URL: https://skybrary.aero/articles/controlled-flight-terrain-cfit
  - Used for: CFIT causes and TAWS/GPWS effectiveness

- **FAA. (2022).** *Controlled Flight Into Terrain*. Federal Aviation Administration.
  - URL: https://www.faa.gov/sites/faa.gov/files/2022-01/Controlled%20Flight%20into%20Terrain.pdf
  - Used for: CFIT prevention and contributing factors

- **ICAO. (2025).** *Controlled Flight Into Terrain (CFIT): An Aviation Safety Challenge*. IBOM-Air presentation.
  - URL: https://www.icao.int/sites/default/files/WACAF/MeetingDocs/2025/CFIT%20Workshop/
  - Used for: CFIT causes and mitigations

### Topic Modeling & NLP

Used for unsupervised topic discovery (Phase 6A evaluation):

- **Grootendorst, M. (2022).** *BERTopic: Neural Topic Modeling with a Class-based TF-IDF Procedure*. arXiv:2203.05794.
  - URL: https://maartengr.github.io/BERTopic/
  - Used for: Initial topic discovery attempt (led to CICTT pivot)

- **BERTopic Documentation. (2024).** *Guided Topic Modeling*.
  - URL: https://maartengr.github.io/BERTopic/getting_started/guided/guided.html
  - Used for: Seed word and guided topic modeling techniques

- **BERTopic Documentation. (2024).** *Seed Words*.
  - URL: https://maartengr.github.io/BERTopic/getting_started/seed_words/seed_words.html
  - Used for: ClassTfidfTransformer seed word implementation

### Embedding Models

- **NASA-AIML. (2023).** *MIKA_Custom_IR: Aviation Domain Information Retrieval Model*. Hugging Face.
  - URL: https://huggingface.co/NASA-AIML/MIKA_Custom_IR
  - Used for: Primary embedding model (768 dimensions, aviation-domain trained)

- **Reimers, N. & Gurevych, I. (2019).** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
  - URL: https://www.sbert.net/
  - Used for: MiniLM baseline model and sentence-transformers library

### ICAO Safety Reports

- **ICAO. (2024).** *Safety Report 2024 Edition*. International Civil Aviation Organization.
  - URL: https://www.icao.int/sites/default/files/sp-files/safety/Documents/ICAO_SR_2024.pdf
  - Used for: Global aviation safety statistics and high-risk categories

- **ICAO. (2025).** *Safety Report 2025 Edition*. International Civil Aviation Organization.
  - URL: https://www.icao.int/sites/default/files/sp-files/safety/Documents/ICAO_SR_2025.pdf
  - Used for: Updated CICTT occurrence category references

### Data Source

- **NTSB. (2024).** *Aviation Accident Reports*. National Transportation Safety Board.
  - URL: https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx
  - Used for: Source of 510 aviation accident report PDFs (1966-present)

---

*For detailed project narrative, technical challenges, and learnings, see [PORTFOLIO.md](PORTFOLIO.md).*
