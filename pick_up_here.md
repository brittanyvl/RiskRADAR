# RiskRADAR - Pick Up Here

**Last Updated:** 2026-02-10
**Last Session:** Roadmap updated — analytics-first approach (Phase 7 → Phase 8)

---

## Current Status

### Completed Phases

| Phase | Status | Key Metrics |
|-------|--------|-------------|
| Phase 1: PDF Scraping | Complete | 510 PDFs downloaded |
| Phase 2: Metadata | Complete | Titles, dates, locations captured |
| Phase 3: Text Extraction | Complete | 30,602 pages extracted |
| Phase 4: Chunking v2 | Complete | 24,766 chunks (95.6% in target range) |
| Phase 5: Embeddings | Complete | MiniLM + MIKA models in Qdrant |
| Phase 5b: Benchmark | Complete | MiniLM recommended (75.5% precision) |
| Phase 6A: L1 Taxonomy | Complete | 27 CICTT categories, 453 reports classified |
| Phase 6A-Sub: L2 Taxonomy | Complete | 32 subcategories, 1,106 assignments |
| Qdrant Enrichment | Complete | Payloads enriched with taxonomy + PDF URLs |
| Aircraft Extraction | Complete | 92.7% coverage (473/510 reports) |
| Report Type Classification | Complete | 510 reports classified (436 accident, 74 other) |
| BM25 + Hybrid Search | Complete | BM25, semantic, and hybrid (RRF) search |
| Weather Extraction | Complete | 72.4% coverage (369/510 reports) |
| Time-of-Day Extraction | Complete | 90.8% coverage (463/510 reports) |
| Bayesian Risk Model v1 | Complete | Initial softmax NB (superseded by v2) |
| **Bayesian Risk Model v2** | **Complete** | **Binary Relevance NB, ECE=0.021, Hit@5=86.8%** |
| **Statistical Audit** | **Complete** | **4 critical flaws found and fixed** |
| **Production Validation** | **Complete** | **Calibration, ablation, discrimination all passing** |

### Data Quality Summary

| Feature | Count | Coverage |
|---------|-------|----------|
| Aircraft Category | 473 | 92.7% |
| Aircraft Make | 473 | 92.7% |
| Region | 461 | 90.4% |
| Season | 489 | 95.9% |
| Weather (VMC/IMC) | 369 | 72.4% |
| Time of Day | 463 | 90.8% |
| L1 Taxonomy | 453 | 88.8% |
| L2 Taxonomy | ~400 | ~78% |

### Bayesian Model Results (v2 — Binary Relevance)

- **Algorithm:** Binary Relevance Naive Bayes (27 independent binary classifiers)
- **Training data:** 431 accident reports (filtered via `report_types`)
- **Features:** aircraft_category, season, region, weather_category, time_of_day
- **Schema:** v8 (positive_count in priors, label column in likelihoods PK)

**Validation (proper LOO with baseline):**

| Metric | Model | Baseline | Lift |
|--------|-------|----------|------|
| Hit@1 | 44.8% | 45.9% | 0.97x |
| Hit@3 | 76.3% | 81.4% | 0.94x |
| Hit@5 | 86.8% | 85.4% | 1.02x |
| ECE | 0.021 | — | — |

**Key properties:**
- Probabilities are independent per category (do NOT sum to 1; sum ≈ 4.19)
- Risk thresholds: HIGH > 67.1%, MODERATE > 54.3%
- All 27 categories have positive discrimination (mean separation = +0.075)
- Save/load roundtrip is lossless
- Unseen values handled via proper Laplace smoothing

### What the v1 → v2 Audit Fixed

| Flaw | v1 Problem | v2 Fix |
|------|-----------|--------|
| Softmax on multi-label | 95.8% multi-label data forced to sum-to-1 | Independent sigmoid per category |
| Fake LOO | Full model reused without retraining | Count-adjusted proper LOO |
| No baseline | Model worse than prior-only at Hit@3 | Explicit baseline printed |
| Unseen value fallback | Arbitrary α/100 | Proper Laplace with n+1 |

### Weather Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| VMC | 164 | 32.2% |
| IMC | 205 | 40.2% |
| Unknown | 141 | 27.6% |

### Time-of-Day Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| Morning | 106 | 20.8% |
| Afternoon | 159 | 31.2% |
| Evening | 57 | 11.2% |
| Night | 141 | 27.6% |
| Unknown | 47 | 9.2% |

### Key Files

| File | Purpose |
|------|---------|
| `sqlite/riskradar.db` | Main SQLite database (schema v8) |
| `sqlite/schema.py` | Schema definitions (BAYES_PRIORS_TABLE, BAYES_LIKELIHOODS_TABLE) |
| `risk_profiler/bayesian_model.py` | Binary Relevance NB with persistence + proper LOO validation |
| `risk_profiler/extract_weather.py` | VMC/IMC extraction from chunk text |
| `risk_profiler/extract_time.py` | Time-of-day extraction from chunk text |
| `risk_profiler/extract_features.py` | Feature extraction pipeline |
| `risk_profiler/aircraft_data.py` | Aircraft lookup with 686 patterns |
| `risk_profiler/report_types.py` | Report type classification (accident vs other) |
| `risk_profiler/cli.py` | CLI: train-model, validate-model, extract-weather, extract-time |
| `search/` | BM25 + semantic + hybrid (RRF) search module |
| `app/pages/4_Risk_Profiler.py` | Streamlit risk profiler page (binary relevance, 5 feature dropdowns) |
| `taxonomy/` | CICTT classification system |
| `embeddings/` | Vector embedding pipeline |
| `portfolio.md` | Full model evolution narrative (audit, rewrite, validation) |

---

## TODO List (Priority Order)

### Phase 7: Stakeholder Analytics (NEXT — do BEFORE Streamlit)

Build a formal analytics layer with reusable DuckDB views, Parquet exports, and query modules. This must be completed before building the Streamlit app so dashboards have a solid analytical foundation.

**Analytics layer (`analytics/`):**
- [ ] **Core shared analytics module**
  - Co-occurrence matrix (category × category)
  - Trend aggregations (category prevalence over time)
  - Risk score calculations per stakeholder dimension
  - Parquet exports for each stakeholder view
  - Reusable SQL views for dashboards

- [ ] **Manufacturer's Risk Summary analytics**
  - Risk breakdown by aircraft_category + aircraft_make
  - SCF-PP / SCF-NP prevalence by manufacturer/type
  - Failure mode trends (L2 subcategories for SCF-PP, SCF-NP)
  - Data sources: aircraft_category, aircraft_make, L1/L2 taxonomy

- [ ] **Maintenance Risk Summary analytics**
  - SCF-PP / SCF-NP deep dive (engine, fuel, hydraulic, electrical, structural)
  - Component failure patterns from L2 subcategories
  - Cross-tabulation with aircraft_category and season
  - Data sources: L1/L2 taxonomy, aircraft_category, season

- [ ] **Insurance Risk Summary analytics**
  - Category co-occurrence matrix (which risks cluster together)
  - Risk profiles by region × season × weather
  - Severity proxies (multi-category incidents as complexity indicator)
  - Data sources: all features + taxonomy + report metadata

- [ ] **Pilot's Risk Summary analytics**
  - LOC-I / CFIT / UIMC profiles by aircraft type
  - Weather × time-of-day risk matrix
  - Seasonal patterns for human-factors categories
  - Data sources: aircraft_category, weather_category, time_of_day, L1/L2 taxonomy

### Phase 8: Streamlit Application (AFTER analytics)

Build after Phase 7 analytics are developed. Each page consumes the analytics layer.

- [ ] **Semantic Search page**
  - Hybrid search (BM25 + semantic + RRF) via `search/` module
  - Taxonomy filter dropdowns (L1, L2)
  - Aircraft/region/date filters
  - Results with relevance scores + PDF links

- [ ] **Taxonomy Explorer page**
  - Category distribution charts (L1 and L2)
  - Drill-down from L1 to L2
  - Report list per category

- [ ] **Manufacturer Dashboard page** — from Phase 7 analytics
- [ ] **Maintenance Dashboard page** — from Phase 7 analytics
- [ ] **Insurance Dashboard page** — from Phase 7 analytics
- [ ] **Pilot Dashboard page** — from Phase 7 analytics

- [ ] **Finalize Risk Profiler page** (exists, needs polish)
  - Test end-to-end with all 5 feature dropdowns
  - Add visualization of risk distribution (bar chart of posteriors)
  - Add confidence indicators for each prediction

### Future: API & Advanced Features

- [ ] **FastAPI backend** (if needed beyond Streamlit)
  - Semantic search endpoint
  - Bayesian risk scoring endpoint
  - KPI query endpoints

- [ ] **Model improvements**
  - Experiment with feature interactions (e.g., weather x time_of_day)
  - Try hierarchical Bayes with L2 subcategories
  - Improve weather extraction coverage (currently 72.4%)
  - Consider adding aircraft_make as a Bayesian model feature

---

## Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Taxonomy | CICTT L1 + Industry L2 | Industry standard, interpretable |
| Embedding Model | MiniLM (production) | Better semantic precision (75.5% vs 60%) |
| Search | Vector + BM25 hybrid (RRF) | Best retrieval performance |
| Weather Feature | VMC vs IMC (binary) | NTSB standard; specific phenomena are outcome categories |
| Time Feature | 4 buckets (Morning/Afternoon/Evening/Night) | Simple, no external dependencies |
| Model Training | Accident-only (431 reports) | Excludes 74 non-accident docs (safety studies, etc.) |
| **Model Architecture** | **Binary Relevance NB (v2)** | **Fixed 4 critical flaws: softmax on multi-label, fake LOO, no baseline, arbitrary unseen values** |
| Risk Thresholds | Data-driven (percentile-based) | 90th percentile = HIGH (67.1%), 50th = MODERATE (54.3%) |
| Model Persistence | SQLite tables (v8 schema) | Fast Streamlit loading via `load_model()` |
| Calibration Metric | ECE (Expected Calibration Error) | Standard metric for probability reliability; model achieves 0.021 |
| Analytics Before App | Phase 7 (analytics) → Phase 8 (Streamlit) | Build formal analytics layer with DuckDB views before dashboards |
| Stakeholder Dashboards | 4 dashboards (Manufacturer, Maintenance, Insurance, Pilot) | Each stakeholder has distinct analytical focus and use case |
| Aircraft Data | Use existing aircraft_category + aircraft_make | No new extractions; 92.7% coverage is sufficient |

---

## CLI Quick Reference

```bash
# Activate environment
cd C:\Users\bvlma\CODE\riskRADAR
venv\Scripts\activate

# Risk Profiler commands
python -m risk_profiler.cli classify-types [--dry-run]
python -m risk_profiler.cli load-chunks
python -m risk_profiler.cli extract-weather [--jsonl path]
python -m risk_profiler.cli extract-time [--jsonl path]
python -m risk_profiler.cli train-model [--features f1,f2,...] [--skip-validate]
python -m risk_profiler.cli validate-model [--features f1,f2,...]

# Search commands
python -m search.cli build-index|search|benchmark|stats

# Taxonomy commands
python -m taxonomy.cli classify|categories|subcategories|stats
python -m taxonomy.cli retry-unclassified [--run-id 2]

# Quick model verification
python -c "from risk_profiler.bayesian_model import load_model; m=load_model(); print(m.predict(top_k=5, aircraft_category='turboprop', weather_category='IMC'))"

# Check DB schema version
python -c "import sqlite3; c=sqlite3.connect('sqlite/riskradar.db'); print('label' in [r[1] for r in c.execute('PRAGMA table_info(bayes_likelihoods)').fetchall()])"
```

---

## Notes for Next Session

1. **Analytics FIRST, then Streamlit** — build the formal analytics layer (Phase 7) before building the app (Phase 8)
2. Phase 7 = DuckDB views + Parquet exports + reusable query modules for 4 stakeholder dashboards
3. Data sources for analytics: existing features (aircraft_category, aircraft_make, season, region, weather_category, time_of_day) + taxonomy (L1, L2) + report metadata — **no new extractions needed**
4. Risk Profiler page already exists and works with binary relevance model — just needs polish
5. Search module (`search/`) is complete — just needs Streamlit integration in Phase 8
6. All extraction pipelines are complete — no more feature engineering unless improving coverage
7. Schema is at version 8 — includes binary relevance bayes tables (label + positive_count)
8. The Bayesian model is **production-ready** — audited, rewritten, and validated with proper LOO, ECE=0.021, and baseline comparison
9. Portfolio.md has the full model evolution narrative (initial → audit → rewrite → production validation)
10. CLAUDE.md has the full roadmap with Phase 7/8 details
