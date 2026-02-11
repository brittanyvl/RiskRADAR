# RiskRADAR - Pick Up Here

**Last Updated:** 2026-02-10
**Last Session:** Bayesian model refinement + weather/time feature extraction completed

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
| Bayesian Risk Model | Complete | 5 features, Hit@1=54.8%, Hit@5=90.7% |

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

### Bayesian Model Results

- **Training data:** 431 accident reports (filtered via `report_types`)
- **Features:** aircraft_category, season, region, weather_category, time_of_day
- **Validation (LOO):** Hit@1=54.8%, Hit@3=81.4%, Hit@5=90.7%
- **Persistence:** Priors + likelihoods saved to SQLite tables
- **Thresholds:** Data-driven (90th/50th percentile)

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
| `sqlite/riskradar.db` | Main SQLite database (schema v7) |
| `risk_profiler/bayesian_model.py` | Naive Bayes risk model with persistence + validation |
| `risk_profiler/extract_weather.py` | VMC/IMC extraction from chunk text |
| `risk_profiler/extract_time.py` | Time-of-day extraction from chunk text |
| `risk_profiler/extract_features.py` | Feature extraction pipeline |
| `risk_profiler/aircraft_data.py` | Aircraft lookup with 686 patterns |
| `risk_profiler/report_types.py` | Report type classification (accident vs other) |
| `risk_profiler/cli.py` | CLI: train-model, validate-model, extract-weather, extract-time |
| `search/` | BM25 + semantic + hybrid (RRF) search module |
| `app/pages/4_Risk_Profiler.py` | Streamlit risk profiler page (5 feature dropdowns) |
| `taxonomy/` | CICTT classification system |
| `embeddings/` | Vector embedding pipeline |

---

## TODO List (Priority Order)

### Immediate: Streamlit App Completion

- [ ] **Finalize Risk Profiler page**
  - Test end-to-end with all 5 feature dropdowns
  - Verify `load_model()` fast path works in Streamlit
  - Add visualization of risk distribution (bar chart of posteriors)
  - Add confidence indicators for each prediction

- [ ] **Search page integration**
  - Connect hybrid search (BM25 + semantic + RRF) to Streamlit
  - Add taxonomy filter dropdowns
  - Add aircraft/region/date filters
  - Display results with relevance scores + PDF links

- [ ] **Taxonomy Explorer page**
  - Category distribution charts
  - Drill-down from L1 to L2
  - Report list per category

### Next: Trend Analytics & Visualization

- [ ] **KPI SQL views**
  - Category prevalence over time
  - Category co-occurrence matrix
  - Regional heatmaps
  - Seasonal patterns
  - Aircraft type risk comparisons

- [ ] **Trends dashboard page**
  - Time series visualizations
  - Heatmaps (region x category, season x category)
  - Co-occurrence matrices
  - Weather/time cross-tabulations with accident categories

### Future: API & Advanced Features

- [ ] **FastAPI backend** (if needed beyond Streamlit)
  - Semantic search endpoint
  - Bayesian risk scoring endpoint
  - KPI query endpoints

- [ ] **Model improvements**
  - Experiment with feature interactions (e.g., weather x time_of_day)
  - Try hierarchical Bayes with L2 subcategories
  - Improve weather extraction coverage (currently 72.4%)
  - Consider adding aircraft_make as a feature

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
| Risk Thresholds | Data-driven (percentile-based) | 90th percentile = HIGH, 50th = MODERATE |
| Model Persistence | SQLite tables (bayes_priors, bayes_likelihoods) | Fast Streamlit loading via `load_model()` |

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

# Check data coverage
python -c "import sqlite3; c=sqlite3.connect('sqlite/riskradar.db'); print(c.execute('SELECT weather_category, COUNT(*) FROM report_features GROUP BY weather_category').fetchall())"
python -c "import sqlite3; c=sqlite3.connect('sqlite/riskradar.db'); print(c.execute('SELECT time_of_day, COUNT(*) FROM report_features GROUP BY time_of_day').fetchall())"
```

---

## Notes for Next Session

1. Streamlit app is the main focus — Risk Profiler page is functional but needs polish
2. Search page needs to be connected to the `search/` module (BM25 + hybrid)
3. Trend analytics and visualization are the big remaining analytical features
4. All extraction pipelines are complete — no more feature engineering needed unless improving coverage
5. Schema is at version 7 — includes bayes_priors + bayes_likelihoods tables

**Key insight:** The Bayesian model with 5 features (Hit@5=90.7%) shows strong performance. Weather and time features added +3.5pp to Hit@1 over the 3-feature baseline. The model is ready for portfolio demonstration.
