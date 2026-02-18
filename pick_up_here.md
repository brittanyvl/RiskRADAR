# RiskRADAR - Pick Up Here

**Last Updated:** 2026-02-17
**Last Session:** Phase 7 (analytics) + Phase 8 (Streamlit app) completed. 3 narrative reports, Risk Profiler, and Terminology page all functional. Multiple bug fixes applied (duplicate keys, hover_data KeyError, SQLite thread safety, DB path resolution).

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
| Bayesian Risk Model v2 | Complete | Binary Relevance NB, ECE=0.021, Hit@5=86.8% |
| Statistical Audit | Complete | 4 critical flaws found and fixed |
| **Phase 7: Stakeholder Analytics** | **Complete** | **4 query modules: shared, fleet_safety, underwriting, operational_risk** |
| **Phase 8: Streamlit App** | **Complete** | **5 pages: 3 reports + Risk Profiler + Terminology** |

### Streamlit App Pages (All Functional)

| Page | Module | Status | Notes |
|------|--------|--------|-------|
| Fleet Safety Report | `app/views/fleet_safety.py` | Working | Aircraft type comparison, manufacturer profiles, SCF breakdown, co-occurrence, trends |
| Underwriting Risk Report | `app/views/underwriting.py` | Working | Co-occurrence matrix, region/season, VMC/IMC, time-of-day, Bayesian profiles |
| Operational Risk Report | `app/views/operational_risk.py` | Working | LOC-I/CFIT deep dives, weather x time, seasonal patterns, aircraft signatures |
| Risk Profiler | `app/pages/4_Risk_Profiler.py` | Working | 5-feature Bayesian model with calibrated probabilities |
| Terminology | `app/pages/5_Terminology.py` | Working | Searchable glossary (L1, L2, aviation terms, statistical terms) |

### Analytics Query Modules (All Functional)

| Module | File | Key Functions |
|--------|------|---------------|
| Shared | `analytics/queries/shared.py` | `category_counts()`, `cooccurrence_matrix()`, `category_by_feature()`, `dataset_summary()` |
| Fleet Safety | `analytics/queries/fleet_safety.py` | `risk_by_aircraft_category()`, `risk_by_manufacturer()`, `scf_pp_breakdown()`, `failure_trends_by_decade()` |
| Underwriting | `analytics/queries/underwriting.py` | `region_season_matrix()`, `vmc_imc_category_distribution()`, `bayesian_profile_comparison()` |
| Operational Risk | `analytics/queries/operational_risk.py` | `loc_i_breakdown()`, `cfit_breakdown()`, `weather_time_matrix()`, `seasonal_patterns()` |
| Glossary | `analytics/queries/glossary_data.py` | `get_l1_glossary()`, `get_l2_glossary()`, `get_aviation_terms()`, `get_statistical_terms()` |

### Shared App Components

| Component | File | Purpose |
|-----------|------|---------|
| Data Loader | `app/components/data_loader.py` | `@st.cache_data` wrappers, `@st.cache_resource` for Bayesian model |
| Charts | `app/components/charts.py` | Plotly builders: horizontal_bar, grouped_bar, heatmap, line_chart, stacked_bar, donut_chart |
| Report Layout | `app/components/report_layout.py` | page_header, kpi_row, section_divider, insight, chart_with_insight, methodology_section, abbr() |
| Theme | `app/components/theme.py` | Brand colors, colorblind-safe palette, TIME_COLORS, TIME_WINDOWS, CSS injection |

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
- **ECE:** 0.021 (near-perfect calibration)

---

## TODO List (Priority Order)

### NEXT: Human UI Review

The very next task is for a **human to review the Streamlit interface** and provide a detailed critique. The app is functional but has not had a thorough visual/UX review.

**Review checklist:**
- [ ] Navigate all 5 pages — do they load without errors?
- [ ] Check all KPI cards — are values meaningful and accurate?
- [ ] Verify chart readability — labels, colors, axes clear?
- [ ] Test interactive elements — selectboxes, comparisons, filters
- [ ] Check narratives — do they make sense to a non-aviation reader?
- [ ] Test Risk Profiler — do all 5 dropdowns work? Are predictions sensible?
- [ ] Check Terminology page — is it searchable and comprehensive?
- [ ] Mobile/responsive — does the layout work on smaller screens?
- [ ] Colorblind accessibility — are color choices distinguishable?
- [ ] Abbreviation tooltips — do they appear on hover for all codes?

### After UI Review: Polish & Enhancement

- [ ] Fix any bugs or layout issues found during UI review
- [ ] Add Semantic Search page (hybrid search via `search/` module)
- [ ] Add Taxonomy Explorer page (L1/L2 drill-down)
- [ ] Polish Risk Profiler visualization (bar chart of posteriors, confidence indicators)
- [ ] Consider adding landing page with project overview

### Future: API & Advanced Features

- [ ] FastAPI backend (if needed beyond Streamlit)
- [ ] Model improvements (feature interactions, hierarchical Bayes)
- [ ] Improve weather extraction coverage (currently 72.4%)

---

## Bugs Fixed in Last Session

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `StreamlitDuplicateElementKey` on fleet_ac_compare | Selectbox widget and chart_with_insight shared same key | Renamed selectbox key to `fleet_ac_compare_sel` |
| `KeyError: "Prevalence"` in fleet_safety.py | `hover_data` dict key was a display label, not a DataFrame column | Removed `hover_data` param; count shown via `show_values=True` |
| `sqlite3.ProgrammingError` thread safety | `@st.cache_resource` caches model object across threads | Added `check_same_thread=False` to `sqlite3.connect()` |
| Bayesian model "unable to open database file" | Relative DB path failed when Streamlit CWD didn't match project root | Pass absolute path from `riskradar.config.DB_PATH` |

## Design Decisions Made in Last Session

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Report style | Consulting-style narrative | Charts embedded in explanatory text, not bare dashboards |
| Persona approach | 3 reports (Fleet, Underwriting, Ops) | Merged manufacturer + maintenance into Fleet Safety; data is ~47% commercial jets |
| Color palette | Colorblind-safe (no red for data) | Orange (AMBER) for IMC weather; avoids red/green confusion |
| Co-occurrence matrix | Lower triangle only | Symmetric matrix; upper half is redundant |
| Season order | Spring → Summer → Fall → Winter | Chronological, user preference |
| Time-of-day colors | Gold/Orange/Light Blue/Navy | Intuitive mapping to sun position (warm→cool) |
| Abbreviations | HTML `<abbr>` tooltips + spell-then-abbreviate | 39 codes with full definitions; narratives spell out first use |
| KPI selection | Data-meaningful metrics only | Removed "weather coverage" (limitation, not KPI); added "High Complexity %" |
| Navigation | Horizontal top menu via streamlit-option-menu | All pages accessible from any page; cleaner than sidebar |

---

## Key Files

| File | Purpose |
|------|---------|
| `sqlite/riskradar.db` | Main SQLite database (schema v8) |
| `app/main.py` | Streamlit entry point + horizontal navigation |
| `app/views/fleet_safety.py` | Fleet Safety Report page |
| `app/views/underwriting.py` | Underwriting Risk Report page |
| `app/views/operational_risk.py` | Operational Risk Report page |
| `app/pages/4_Risk_Profiler.py` | Bayesian risk profiler page |
| `app/pages/5_Terminology.py` | Searchable glossary page |
| `app/components/data_loader.py` | Cached data access wrappers |
| `app/components/charts.py` | Plotly chart builders |
| `app/components/report_layout.py` | Narrative layout components + abbreviation tooltips |
| `app/components/theme.py` | Brand colors + CSS |
| `analytics/queries/shared.py` | Core analytics queries |
| `analytics/queries/fleet_safety.py` | Fleet safety queries |
| `analytics/queries/underwriting.py` | Underwriting queries |
| `analytics/queries/operational_risk.py` | Operational risk queries |
| `analytics/queries/glossary_data.py` | Glossary content queries |
| `risk_profiler/bayesian_model.py` | Binary Relevance NB with persistence + LOO validation |
| `risk_profiler/extract_weather.py` | VMC/IMC extraction from chunk text |
| `risk_profiler/extract_time.py` | Time-of-day extraction from chunk text |
| `search/` | BM25 + semantic + hybrid (RRF) search module |
| `portfolio.md` | Full project narrative and lessons learned |

---

## CLI Quick Reference

```bash
# Activate environment
cd C:\Users\bvlma\CODE\riskRADAR
venv\Scripts\activate

# Run Streamlit app
streamlit run app/main.py

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

# Quick analytics verification
python -c "from analytics.queries.shared import category_counts; print(category_counts())"
```

---

## Notes for Next Session

1. **FIRST TASK: Human UI review** — run the app, navigate all 5 pages, provide detailed critique
2. All 3 narrative reports are functional but haven't had a thorough visual/UX review
3. Risk Profiler page works with binary relevance model — dropdown selections trigger predictions
4. Search module (`search/`) is complete — needs Streamlit page integration
5. Taxonomy Explorer page not yet built — planned for after UI review
6. Schema is at version 8 — includes binary relevance bayes tables (label + positive_count)
7. The Bayesian model is production-ready — audited, rewritten, validated with proper LOO, ECE=0.021
8. Weather coverage is 72.4% — narratives include coverage caveats where weather data is used
9. `check_same_thread=False` was added to bayesian_model.py for Streamlit compatibility
10. Portfolio.md has the full project narrative including Streamlit design lessons
