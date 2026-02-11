# RiskRADAR - Pick Up Here

**Last Updated:** 2026-01-27
**Last Session:** Aircraft extraction improvements completed

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
| Phase 6A: L1 Taxonomy | Complete | 27 CICTT categories, 453 reports classified |
| Phase 6A-Sub: L2 Taxonomy | Complete | 32 subcategories, 1,106 assignments |
| Qdrant Enrichment | Complete | Payloads enriched with taxonomy + PDF URLs |
| Aircraft Extraction | Complete | 92.7% coverage (473/510 reports) |

### Data Quality Summary

| Feature | Count | Coverage |
|---------|-------|----------|
| Aircraft Category | 473 | 92.7% |
| Aircraft Make | 473 | 92.7% |
| Region | 461 | 90.4% |
| Season | 489 | 95.9% |
| L1 Taxonomy | 453 | 88.8% |
| L2 Taxonomy | ~400 | ~78% |

### Key Files

| File | Purpose |
|------|---------|
| `sqlite/riskradar.db` | Main SQLite database |
| `risk_profiler/aircraft_data.py` | Aircraft lookup with 686 patterns |
| `risk_profiler/extract_features.py` | Feature extraction pipeline |
| `taxonomy/` | CICTT classification system |
| `embeddings/` | Vector embedding pipeline |
| `docs/feature_extraction_learnings.md` | Validation learnings |

---

## TODO List (Priority Order)

### Phase 3.5: Data Quality & Enrichment (NEXT)

- [ ] **Manual review of reports WITHOUT taxonomy (57 reports)**
  - Export list of unclassified reports
  - Review titles to determine if they are:
    - Safety Studies (label as `report_type: safety_study`)
    - Multi-Accident Analysis (label as `report_type: multi_accident`)
    - Hazmat Reports (label as `report_type: hazmat`)
    - Policy Recommendations (label as `report_type: recommendation`)
    - Actual accidents missing classification
  - Create `report_types` table to track document categories
  - Goal: Clear picture of what we have for retrieval

- [ ] **Improve taxonomy hit rate**
  - Analyze why 57 reports have no taxonomy
  - Consider lowering similarity threshold for edge cases
  - Review seed phrases for underrepresented categories
  - Add manual classifications where needed
  - Target: 95%+ taxonomy coverage for actual accident reports

- [ ] **Label non-accident documents**
  - ASR series (Aviation Safety Recommendations)
  - AIR series (Safety studies)
  - HZB/HZMSR (Hazmat reports)
  - Generic AAR titles (need body parsing)

### Phase 4: Bayesian Risk Model

- [ ] **Define Bayesian model structure**
  - P(category | aircraft_type, region, season)
  - Prior probabilities from historical data
  - Likelihood functions for each feature
  - Posterior calculation for risk scoring

- [ ] **Build model tables**
  ```sql
  -- Prior probabilities
  CREATE TABLE bayes_priors (
      category_code TEXT PRIMARY KEY,
      prior_probability REAL,
      sample_count INTEGER
  );

  -- Conditional probabilities
  CREATE TABLE bayes_likelihoods (
      category_code TEXT,
      feature_name TEXT,  -- 'aircraft_category', 'region', 'season', 'decade'
      feature_value TEXT,
      likelihood REAL,
      sample_count INTEGER
  );

  -- Risk scores by combination
  CREATE TABLE risk_scores (
      aircraft_category TEXT,
      region TEXT,
      category_code TEXT,
      risk_score REAL,
      confidence_interval_low REAL,
      confidence_interval_high REAL
  );
  ```

- [ ] **Train model on historical data**
- [ ] **Validate model predictions**
- [ ] **Document model assumptions and limitations**

### Phase 5: KPI Database Design

Define SQL tables and views for dashboard KPIs BEFORE building UI.

#### 5.1 Insurance Dashboard KPIs

Target users: Insurance underwriters assessing risk

| KPI | Description | SQL View |
|-----|-------------|----------|
| Risk Score by Aircraft Type | Bayesian risk per aircraft category | `kpi_risk_by_aircraft` |
| Risk Score by Region | Geographic risk distribution | `kpi_risk_by_region` |
| Risk Trend Over Time | Year-over-year risk changes | `kpi_risk_trend` |
| High-Risk Combinations | Aircraft + Region + Season combos | `kpi_high_risk_combos` |
| Claim Severity Indicators | Fatality rates by category | `kpi_severity_indicators` |
| Portfolio Risk Assessment | Aggregate risk for aircraft fleet | `kpi_portfolio_risk` |

#### 5.2 Manufacturer Dashboard KPIs

Target users: Aircraft/component manufacturers

| KPI | Description | SQL View |
|-----|-------------|----------|
| SCF-PP by Manufacturer | System failures - powerplant | `kpi_scf_pp_by_make` |
| SCF-NP by Manufacturer | System failures - non-powerplant | `kpi_scf_np_by_make` |
| Model-Specific Issues | Recurring problems by model | `kpi_model_issues` |
| Component Failure Trends | Failure modes over time | `kpi_component_trends` |
| Design-Related Factors | Categories manufacturers can address | `kpi_design_factors` |
| Comparison to Fleet Average | Manufacturer vs industry baseline | `kpi_manufacturer_comparison` |

#### 5.3 Maintenance Dashboard KPIs

Target users: MRO providers, airline maintenance teams

| KPI | Description | SQL View |
|-----|-------------|----------|
| Maintenance-Related Accidents | MAINT category analysis | `kpi_maint_accidents` |
| Fuel System Issues | FUEL category breakdown | `kpi_fuel_issues` |
| Icing-Related Events | ICE category by aircraft type | `kpi_icing_events` |
| Pre-Flight Detection Rate | Issues detectable before flight | `kpi_preflight_detection` |
| Age-Related Failures | Correlation with aircraft age | `kpi_age_failures` |
| Inspection Effectiveness | Post-inspection incident rates | `kpi_inspection_effectiveness` |

#### 5.4 Human Factors Dashboard KPIs

Target users: Flight crews, training departments, safety officers

| KPI | Description | SQL View |
|-----|-------------|----------|
| LOC-I Analysis | Loss of control breakdown | `kpi_loci_analysis` |
| CFIT Analysis | Controlled flight into terrain | `kpi_cfit_analysis` |
| HFACS Categories | Human factors taxonomy | `kpi_hfacs_breakdown` |
| CRM-Related Events | Crew resource management | `kpi_crm_events` |
| Fatigue Indicators | Time-of-day correlations | `kpi_fatigue_indicators` |
| Training Gap Indicators | Skill-based vs decision errors | `kpi_training_gaps` |
| Phase of Flight Analysis | Takeoff, cruise, landing risks | `kpi_flight_phase` |

#### 5.5 Trends & Analytics Dashboard KPIs

Target users: Safety researchers, analysts

| KPI | Description | SQL View |
|-----|-------------|----------|
| Category Prevalence Over Time | Trend lines by decade/year | `kpi_category_trends` |
| Category Co-occurrence Matrix | Heatmap of related categories | `kpi_category_cooccurrence` |
| Regional Heatmaps | Geographic distribution | `kpi_regional_heatmap` |
| Seasonal Patterns | Month/quarter analysis | `kpi_seasonal_patterns` |
| Aircraft Evolution | Risk changes as aircraft age | `kpi_aircraft_evolution` |
| Linear Regression: Risk Factors | Statistical correlations | `kpi_regression_analysis` |
| Year-over-Year Comparisons | Delta analysis | `kpi_yoy_comparison` |
| Aggregations by Metadata | Pivot tables by any dimension | `kpi_metadata_aggregations` |

### Phase 6: API Development

Build FastAPI backend BEFORE Streamlit.

- [ ] **Semantic Search API**
  ```
  POST /api/search
  - query: string
  - filters: {categories, aircraft_type, region, date_range}
  - top_k: int
  - model: "minilm" | "mika"
  ```

- [ ] **Bayesian Risk API**
  ```
  POST /api/risk/score
  - aircraft_category: string
  - region: string
  - season: string

  GET /api/risk/profile/{aircraft_type}
  GET /api/risk/trends/{category_code}
  ```

- [ ] **KPI API Endpoints**
  ```
  GET /api/kpi/insurance/{kpi_name}
  GET /api/kpi/manufacturer/{kpi_name}
  GET /api/kpi/maintenance/{kpi_name}
  GET /api/kpi/human-factors/{kpi_name}
  GET /api/kpi/trends/{kpi_name}
  ```

- [ ] **Report API**
  ```
  GET /api/reports/{report_id}
  GET /api/reports/{report_id}/chunks
  GET /api/reports/{report_id}/taxonomy
  ```

### Phase 7: Streamlit Application (LAST)

Only begin after Phases 4-6 are complete.

#### Page 1: Semantic Search
- Natural language query
- Taxonomy filters
- Aircraft/region/date filters
- Results with relevance scores
- PDF links

#### Page 2: Insurance Dashboard
- Risk assessment tools
- Portfolio analysis
- Trend visualizations
- Export capabilities

#### Page 3: Manufacturer Dashboard
- System failure analysis
- Model-specific insights
- Design factor breakdown
- Comparison tools

#### Page 4: Maintenance Dashboard
- Maintenance-related events
- Component analysis
- Age/inspection correlations
- Actionable insights

#### Page 5: Human Factors Dashboard
- LOC-I/CFIT deep dives
- HFACS analysis
- CRM and training gaps
- Phase of flight analysis

#### Page 6: Trends & Analytics
- Time series visualizations
- Heatmaps
- Co-occurrence matrices
- Regression analysis
- Custom aggregations

---

## Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Taxonomy | CICTT L1 + Industry L2 | Industry standard, interpretable |
| Embedding Model | MiniLM (production) | Better semantic precision (75.5% vs 60%) |
| Search | Vector + BM25 hybrid | Best retrieval performance |
| Case Sensitivity | UPPER() in SQL | Robust pattern matching |
| Fuzzy Matching | Trigram Jaccard @ 0.85 | Catches typos and variants |

---

## Database Schema Additions Needed

```sql
-- Report type classification (safety study vs accident)
CREATE TABLE report_types (
    report_id TEXT PRIMARY KEY,
    report_type TEXT,  -- 'accident', 'safety_study', 'recommendation', 'hazmat', 'multi_accident'
    classification_source TEXT,  -- 'auto', 'manual'
    notes TEXT,
    FOREIGN KEY (report_id) REFERENCES reports(filename)
);

-- KPI materialized views (create these before dashboards)
-- See Phase 5 for full list

-- API rate limiting / usage tracking
CREATE TABLE api_usage (
    id INTEGER PRIMARY KEY,
    endpoint TEXT,
    timestamp TEXT,
    response_time_ms INTEGER,
    status_code INTEGER
);
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `risk_profiler/bayesian_model.py` | Bayesian risk calculation |
| `risk_profiler/kpi_views.sql` | SQL views for all KPIs |
| `api/main.py` | FastAPI application |
| `api/routers/search.py` | Search endpoints |
| `api/routers/risk.py` | Risk score endpoints |
| `api/routers/kpi.py` | KPI endpoints |
| `app/pages/` | Streamlit pages (after API) |

---

## Quick Start Commands

```bash
# Activate environment
cd C:\Users\bvlma\CODE\riskRADAR
venv\Scripts\activate

# Check current coverage
python -c "import sqlite3; conn = sqlite3.connect('sqlite/riskradar.db'); print(conn.execute('SELECT COUNT(*) FROM report_features WHERE aircraft_category IS NOT NULL').fetchone()[0], 'reports with aircraft')"

# Run feature extraction
python -m risk_profiler.extract_features

# Check taxonomy coverage
python -m taxonomy.cli stats
```

---

## Notes for Next Session

1. Start with **Phase 3.5** - manual review of unclassified reports
2. Create `report_types` table to properly categorize documents
3. Design Bayesian model tables before implementation
4. Define ALL KPI SQL views before ANY Streamlit work
5. Build FastAPI backend as the data access layer
6. Streamlit is the LAST step, not the first

**Remember:** Backend first, frontend last. Data quality drives everything.
