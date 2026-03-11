# RiskRADAR - Pick Up Here

**Last Updated:** 2026-03-10

## Completed

- [x] Fleet Safety Report — full overhaul (KPIs, multi-select aircraft comparison, manufacturer heatmap, human factors, LOC-I subtypes, IMC/VMC diverging chart, decade trends)
- [x] Underwriting Risk Report — full overhaul (6 KPIs, 10 sections, quadrant bubble scatter severity spectrum, interactive Bayesian profile builder, co-occurrence heatmap, decade trends, storytelling-driven section ordering)
- [x] Operational Risk Report — full overhaul (6 KPIs, 11 sections + methodology, heatmaps, human factors, decade trends, aircraft risk signatures)
- [x] Semantic Search page — built with clean architecture (4 new files: result_types, enrichment, service, search_filters). Horizontal filter bar (mode/category/aircraft/dates), 3 search modes (hybrid/semantic/keyword), paginated results with highlighted snippets, XSS-safe rendering. Code-reviewed with 8 fixes applied.
- [x] Qdrant Cloud re-provisioned — free-tier cluster expired due to inactivity; new cluster created, both collections re-uploaded and enriched (24,766 points each for MiniLM + MIKA)

## Roadmap (in order)

### 1. Search page — QA + result consolidation
- [ ] Manual QA of all 3 search modes (hybrid, semantic, keyword) with various queries
- [ ] **Consolidate multi-chunk results per report** — currently returns one result per matching chunk, so the same report can appear multiple times. Need to group by report_id and show the best-matching snippet per report.
- [ ] Verify filters (L1, L2, aircraft type, date range) work correctly across modes
- [ ] Test pagination, edge cases (empty results, single result, etc.)

### 2. Taxonomy Explorer page
- [ ] L1 category overview with counts and descriptions
- [ ] L1 → L2 drill-down with report lists
- [ ] Visual category distribution (treemap or sunburst)

### 3. Risk Profiler — review & finalize
- [ ] Review current page (`app/pages/4_Risk_Profiler.py`)
- [ ] Polish visualization (bar chart of posteriors, confidence indicators)
- [ ] Validate UX flow and output clarity

### 4. About page — review & finalize
- [ ] Review and fix current About/landing page
- [ ] Ensure project description, methodology, and data sourcing are clear
- [ ] Final layout and copy polish

## Important: Qdrant Cloud Maintenance

Free-tier Qdrant clusters are deleted after extended inactivity. If search returns 404 errors:
1. Create a new free cluster at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Update `QDRANT_URL` and `QDRANT_API_KEY` in `.env` and `.streamlit/secrets.toml`
3. Re-upload: `python -m embeddings.cli upload both`
4. Re-enrich: `python -m embeddings.cli enrich both --l1-run 1 --l2-run 1`
5. Restart Streamlit (cached client holds old connection)

All embedding vectors are stored locally in `embeddings_data/v2/` — no data is lost.

## Future (post-deployment)
- [ ] FastAPI backend (if needed beyond Streamlit)
- [ ] Model improvements (feature interactions, hierarchical Bayes)
- [ ] Improve weather extraction coverage (currently 72.4%)
