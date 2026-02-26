# Underwriting Risk Report -- Design Specification

**Persona:** Aviation specialty underwriter evaluating a book of business
**Core question:** "What risk segments should I price differently?"
**Differentiator from Fleet Safety:** Fleet safety asks "what risks should I prioritize for my fleet?" (operational focus). Underwriting asks "where is my portfolio exposed, and what justifies premium adjustments?" (financial/actuarial focus).

---

## KPIs (Row 1: 4 cards, Row 2: 2 cards)

### Row 1

| # | Label | Value Formula | Detail Text | Accent |
|---|-------|---------------|-------------|--------|
| 1 | **Fatal Category Rate** | `COUNT(DISTINCT reports with LOC-I or CFIT) / n_accidents * 100` | "Of accidents involve the two most lethal categories" | CORAL |
| 2 | **Multi-Peril Exposure** | `COUNT(reports with 3+ L1 categories) / n_accidents * 100` | "Of accidents have 3+ contributing factors (compound risk)" | AMBER |
| 3 | **IMC Risk Multiplier** | `AVG(risk_ratio) across top 5 categories where risk_ratio > 1` | "Average IMC overrepresentation for weather-sensitive categories" | STEEL |
| 4 | **Night Ops Share** | `COUNT(night accidents involving LOC-I, CFIT, or UIMC) / COUNT(all night accidents) * 100` | "Of night accidents involve the highest-severity categories" | NAVY |

### Row 2

| # | Label | Value Formula | Detail Text | Accent |
|---|-------|---------------|-------------|--------|
| 5 | **Top Concentration Region** | Region with the most accidents (from region counts) | "X reports -- highest geographic exposure" | TEAL |
| 6 | **Single-Piston Share** | `COUNT(single-piston accidents) / n_accidents * 100` | "Of the portfolio is single-piston aircraft (highest-volume segment)" | STEEL |

**Why these KPIs matter to an underwriter:**
- KPI 1 (Fatal Category Rate): Fatal accidents = total loss claims. LOC-I and CFIT are the known killers. This is the portfolio's worst-case exposure rate.
- KPI 2 (Multi-Peril Exposure): Multi-label complexity is a severity proxy. More factors = more complex event = higher total claim. This tells the underwriter what % of their book is "complex loss" territory.
- KPI 3 (IMC Risk Multiplier): Quantifies the premium loading justified for IFR operations. If IMC multiplier is 3.2x, that's a concrete input for weather-based pricing.
- KPI 4 (Night Ops Share): Quantifies how concentrated high-severity risk is during night operations -- justifies night surcharges.
- KPI 5 (Top Concentration Region): Portfolio concentration risk. If 40% of exposure is in one region, that's a geographic accumulation problem.
- KPI 6 (Single-Piston Share): Shows portfolio dominance of the highest-volume segment. Underwriters need to know if their book is concentrated in one aircraft type.

---

## Section Plan (10 Sections)

---

### Section 1: Loss Severity Spectrum
**Subtitle:** "Which categories produce the most severe and complex losses?"

**Analysis:** Horizontal bar chart ranking all L1 categories by a "severity score" -- the average number of co-occurring categories in reports that contain each category. Categories where reports tend to have 4+ co-labels signal complex, cascading events (likely hull loss + fatalities). This is NOT the same as the simple category frequency chart in fleet safety -- this reframes frequency through a severity/complexity lens.

**Chart type:** `horizontal_bar` -- categories ranked by average co-label count (severity proxy), with report count as hover data.

**SQL query needed:** NEW -- `severity_ranked_categories()`
```sql
SELECT t.category_code, t.category_name,
       COUNT(DISTINCT t.report_id) AS report_count,
       AVG(rc.n_categories) AS avg_complexity,
       SUM(CASE WHEN rc.n_categories >= 4 THEN 1 ELSE 0 END) AS high_complexity_count
FROM report_taxonomy t
JOIN report_types rt ON t.report_id = rt.report_id
JOIN (
    SELECT report_id, COUNT(DISTINCT category_code) AS n_categories
    FROM report_taxonomy
    WHERE level = 'L1'
    GROUP BY report_id
) rc ON t.report_id = rc.report_id
WHERE t.level = 'L1' AND rt.report_type = 'accident'
GROUP BY t.category_code, t.category_name
ORDER BY avg_complexity DESC
```

**Insight text pattern:** "**[Top category]** has the highest average complexity at [X.X] co-occurring categories per report, meaning it typically appears alongside [X] other hazards in the same event. For underwriters, high-complexity categories represent compound loss scenarios where multiple policy sections may be triggered simultaneously."

**Differentiation from fleet safety:** Fleet safety shows raw category counts ("what happens most?"). Underwriting reframes as severity/complexity ("what costs the most when it happens?").

---

### Section 2: Risk Segmentation by Aircraft Type
**Subtitle:** "How does the risk profile shift across fleet segments?"

**Analysis:** Heatmap showing aircraft_category (rows) x top 8 L1 categories (columns), with values as prevalence %. This gives underwriters a single view of which aircraft segments carry which risks, enabling segment-specific pricing.

**Chart type:** `heatmap` with `value_format="pct"`, hover_labels=ABBREVIATIONS

**SQL query needed:** EXISTING -- `category_by_feature("aircraft_category")` from shared.py. Need to normalize to prevalence % (divide by total reports per aircraft type).

**Insight text pattern:** "**[Aircraft type]** shows disproportionate exposure to **[category]** ([X]% prevalence vs [Y]% portfolio average). Policies covering [aircraft type] should be priced to reflect this elevated [category] risk. Conversely, **[aircraft type 2]** has notably lower [category] exposure, supporting a favorable rate for that segment."

**Differentiation from fleet safety:** Fleet safety uses multi-select grouped bar comparison (interactive exploration). Underwriting uses a single heatmap for quick segment-level pricing decisions -- all segments visible at once.

---

### Section 3: Weather Pricing Factors (IMC vs VMC)
**Subtitle:** "What premium adjustments does weather justify?"

**Analysis:** Diverging bar chart showing IMC prevalence (right) vs VMC prevalence (left) for top categories with a combined sample >= 10. This is the "pricing factor" view -- the divergence directly quantifies the weather surcharge for each category.

**Chart type:** `diverging_bar` (reuse the exact same chart type from fleet safety, but with underwriting framing)

**SQL query needed:** EXISTING -- `weather_risk_ratios()` from fleet_safety.py (already available via data_loader)

**Insight text pattern:** "**[Category]** is [X.X]x more prevalent in IMC than VMC, the largest weather-driven risk multiplier. For an underwriter, this translates to a concrete pricing factor: policies covering frequent IMC operations in [category]-prone segments should carry a proportional surcharge. Categories that skew VMC (left side) are primarily piloting-skill risks rather than weather risks."

**Differentiation from fleet safety:** Fleet safety frames weather as "what should I train for in IMC?" Underwriting frames as "what premium loading does IMC justify?" Same data, different lens -- the insight text explicitly connects to pricing.

---

### Section 4: Night Operations Risk Premium
**Subtitle:** "What surcharge should night operations carry?"

**Analysis:** Two-part section:
1. KPI mini-row: Night share of LOC-I, CFIT, UIMC, ICE (4 cards showing % of each category's accidents that occur at night)
2. Grouped bar: Top 6 categories by time-of-day, using TIME_COLORS palette

**Chart type:** `kpi_row` (4 cards) + `grouped_bar` with TIME_COLORS

**SQL query needed:** EXISTING -- `time_of_day_distribution()` from underwriting.py

**Insight text pattern:** "Night operations account for [X]% of **CFIT** and [Y]% of **UIMC** accidents -- significantly above the overall night share of [Z]%. This quantifies the night operations surcharge: categories with disproportionate nighttime occurrence justify a [ratio] premium factor for night-approved policies."

**Differentiation from fleet safety:** Fleet safety doesn't have a time-of-day section. This is underwriting-unique. The framing is explicitly about surcharge justification.

---

### Section 5: Geographic Exposure Concentration
**Subtitle:** "Where is portfolio risk concentrated?"

**Analysis:** Region x Season heatmap (existing) PLUS a new horizontal bar showing top categories per region (or a region x top-5-category heatmap). The underwriter needs to know not just WHERE but WHAT risks concentrate WHERE.

**Chart type:** `heatmap` for region x season (existing), followed by a second `heatmap` for region x top categories

**SQL query needed:**
- EXISTING: `region_season_matrix()` from underwriting.py
- NEW: `category_by_feature("region")` from shared.py, normalized to prevalence. Needs a `region_category_heatmap()` wrapper.

```sql
-- Use shared.category_by_feature("region") then normalize
-- Each cell = report_count / total_accidents_in_region * 100
```

**Insight text pattern:** "The **[Region]** region accounts for [X]% of the accident portfolio, with particularly elevated **[Category]** prevalence ([Y]% vs [Z]% nationally). Underwriters with heavy [Region] exposure should monitor [Category] trends and consider geographic loading factors. **[Season]** is the peak season in this region."

**Differentiation from fleet safety:** Fleet safety has no geographic analysis at all. This is entirely underwriting-unique.

---

### Section 6: Multi-Peril Complexity Distribution
**Subtitle:** "How complex are the losses in this portfolio?"

**Analysis:** Vertical bar chart showing the distribution of categories-per-report (1, 2, 3, 4+), reframed as a loss-complexity distribution. Add a secondary view: for 4+ category reports, show which categories appear most often (these are the "complex loss drivers").

**Chart type:** `vertical_bar` (existing complexity distribution) + `horizontal_bar` (top categories in 4+ reports)

**SQL query needed:**
- EXISTING: `multi_label_complexity()` from underwriting.py
- NEW: `high_complexity_categories()` -- top L1 categories among reports with 4+ labels

```sql
SELECT t.category_code, t.category_name,
       COUNT(DISTINCT t.report_id) AS report_count
FROM report_taxonomy t
JOIN report_types rt ON t.report_id = rt.report_id
JOIN (
    SELECT report_id
    FROM report_taxonomy
    WHERE level = 'L1'
    GROUP BY report_id
    HAVING COUNT(DISTINCT category_code) >= 4
) complex ON t.report_id = complex.report_id
WHERE t.level = 'L1' AND rt.report_type = 'accident'
GROUP BY t.category_code, t.category_name
ORDER BY report_count DESC
```

**Insight text pattern:** "[X]% of accidents involve 4+ categories, representing the most complex and costly loss events. Among these high-complexity accidents, **[Category 1]** and **[Category 2]** appear most frequently -- they are the nucleus around which cascading failures develop. Loss reserves for these multi-peril events should reflect the compound nature of the claims."

**Differentiation from fleet safety:** Fleet safety mentions avg categories per report as a KPI but doesn't analyze complexity as a loss-severity proxy. Underwriting treats complexity as a direct input to loss modeling.

---

### Section 7: Co-occurrence Risk Clustering
**Subtitle:** "Which hazards travel together? Correlated loss exposure"

**Analysis:** Co-occurrence heatmap (lower triangle, top 10 categories), reframed as correlated-loss exposure. When two categories frequently co-occur, a policy covering one hazard is implicitly exposed to the other.

**Chart type:** `heatmap` with `mask_diagonal=True`, `lower_triangle_only=True`

**SQL query needed:** EXISTING -- `cooccurrence_matrix()` from shared.py

**Insight text pattern:** "**[Cat1]** and **[Cat2]** co-occur in [N] reports, the strongest pairing. For underwriters, this means that a policy triggered by a [Cat1] event has a [X]% probability of also involving [Cat2] -- correlated exposure that should be reflected in aggregate risk modeling. The top 3 co-occurrence pairs are: [list]."

**Differentiation from fleet safety:** Fleet safety shows co-occurrence as "cascading failure patterns for safety management." Underwriting reframes as "correlated loss exposure for aggregate risk modeling" -- same chart, fundamentally different business meaning.

---

### Section 8: High-Severity Category Trends by Decade
**Subtitle:** "Are the most expensive risks growing or shrinking?"

**Analysis:** Line chart showing decade-over-decade prevalence trends for the underwriting-critical categories: LOC-I, CFIT, SCF-PP, ICE, FUEL, UIMC. These are chosen because they are the highest-severity and most pricing-relevant categories. Trends inform long-term pricing strategy -- if LOC-I prevalence is declining, rates may soften; if ICE is rising, rates should harden.

**Chart type:** `line_chart` with color by category

**SQL query needed:** EXISTING -- `category_prevalence_by_decade(categories=["LOC-I", "CFIT", "SCF-PP", "ICE", "FUEL", "UIMC"])` from shared.py

**Insight text pattern:** "Over the last [N] decades, **[Category]** prevalence has [increased/decreased] from [X]% to [Y]%, a [+/-Z] percentage point shift. For long-term pricing, [rising categories] suggest rate hardening may be warranted, while [declining categories] support rate stability. Note: earlier decades have fewer reports, so trend confidence improves in recent periods."

**Differentiation from fleet safety:** Fleet safety trends focus on SCF-PP vs SCF-NP (maintenance) and "fleet risk" categories. Underwriting trends focus on the highest-severity, pricing-relevant categories including UIMC (which fleet safety doesn't trend) and explicitly connect trends to rate hardening/softening.

---

### Section 9: Bayesian Risk Profiling
**Subtitle:** "How do operational profiles compare for pricing?"

**Analysis:** Side-by-side table comparing 4 representative underwriting profiles via the Bayesian model. Profiles chosen to represent distinct risk segments that an underwriter would price differently. Add a "risk differential" column showing the max probability gap between the highest and lowest risk profile for each category.

**Chart type:** `st.dataframe` with conditional highlighting (existing pattern)

**SQL query needed:** EXISTING -- `bayesian_profile_comparison(profiles)` from underwriting.py

**Profiles (revised for clearer underwriting contrast):**
1. "GA Day VFR" -- single-piston, Summer, South, VMC, Afternoon (bread-and-butter GA policy)
2. "Corporate IFR Night" -- turboprop, Winter, Northeast, IMC, Night (high-risk corporate)
3. "Helicopter Ops" -- helicopter, Summer, West, VMC, Morning (utility/EMS segment)
4. "Cold-Weather IFR" -- single-piston, Winter, Midwest, IMC, Night (worst-case GA)

**Insight text pattern:** "The largest pricing gap is in **[Category]**: the highest-risk profile ([profile name]) shows a [X]% probability vs [Y]% for the lowest-risk profile -- a [Z]x differential that directly justifies segment-specific pricing. IMC + Night profiles consistently elevate CFIT and UIMC probabilities, quantifying the surcharge for instrument and nighttime operations."

**Differentiation from fleet safety:** Fleet safety has no Bayesian profiling. This is entirely underwriting-unique.

---

### Section 10: Methodology
**Subtitle:** Expandable section with data notes

**Content updates from current:**
- Keep all existing methodology text
- Add: "Severity proxy: Multi-label complexity (number of CICTT categories per report) is used as a proxy for loss severity. While the dataset does not contain direct cost or fatality data, research shows that accidents with more contributing factors tend to be more severe."
- Add: "Risk multipliers: IMC risk multipliers are computed as the ratio of category prevalence in IMC conditions to prevalence in VMC conditions. A multiplier of 2.0x means the category appears in twice the share of IMC accidents vs VMC accidents."
- Add: "Pricing factors: All risk differentials shown in this report are descriptive analytics based on historical accident data. They quantify relative risk between segments but should be combined with exposure data, claims history, and actuarial judgment for actual premium calculations."

---

## New Queries Required

| Query Function | Module | Purpose |
|----------------|--------|---------|
| `severity_ranked_categories()` | underwriting.py | Categories ranked by avg co-label complexity |
| `high_complexity_categories()` | underwriting.py | Top categories among 4+ label reports |
| `region_category_heatmap()` | underwriting.py | Region x category prevalence heatmap |

All other sections use existing queries from shared.py, fleet_safety.py, or underwriting.py.

---

## Existing Queries Reused

| Query | Source | Used In Section |
|-------|--------|-----------------|
| `dataset_summary()` | shared.py | KPIs |
| `weather_risk_ratios()` | fleet_safety.py | KPIs (IMC multiplier), Section 3 |
| `time_of_day_distribution()` | underwriting.py | KPIs, Section 4 |
| `region_season_matrix()` | underwriting.py | Section 5 |
| `category_by_feature("region")` | shared.py | Section 5 |
| `multi_label_complexity()` | underwriting.py | KPIs, Section 6 |
| `cooccurrence_matrix()` | shared.py | Section 7 |
| `category_prevalence_by_decade()` | shared.py | Section 8 |
| `bayesian_profile_comparison()` | underwriting.py | Section 9 |
| `category_counts()` | shared.py | Various |

---

## Differentiation from Fleet Safety (Summary)

| Dimension | Fleet Safety | Underwriting |
|-----------|-------------|--------------|
| **Lens** | "What risks should I prioritize?" | "What risk segments should I price differently?" |
| **KPIs** | LOC-I/CFIT rate, component failures, IMC involvement, avg factors | Fatal category rate, multi-peril exposure, IMC multiplier, night ops share, concentration |
| **Aircraft analysis** | Interactive multi-select grouped bar | Static heatmap (all segments at once for pricing) |
| **Manufacturer** | Manufacturer x category heatmap | NOT included (underwriters segment by aircraft type, not manufacturer) |
| **Component failures** | SCF-PP/SCF-NP L2 breakdowns | NOT included (too operational; underwriters care about severity, not subsystem details) |
| **Human factors** | HF breakdown + HF x category heatmap | NOT included (training focus, not pricing focus) |
| **LOC-I subtypes** | LOC-I subtype x aircraft heatmap | NOT included (too granular for pricing) |
| **Weather** | IMC vs VMC diverging chart (training lens) | IMC vs VMC diverging chart (pricing lens) + weather pricing KPI |
| **Time of day** | NOT included | Night operations surcharge analysis |
| **Geographic** | NOT included | Region x season + region x category heatmaps |
| **Complexity** | Avg categories KPI only | Full complexity distribution + high-complexity drivers |
| **Co-occurrence** | Cascading failure patterns | Correlated loss exposure |
| **Trends** | SCF-PP/NP + key risk categories | High-severity pricing categories including UIMC |
| **Bayesian model** | NOT included | Side-by-side profile pricing comparison |
| **Unique sections** | Manufacturer heatmap, component failures, human factors, LOC-I subtypes | Severity spectrum, night ops, geographic concentration, complexity deep-dive, Bayesian profiles |

---

## Chart Type Summary

| Section | Chart Type | Height |
|---------|-----------|--------|
| 1. Severity Spectrum | `horizontal_bar` | ~450px (dynamic) |
| 2. Aircraft Segmentation | `heatmap` | ~400px (dynamic) |
| 3. Weather Pricing | `diverging_bar` | ~450px (dynamic) |
| 4. Night Operations | `kpi_row` + `grouped_bar` | KPI + ~420px |
| 5. Geographic | `heatmap` + `heatmap` | ~400px + ~400px |
| 6. Complexity | `vertical_bar` + `horizontal_bar` | ~350px + ~350px |
| 7. Co-occurrence | `heatmap` | ~500px |
| 8. Decade Trends | `line_chart` | ~450px |
| 9. Bayesian | `st.dataframe` | ~400px |
| 10. Methodology | Expander | Auto |

---
---

# UX & Visual Design Recommendations

*Authored by UX/Visualization Specialist — complements the Insurance SME section plan above.*

---

## 1. Current State Assessment

### Fleet Safety Report (Gold Standard) Patterns Worth Replicating

| Pattern | Implementation | Why It Works |
|---------|---------------|--------------|
| Two-row KPI layout | `kpi_row()` called twice with accent colors | Creates visual hierarchy; accent colors convey semantic meaning at a glance |
| Chart type alternation | No two consecutive sections use the same chart | Prevents visual fatigue; each section feels distinct |
| `chart_with_insight()` everywhere | Every chart followed by narrative insight | The "so what?" is never left to the reader |
| Expandable references | `st.expander("Category Acronym Reference")` after dense heatmaps | Technical detail available on demand without cluttering |
| `coverage_note()` for partial data | Weather, time-of-day sections | Builds trust; reader knows the data limitations |
| Dynamic insight text | Insight narratives computed from data, not hardcoded | Insights update automatically if data changes |

### Current Underwriting Report Weaknesses

1. **No accent colors on KPIs** -- all four KPI cards look identical, losing the semantic signal (danger vs neutral vs informational).
2. **Co-occurrence heatmap is near-identical to fleet safety** -- same data, same framing, same insight language. Feels like a copy-paste.
3. **Only 3 chart types** (heatmap, grouped_bar, vertical_bar) -- the report is visually monotonous. Fleet safety uses 5+ types.
4. **Bayesian profiles as a styled dataframe** -- the most underwriting-unique data in the entire app is displayed as the least visual element. A heatmap would be far more scannable.
5. **No primary interactive element** -- fleet safety has the multiselect aircraft comparison that makes the page feel alive. Underwriting is entirely static.
6. **Back-to-back grouped bars** -- weather (Section 4) and time-of-day (Section 5) both use `grouped_bar`, creating visual repetition.
7. **Insight text inconsistency** -- some insights use `<b>` HTML tags, others should use `**markdown**`.
8. **Missing chart types** -- `diverging_bar`, `line_chart`, and `stacked_bar` are available in the chart library but unused.

---

## 2. Visual Design Recommendations

### Chart Type Variety -- Final Mapping

The SME section plan defines 10 sections. Here is the recommended chart type sequence, designed so no two consecutive sections use the same primary chart type:

| # | Section | Primary Chart | Secondary Chart | Why This Chart |
|---|---------|--------------|-----------------|----------------|
| 1 | Loss Severity Spectrum | `horizontal_bar` | -- | Ranked list is the canonical format for "which is highest?" |
| 2 | Aircraft Segmentation | `heatmap` | -- | 2D cross-tab (types x categories) is the heatmap's sweet spot |
| 3 | Weather Pricing | `diverging_bar` | `kpi_row` (4 IMC cards above) | Butterfly chart makes VMC/IMC divergence viscerally obvious |
| 4 | Night Operations | `grouped_bar` | `kpi_row` (4 night-share cards above) | 4 time windows x categories is a natural grouped-bar shape |
| 5 | Geographic Exposure | `heatmap` | -- | Region x season is a 2D cross-tab; region x category as a second heatmap below |
| 6 | Complexity Distribution | `vertical_bar` | `horizontal_bar` (drivers) | Vertical for the 4-bucket distribution; horizontal for the "what drives 4+ complexity" drill-down |
| 7 | Co-occurrence Clustering | `heatmap` | -- | Symmetric matrix = heatmap with lower triangle |
| 8 | Decade Trends | `line_chart` | -- | Time series = line chart, period |
| 9 | Bayesian Profiles | `heatmap` (OVERRIDE: replace dataframe) | Profile builder selectboxes | Heatmap of probabilities is far more scannable than a styled table |
| 10 | Methodology | `st.expander` | -- | Standard pattern |

**Chart type tally:** horizontal_bar (x2), heatmap (x4, but with different data/scales), diverging_bar (x1), grouped_bar (x1), vertical_bar (x1), line_chart (x1) = **6 distinct chart types**.

**Consecutive-type check:**
1. horizontal_bar -> 2. heatmap -> 3. diverging_bar -> 4. grouped_bar -> 5. heatmap -> 6. vertical_bar + horizontal_bar -> 7. heatmap -> 8. line_chart -> 9. heatmap

Sections 5 and 7 are both heatmaps, but Section 6 (vertical + horizontal bars) breaks them up. Sections 7 and 9 are both heatmaps, but Section 8 (line chart) breaks them up. The visual rhythm is maintained.

**Critical override from SME plan:** Section 9 (Bayesian Profiles) should use a **heatmap**, not `st.dataframe`. The SME plan lists `st.dataframe` -- I strongly recommend overriding this. Rationale:
- The Bayesian profile comparison is the most underwriting-unique data in the entire app
- A heatmap with HEATMAP_SCALE (diverging blue-white-red) makes probability hotspots immediately visible
- A dataframe requires the reader to scan numbers cell by cell; a heatmap lets them see the pattern instantly
- The heatmap becomes the report's "hero visualization" -- its visual signature

---

### Layout Structure -- Section by Section

#### Section 1: Loss Severity Spectrum
- **Layout:** Full-width
- `horizontal_bar` at full width, ~450px dynamic height
- `chart_with_insight()` below
- Consider adding `show_values=True, value_format="1f"` for the avg complexity score

#### Section 2: Aircraft Risk Segmentation
- **Layout:** Full-width
- `heatmap` at full width, ~400px dynamic height (scale with number of aircraft types)
- `chart_with_insight()` below
- `st.expander("Category Acronym Reference")` for CICTT code lookups

#### Section 3: Weather Pricing Factors
- **Layout:** Full-width, two parts
- Part A: `kpi_row()` with 4 IMC-share KPI cards (CFIT in IMC, ICE in IMC, UIMC in IMC, LOC-I in IMC) -- keep from current report
- Part B: `diverging_bar` at full width, ~450px dynamic height
- `chart_with_insight()` below
- `coverage_note("Weather", ...)` before charts

#### Section 4: Night Operations Risk Premium
- **Layout:** Full-width, two parts
- Part A: `kpi_row()` with 4 night-share mini-KPIs
- Part B: `grouped_bar` at full width, ~420px height
- **Interaction:** `st.selectbox()` above the chart to filter to a single category (default: "Top 6 categories")
- `chart_with_insight()` below
- `coverage_note("Time-of-day", ...)` before charts

#### Section 5: Geographic Exposure Concentration
- **Layout:** Full-width, two sub-sections
- Sub A: Region x Season `heatmap`, ~400px
- Sub B: Region x Top Categories `heatmap`, ~350px (new query needed)
- `chart_with_insight()` after each
- Each sub-section gets its own `st.markdown("#### ...")` sub-header

#### Section 6: Multi-Peril Complexity
- **Layout:** `st.columns([2, 1])` for the first part
- Left column: `vertical_bar` showing 1/2/3/4+ distribution
- Right column: Markdown with key stats (single vs multi %, 4+ share, pricing implication)
- Below the columns: Full-width `horizontal_bar` showing top categories in 4+ reports
- `chart_with_insight()` after the horizontal bar

#### Section 7: Co-occurrence Risk Clustering
- **Layout:** Full-width
- `heatmap` with `lower_triangle_only=True, mask_diagonal=True`
- `chart_with_insight()` below
- `st.expander("Category Acronym Reference")` below

#### Section 8: High-Severity Trends by Decade
- **Layout:** Full-width
- `line_chart` at full width, ~450px
- `chart_with_insight()` with biggest-mover analysis (computed from data)

#### Section 9: Bayesian Risk Profiles (Hero Section)
- **Layout:** Full-width, with interactive profile builder above
- **Profile builder:** `st.columns(5)` with selectboxes for aircraft, season, region, weather, time-of-day
- Below: `heatmap` (categories x profiles) with HEATMAP_SCALE, ~550px
- `chart_with_insight()` below
- `st.expander("Category Acronym Reference")` below
- Coverage note about model calibration (ECE=0.021)

#### Section 10: Methodology
- **Layout:** `methodology_section()` in expander
- Full-width, standard pattern

---

## 3. Color Strategy

### KPI Accent Colors (Final)

| KPI | Accent | Rationale |
|-----|--------|-----------|
| Fatal Category Rate (LOC-I/CFIT) | `CORAL` | Danger -- these are the fatal categories |
| Multi-Peril Exposure (3+) | `AMBER` | Warning -- complexity = elevated cost |
| IMC Risk Multiplier | `STEEL` | Informational -- quantitative metric |
| Night Ops Share | `NAVY` | Night = dark navy, semantic match |
| Top Concentration Region | `TEAL` | Geographic/opportunity signal |
| Single-Piston Share | `STEEL` | Neutral -- portfolio composition fact |

### Chart Color Assignments

| Section | Color Choice | Specific Values |
|---------|-------------|-----------------|
| 1. Severity (horizontal_bar) | Single color | `CORAL` -- severity = danger signal |
| 2. Aircraft Segmentation (heatmap) | Sequential scale | `SEQUENTIAL_SCALE` (single-hue blue) |
| 3. Weather (diverging_bar) | Two opposing colors | `left_color=STEEL` (VMC baseline), `right_color=CORAL` (IMC danger) |
| 3. Weather KPIs | Per-card accents | CFIT=`CORAL`, ICE=`AMBER`, UIMC=`NAVY`, LOC-I=`CORAL` |
| 4. Night (grouped_bar) | TIME_COLORS dict | Morning=`#DDAA33`, Afternoon=`#DD8452`, Evening=`#88CCEE`, Night=`#1B2A4A` |
| 4. Night KPIs | All NAVY accent | Night theme consistency |
| 5. Geographic (heatmap x2) | Sequential scale | `SEQUENTIAL_SCALE` -- intensity = count |
| 6. Complexity (vertical_bar) | Single color | `AMBER` -- warning tone for complexity |
| 6. Complexity drivers (horizontal_bar) | Single color | `CORAL` -- these drive the worst losses |
| 7. Co-occurrence (heatmap) | Sequential scale | `SEQUENTIAL_SCALE` |
| 8. Trends (line_chart) | CHART_PALETTE | Auto-assigned per category series |
| 9. Bayesian (heatmap) | **Diverging scale** | `HEATMAP_SCALE` (blue-white-red) -- makes high-probability cells pop in red |

### Semantic Consistency Rules (Enforce Across All Sections)

- **IMC** is always `CORAL` or `AMBER` when shown as a risk factor (never `TEAL`)
- **VMC** is always `STEEL` (neutral baseline)
- **Night** is always `NAVY` (TIME_COLORS["Night"])
- **LOC-I and CFIT** references in insight text always in `**bold**`, associated with `CORAL` when charted individually
- **TEAL** reserved for positive/opportunity signals or geographic neutrality
- **AMBER** for moderate/warning signals (complexity, icing, moderate risk)
- **NAVY** for informational/depth signals (nighttime, Bayesian model, methodology)

---

## 4. Interaction Design

### Primary Interaction: Bayesian Profile Builder (Section 9)

This is the **signature interaction** of the underwriting report, differentiated from fleet safety's aircraft multiselect.

**Implementation:**

```python
st.markdown("#### Build a Custom Risk Profile")
st.markdown("Select operational characteristics to generate a risk estimate, "
            "compared against three benchmark profiles.")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    custom_aircraft = st.selectbox("Aircraft Type",
        options=["single-piston", "multi-piston", "turboprop", "turbojet",
                 "helicopter", "glider"],
        index=0, key="uw_aircraft")
with c2:
    custom_season = st.selectbox("Season",
        options=["Spring", "Summer", "Fall", "Winter"],
        index=1, key="uw_season")
with c3:
    custom_region = st.selectbox("Region",
        options=["Northeast", "South", "Midwest", "West"],
        index=1, key="uw_region")
with c4:
    custom_weather = st.selectbox("Weather",
        options=["VMC", "IMC"],
        index=0, key="uw_weather")
with c5:
    custom_time = st.selectbox("Time of Day",
        options=["Morning", "Afternoon", "Evening", "Night"],
        index=1, key="uw_time")
```

The custom profile joins the 3 hardcoded benchmark profiles as a 4th column in the heatmap. The heatmap updates on every selectbox change (Streamlit's default rerun behavior).

**Benchmark profiles (3 fixed):**
1. "GA Day VFR" -- single-piston, Summer, South, VMC, Afternoon
2. "Corporate IFR Night" -- turboprop, Winter, Northeast, IMC, Night
3. "Helicopter Ops" -- helicopter, Summer, West, VMC, Morning

**Why this is the right primary interaction:**
- Fleet safety's multiselect lets users compare aircraft types (one dimension). The profile builder lets users compare across **five dimensions simultaneously** -- a fundamentally different interaction.
- This matches the underwriter's actual workflow: "What does the risk look like for THIS specific policy application?"
- The 3 fixed benchmarks provide context ("is my custom profile worse than the typical corporate IFR policy?")

### Secondary Interaction: Time-of-Day Category Filter (Section 4)

```python
cat_options = ["Top 6 (Overview)"] + top_10_categories
selected_cat = st.selectbox(
    "Focus on a specific category",
    options=cat_options,
    index=0,
    key="uw_time_cat"
)
```

- Default "Top 6" shows the existing grouped bar with all categories
- Selecting a specific category shows a focused 4-bar chart (Morning/Afternoon/Evening/Night) for that one category with percentage labels
- Keeps the section compact while allowing drill-down

### No Other Interactions
Two interactions total. Do not add filters to every section -- it creates decision paralysis and slows down the page. The rest of the report should be scannable without any clicks.

---

## 5. Insight Text Guidelines

### Voice and Audience
- **Audience:** Non-technical insurance professional who reads 10+ reports/week
- **Register:** Confident, direct, data-driven. No hedging ("may suggest"), no chart-reading instructions ("as you can see from the chart")
- **Vocabulary:** pricing, premium loading, surcharge, portfolio concentration, aggregate exposure, loss severity, rate hardening/softening, risk multiplier, correlated loss

### Structure for Every `chart_with_insight()`

1. **Lead sentence:** State the single most important finding in bold terms
2. **Supporting data:** 1-2 sentences with specific numbers from the chart
3. **Pricing implication:** What this means for the underwriter's decision-making

### Example Insights (Templates)

**Section 3 (Weather Diverging Bar):**
> **CFIT risk more than doubles in IMC** -- 62% prevalence in instrument conditions versus 24% in VMC. For IFR-approved operations, this translates to a measurable weather surcharge factor. Categories on the left (VMC-skewed) are primarily piloting-skill risks that weather pricing alone cannot address.

**Section 8 (Decade Trends):**
> **LOC-I prevalence has declined from 48% in the 1970s to 31% in the 2020s**, a 17 percentage-point improvement likely reflecting advances in training and automation. However, **ICE has risen from 5% to 12%** over the same period. Long-term pricing should account for these diverging trajectories -- LOC-I rates may soften while icing-related rates should harden.

**Section 9 (Bayesian Profiles):**
> The largest pricing gap is in **CFIT**: the Corporate IFR Night profile shows a 58% probability versus 12% for GA Day VFR -- a 4.8x differential. This quantifies the surcharge for instrument and nighttime operations. Your custom profile (**[dynamic label]**) shows [X]% CFIT probability, placing it [above/below] the corporate benchmark.

### Anti-Patterns to Avoid
- "This chart shows..." -- the chart already shows it
- "As we can see..." -- patronizing
- "It's worth noting that..." -- filler
- "may" / "could suggest" / "appears to" -- be direct
- `<b>HTML bold</b>` -- use `**markdown bold**` for consistency with fleet safety
- Raw numbers without context: "48%" should be "48% of accidents" or "48% prevalence"

---

## 6. Differentiation from Fleet Safety

### Visual Identity

| Dimension | Fleet Safety | Underwriting |
|-----------|-------------|--------------|
| **Hero chart** | Grouped bar (aircraft comparison) | Heatmap (Bayesian profiles, diverging blue-white-red) |
| **Primary interaction** | Multiselect (aircraft types) | Profile builder (5 selectboxes across dimensions) |
| **Color signature** | STEEL/NAVY dominated (neutral/informational) | CORAL/HEATMAP_SCALE dominated (risk/severity emphasis) |
| **Opening section** | Category frequency by aircraft type | Loss severity spectrum (complexity-ranked) |
| **Closing analysis** | Human factors + LOC-I subtypes | Bayesian risk profiles + decade trends |

### Sections That Must NOT Overlap

The following fleet safety sections must not appear in underwriting (they are operational, not financial):
- Manufacturer Risk Profiles (underwriters segment by aircraft type, not manufacturer)
- Component Failure Breakdown (SCF-PP/SCF-NP drilldown is a maintenance concern)
- Human Factors (training focus, not pricing focus)
- LOC-I Subtypes (too granular for portfolio-level pricing)

### Sections That Share Data But Differ in Framing

| Shared Data | Fleet Safety Framing | Underwriting Framing |
|------------|---------------------|---------------------|
| Co-occurrence matrix | "Cascading failure patterns for safety management" | "Correlated loss exposure for aggregate risk modeling" |
| Weather risk ratios | "What should I train for in IMC?" | "What premium loading does IMC justify?" |
| Decade trends | "Fleet risk trajectory" | "Rate hardening/softening signals" |

### Sections Unique to Underwriting (No Fleet Safety Equivalent)
1. Loss Severity Spectrum (complexity-ranked categories)
2. Night Operations Risk Premium (time-of-day analysis)
3. Geographic Exposure Concentration (region x season + region x category)
4. Multi-Peril Complexity deep-dive (4+ category drivers)
5. Bayesian Risk Profiles (interactive profile builder)

---

## 7. Plotly Configuration Notes

### Heatmap for Bayesian Profiles (Section 9)

Use `HEATMAP_SCALE` (diverging) instead of `SEQUENTIAL_SCALE`:
```python
fig = heatmap(
    profile_matrix,
    title="Predicted Risk by Operational Profile",
    height=max(500, len(categories_shown) * 35 + 80),
    value_format="pct",
    colorscale=HEATMAP_SCALE,  # Diverging: blue-white-red
    colorbar_title="Probability",
)
```

The diverging scale is critical because probabilities range from near-zero (cool blue) to 50%+ (hot red), and the "midpoint" (white) around 20-25% helps the reader distinguish elevated from baseline risk.

### Diverging Bar for Weather (Section 3)

Reuse the exact `diverging_bar()` function from fleet safety. No custom Plotly config needed -- the function handles negative-value mirroring, text labels, and hover templates.

```python
fig = diverging_bar(
    wrr_filtered,
    y="category_label",
    left_col="vmc_prevalence",
    right_col="imc_prevalence",
    left_label="VMC (clear weather)",
    right_label="IMC (poor weather)",
    left_color=STEEL,
    right_color=CORAL,
    title="Category Prevalence: VMC vs IMC (Pricing Factors)",
    height=max(450, len(wrr_filtered) * 30 + 80),
)
```

### Dynamic Heights

All heatmaps and horizontal bars should use dynamic height based on row count:
```python
height=max(MIN_HEIGHT, num_rows * PIXELS_PER_ROW + PADDING)
```

Recommended values:
- Heatmaps: `max(350, n_rows * 38 + 80)`
- Horizontal bars: `max(300, n_rows * 36 + 60)`
- Grouped bars: fixed 420px (category count is bounded)
- Line charts: fixed 450px

### Annotation Threshold for Heatmaps

- Co-occurrence (Section 7): `annotation_threshold=10` -- suppress tiny values
- Region x Season (Section 5): `annotation_threshold=5`
- Aircraft segmentation (Section 2): annotate all cells (no threshold)
- Bayesian profiles (Section 9): annotate all cells (probability values are always meaningful)

---

## 8. Page Load Performance Considerations

- **Cache everything:** All data functions are already `@st.cache_data(ttl=3600)`. The Bayesian model is `@st.cache_resource`. No changes needed.
- **Bayesian profile builder:** When the user changes a selectbox, Streamlit reruns the entire page. The 3 benchmark profiles are constant, so their predictions should be computed once and cached. Only the custom profile needs recomputation on change. Consider structuring the code so the 3 fixed profiles use a separate cached call.
- **Limit heatmap size:** Cap aircraft type heatmap at top 6-8 types by volume. Cap category axes at top 8-10 categories. Large heatmaps (15x15+) become unreadable at typical screen widths.
- **Lazy sections:** Consider wrapping Sections 7-9 (co-occurrence, trends, Bayesian) in `st.expander` if page load becomes slow. But prefer full display -- expanders hide insights that should be front-and-center.
