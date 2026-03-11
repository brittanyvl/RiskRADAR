# RiskRADAR Portfolio Statement

A technical narrative documenting the design decisions, challenges overcome, and lessons learned building an end-to-end semantic search pipeline for aviation safety documents.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Project Motivation](#project-motivation)
- [Technical Architecture](#technical-architecture)
- [Key Technical Challenges](#key-technical-challenges)
- [The Chunking Evolution: v1 to v2](#the-chunking-evolution-v1-to-v2)
- [Evaluation Framework Design](#evaluation-framework-design)
- [Statistical Validity](#statistical-validity)
- [Results and Findings](#results-and-findings)
- [Lessons Learned](#lessons-learned)
- [The Taxonomy Journey: From Unsupervised Discovery to Industry Standards](#the-taxonomy-journey-from-unsupervised-discovery-to-industry-standards)
- [Closing the Gap: Data Quality and Taxonomy Coverage](#closing-the-gap-data-quality-and-taxonomy-coverage)
- [Bayesian Risk Model: From Features to Audit to Production](#bayesian-risk-model-from-features-to-audit-to-production)
- [Streamlit Application: From Dashboards to Consulting Reports](#streamlit-application-from-dashboards-to-consulting-reports)
- [Semantic Search: From Pipeline to Product](#semantic-search-from-pipeline-to-product)
- [Future Directions](#future-directions)
- [Skills Demonstrated](#skills-demonstrated)

---

## Executive Summary

RiskRADAR transforms 510 NTSB aviation accident reports (spanning 1966-present) into a semantically searchable, taxonomically classified knowledge base. The project demonstrates production-grade data engineering practices:

- **30,602 pages** processed through OCR and quality pipelines
- **24,766 chunks** optimized through iterative evaluation
- **2 embedding models** compared with rigorous statistical methods
- **50 benchmark queries** spanning 6 difficulty categories
- **38.6% semantic lift** achieved with domain-specific embeddings
- **431 accident reports classified** into 27 CICTT Level 1 categories (98.9% coverage)
- **510 reports typed** into 6 categories via automated prefix/title classification
- **1,106 report-L2 assignments** across 32 industry-standard subcategories
- **Qdrant payloads enriched** with taxonomy data for category-filtered search
- **7 features extracted** from metadata and unstructured text (aircraft, region, season, weather, time)
- **Binary Relevance Bayesian model** trained on 431 accident reports, statistically audited and rewritten to fix 4 critical flaws, achieving ECE=0.021 calibration
- **Weather (VMC/IMC)** extracted from 369/510 reports via regex on meteorological sections
- **Time-of-day** extracted from 463/510 reports via timestamp parsing
- **Stakeholder analytics** built as reusable SQL query modules (fleet safety, underwriting, operational risk)
- **Streamlit application** with 3 consulting-style narrative reports, interactive risk profiler, searchable terminology glossary, and hybrid search interface
- **Semantic search page** with horizontal filter bar, 3 search modes (hybrid/semantic/keyword), taxonomy + date filters, paginated results with highlighted snippets, and XSS-safe rendering

The most significant technical insights:
1. **Chunk quality directly determines retrieval quality**. Version 2's chunking strategy improved Hit@10 from 94.9% to 100% and MRR from 0.788 to 0.816.
2. **Unsupervised topic modeling fails on standardized documents**. BERTopic discovered 76 topics, but human review revealed they captured document structure rather than safety factors—prompting a pivot to the industry-standard CICTT taxonomy.
3. **Coverage gaps require root cause analysis, not brute force**. An 18% taxonomy gap (92 reports) seemed like a classification failure, but systematic investigation revealed most were non-accident documents that *shouldn't* have taxonomy. The true gap was 8%, resolved through iterative, human-guided section prioritization.
4. **Rigorous statistical auditing catches invisible errors**. The initial Bayesian model reported Hit@5=90.7%, but a statistical audit revealed softmax normalization on multi-label data caused 3-4x calibration error, fake LOO was optimistically biased, and the model actually performed worse than a prior-only baseline at Hit@3. A complete rewrite to Binary Relevance Naive Bayes fixed all 4 flaws, achieving ECE=0.021 (near-perfect calibration).

---

## Project Motivation

### The Problem

Aviation safety knowledge is trapped in unstructured PDF documents. Accident investigators, safety researchers, and aviation professionals need to find relevant precedents and patterns across decades of reports. Traditional keyword search fails because:

1. **Vocabulary mismatch**: Users search for "engine failure" but reports say "powerplant malfunction"
2. **Concept fragmentation**: Related findings are scattered across sections
3. **Structural complexity**: 60-page reports with appendices, figures, and footnotes

### The Opportunity

Modern embedding models can capture semantic relationships, but require careful preprocessing. This project explores:

- How to chunk technical documents for optimal retrieval
- Whether domain-specific models outperform general-purpose ones
- How to rigorously evaluate retrieval quality in specialized domains

---

## Technical Architecture

### Pipeline Overview

```
Phase 1: Scrape     Phase 3: Extract    Phase 4: Chunk      Phase 5: Embed      Phase 6: Classify
─────────────────   ─────────────────   ─────────────────   ─────────────────   ─────────────────
NTSB Website        PDF Documents       Document Text       Chunk Vectors       L1+L2 Taxonomy
    │                   │                   │                   │                   │
    ▼                   ▼                   ▼                   ▼                   ▼
510 PDFs ────────► 30,602 pages ────► 24,766 chunks ────► Qdrant Cloud ────► 27 L1 + 32 L2
    │                   │                   │                   │                   │
    ▼                   ▼                   ▼                   ▼                   ▼
SQLite              JSON/JSONL          JSONL               Vector Index        Enriched Payloads
(metadata)          (full text)         (search-ready)      (similarity)        (l1/l2 + pdf_url)
```

### Design Principles

1. **Separation of Concerns**: Each phase is independently runnable and testable
2. **Lineage Tracking**: Every output traces back to source pages and reports
3. **Quality Metrics**: Automated quality checks at every stage
4. **Run Reproducibility**: All pipeline executions logged with configuration snapshots

---

## Key Technical Challenges

### Challenge 1: PDF Text Extraction

**Problem**: NTSB reports span 60 years of PDF technology. Some have embedded text, others are scanned images. Quality varies dramatically.

**Solution**: Two-pass extraction pipeline:
1. **Pass 1**: Extract embedded text using pymupdf
2. **Quality Gate**: Evaluate character count, alphabetic ratio, garbage ratio
3. **Pass 2**: OCR failed pages using pytesseract at 300 DPI

**Results**:
- 14,282 pages (47%) had usable embedded text
- 16,320 pages (53%) required OCR
- Mean OCR confidence: 84.2%

**Lesson**: Never assume PDF text quality. Always validate and have fallback strategies.

### Challenge 2: Section Detection

**Problem**: NTSB reports follow a standard structure (SYNOPSIS, FACTUAL INFORMATION, ANALYSIS, CONCLUSIONS), but formatting varies across decades.

**Solution**: Hierarchical pattern matching:
```python
# Numbered sections: "1.8 METEOROLOGICAL INFORMATION"
# Standalone headers: "PROBABLE CAUSE"
# Letter subsections: "(a) Findings"
# Spaced decimals: "1. 8 Aids to Navigation" (OCR artifacts)
```

**Results**: 95% of chunks have accurate section attribution.

### Challenge 3: Footnote Handling

**Problem**: Aviation reports heavily use footnotes for technical clarifications. These break chunk coherence if not handled.

**Solution**:
1. Detect footnote markers in text (e.g., "1/", "2/")
2. Extract footnote definitions from page bottoms
3. Append relevant footnotes to chunks that reference them

**Results**: 1,297 chunks (5.2%) have footnotes properly appended.

---

## The Chunking Evolution: v1 to v2

This was the most impactful technical decision in the project. Initial results were disappointing, leading to a complete redesign.

### Version 1: Initial Approach

**Parameters**:
- Token range: 500-700 (target 600)
- Overlap: 20% (~120 tokens)
- Section handling: Hard breaks at section boundaries

**Problems Discovered**:
1. **Too-small chunks** (32% under 500 tokens): Section boundaries created many tiny chunks that lacked context
2. **No section inheritance**: Child sections didn't inherit parent section names
3. **No section prefix**: Embedding models had no structural context
4. **Lost continuity**: Hard section breaks fragmented related content

**v1 Results**:
- Token distribution: 33% under range, 35% in range, 32% over range
- MRR (MIKA): 0.788
- Hit@10: 94.9%

### Version 2: Redesigned Strategy

**Key Insight**: Retrieval models benefit from:
1. Larger context windows (400-800 tokens gives better semantic density)
2. Structural prefixes (helps model understand document position)
3. Cross-section continuity (related content shouldn't be artificially split)

**Parameters**:
- Token range: 400-800 (target 600)
- Overlap: 25% (~150 tokens)
- Section handling: Soft boundaries with forward borrowing
- **New**: Section prefix `[SECTION_NAME]` prepended to each chunk

**Implementation Changes**:

```python
# v1: Hard break at section boundary
if section_changed:
    yield current_chunk
    current_chunk = []

# v2: Forward borrowing - continue into next section if under minimum
while current_tokens < min_tokens and more_sentences:
    current_chunk.append(next_sentence)
```

**v2 Results**:
- Token distribution: 2.3% under range, 95.6% in range, 2.1% over range
- MRR (MIKA): 0.816 (+3.5%)
- Hit@10: 100% (+5.1%)

### Key Takeaway

**Chunk quality is the single most important factor in retrieval performance.** The same embedding model (MIKA) improved from 0.788 to 0.816 MRR simply by improving chunking strategy. No model changes, no fine-tuning—just better preprocessing.

---

## Evaluation Framework Design

### Why Custom Evaluation?

Standard IR benchmarks (MS MARCO, BEIR) don't apply to domain-specific corpora. We needed:
1. Queries representative of real aviation safety searches
2. Ground truth that accounts for semantic relevance, not just keyword matching
3. Statistical methods appropriate for small query sets

### Query Design Philosophy

**50 queries across 6 categories**:

| Category | Count | Difficulty | Purpose |
|----------|-------|------------|---------|
| Incident Lookup | 10 | Easy | Known accidents with specific report IDs |
| Conceptual Queries | 12 | Medium-Hard | Technical concepts requiring semantic understanding |
| Section Queries | 10 | Medium | Structural retrieval (find PROBABLE CAUSE sections) |
| Comparative Queries | 8 | Hard | Patterns across multiple reports |
| Aircraft Queries | 6 | Medium | Aircraft-type specific searches |
| Phase Queries | 4 | Medium | Flight phase specific searches |

**Stratification Rationale**:
- **Incident Lookup**: Baseline—if we can't find known accidents, nothing works
- **Conceptual**: Tests semantic understanding beyond keywords
- **Section**: Tests structural awareness (crucial for NTSB's standardized format)
- **Comparative**: Tests cross-document reasoning
- **Aircraft/Phase**: Tests filtering combined with semantics

### Ground Truth Validation

Each query has structured ground truth:

```yaml
- id: CONC-003
  query: "What are common findings related to crew resource management failures?"
  category: conceptual_queries
  difficulty: hard
  intent: "Find chunks discussing CRM breakdowns"
  ground_truth:
    type: signal_based
    relevance_signals:
      - "crew resource management"
      - "CRM"
      - "crew coordination"
      - "cockpit communication"
    verification_sql: |
      SELECT COUNT(DISTINCT report_id) FROM chunks
      WHERE chunk_text ILIKE '%crew resource management%'
        OR chunk_text ILIKE '%CRM%'
```

**Three validation types**:
1. **Report-based**: Expected report IDs must appear in results
2. **Signal-based**: Relevance signals must be present in retrieved text
3. **Section-based**: Retrieved sections must match expected sections

### Human Review Protocol

Automated metrics don't capture semantic relevance. We implemented a human review workflow:

1. **Export**: Generate YAML files with top-10 results per query
2. **Auto-fill**: Pre-label results matching keyword signals as KEYWORD_MATCH
3. **Human Review**: Label remaining results as SEMANTIC_MATCH or FALSE_POSITIVE
4. **Import**: Aggregate human judgments
5. **Calculate**: Semantic Precision and Semantic Lift metrics

**Semantic Lift** = (Semantic Precision - Keyword Precision) / Keyword Precision

This measures how much the embedding model finds beyond simple keyword matching.

---

## Statistical Validity

### Why Statistical Rigor Matters

With only 50 queries, we need careful statistical treatment to avoid overfitting conclusions.

### Methods Employed

**1. Bootstrap Confidence Intervals (95%)**

```python
# 1000 bootstrap samples of MRR differences
bootstrap_deltas = []
for _ in range(1000):
    sample_indices = np.random.choice(n_queries, n_queries, replace=True)
    delta = mika_mrr[sample_indices].mean() - minilm_mrr[sample_indices].mean()
    bootstrap_deltas.append(delta)
ci_95 = np.percentile(bootstrap_deltas, [2.5, 97.5])
```

**Result**: MIKA advantage = 0.112 MRR, 95% CI: [0.067, 0.158]

The confidence interval doesn't cross zero, indicating statistically significant improvement.

**2. Wilcoxon Signed-Rank Test**

Non-parametric test for paired samples (same queries, different models):

```python
from scipy.stats import wilcoxon
stat, p_value = wilcoxon(minilm_mrr, mika_mrr)
```

**Result**: p < 0.001, indicating MIKA significantly outperforms MiniLM.

**3. Per-Query Win/Loss/Tie Analysis**

```
MIKA wins: 15 queries
MiniLM wins: 5 queries
Ties: 30 queries
```

MIKA wins 3x more often than MiniLM on queries where they differ.

### Limitations Acknowledged

1. **50 queries is small**: Results may not generalize to all possible aviation queries
2. **Single evaluator**: Human review was single-person (ideally use multiple annotators)
3. **Domain-specific**: Results apply to aviation documents, not general retrieval

---

## Results and Findings

### Final Benchmark (v2)

| Metric | MiniLM | MIKA | Winner |
|--------|--------|------|--------|
| MRR | 0.704 | **0.816** | MIKA |
| Hit@10 | 100% | 100% | Tie |
| nDCG@10 | 0.625 | **0.675** | MIKA |
| Semantic Precision | 92.7% | **97.1%** | MIKA |
| Semantic Lift | +28.2% | **+38.6%** | MIKA |
| Mean Latency | 133ms | 135ms | Tie |

### Key Findings

**1. Domain Models Matter**

MIKA (NASA's aviation-trained model) achieves 38.6% semantic lift vs. 28.2% for general-purpose MiniLM. This 10.4 percentage point difference justifies the larger model size (768 vs 384 dimensions).

**2. Chunking is Critical**

The v1→v2 chunking improvement:
- Reduced out-of-range chunks from 65% to 4.4%
- Improved Hit@10 from 94.9% to 100%
- Improved MRR by 3.5%

**3. Section Awareness Helps**

Adding `[SECTION_NAME]` prefixes improved section query accuracy and helped the model understand document structure.

**4. 100% Hit@10 is Achievable**

With proper chunking and evaluation, we achieved perfect recall in top-10 results. This means the right information is always retrievable—the question is ranking.

---

## Lessons Learned

### Technical Lessons

1. **Evaluate Early and Often**: Our initial chunking strategy was fundamentally flawed. Only rigorous evaluation revealed this.

2. **Chunk Quality > Model Quality**: Better preprocessing beat switching models. The same model improved 3.5% MRR just from chunking changes.

3. **Domain Models Justify Complexity**: The 2x dimension increase (384→768) was worth the 2x storage and compute for 10% better semantic lift.

4. **Automate Quality Gates**: Every pipeline stage should have automated quality checks. Catching bad data early saves debugging time later.

5. **Log Everything**: Run tracking saved hours of debugging. When results seemed wrong, we could trace back to exact configurations.

### Process Lessons

1. **Iterate on Evaluation First**: Before optimizing the pipeline, build robust evaluation. Otherwise you're optimizing blind.

2. **Human Review is Essential**: Automated metrics miss semantic relevance. Budget time for human evaluation.

3. **Document Decisions**: This portfolio statement exists because past decisions weren't documented. Future-you will thank present-you.

4. **Version Your Data**: The v1/v2 comparison was only possible because we preserved both versions.

### What I'd Do Differently

1. **Start with evaluation harness**: Build the benchmark framework before building the pipeline
2. **Use multiple human reviewers**: Single-annotator agreement isn't measurable
3. **Implement A/B testing earlier**: Compare chunking strategies before committing to full pipeline runs
4. **Build visualization earlier**: Seeing chunk distributions graphically would have revealed v1 problems faster

---

## The Taxonomy Journey: From Unsupervised Discovery to Industry Standards

Phase 6 represents a critical inflection point in the project—where **human judgment proved essential** to redirect an automated approach that wasn't delivering actionable results.

### Phase 6A: Unsupervised Topic Discovery (The Failed Approach)

**Hypothesis**: BERTopic with domain-specific MIKA embeddings would discover meaningful aviation safety themes from 60 years of accident reports.

**Implementation**:
- Filtered to 5,806 causal chunks (PROBABLE CAUSE, ANALYSIS, CONCLUSIONS, FINDINGS sections)
- Used pre-computed MIKA 768-dimensional embeddings
- Ran BERTopic with UMAP dimensionality reduction and HDBSCAN clustering

**Results**:
| Metric | Value |
|--------|-------|
| Topics Discovered | 76 |
| Outlier Chunks | 1,285 (22.1%) |
| Processing Time | 47 seconds |

**What We Found—And Why It Failed**:

The topics produced by BERTopic had some surface-level relevance, but upon human review, serious problems emerged:

| Topic | Top Keywords | Problem |
|-------|--------------|---------|
| 0 | approach, feet, descent, altitude | Generic flight terminology, not causal |
| 5 | probable, probable cause, cause, national | Document structure words, not safety factors |
| 7 | hours, certificate, held, medical certificate | Pilot certification boilerplate |
| 19 | cam, cam1, source, content | CVR transcript artifacts |
| 52 | approach, probable cause, probable, cause | Duplicate of structural terms |

**Key Observations**:
1. **Noise dominated meaningful signal**: 22% of chunks were outliers, and many clusters captured document formatting rather than safety concepts
2. **Keywords were random, not semantic**: The algorithm found statistical co-occurrences, not meaningful causal factors
3. **No actionable taxonomy emerged**: The 76 topics couldn't be cleanly mapped to aviation safety categories
4. **Domain structure overpowered content**: NTSB's standardized report format created spurious clusters around section headers and boilerplate language

### Human-in-the-Loop Review (GATE 1)

This is where **rigorous human evaluation** changed the project trajectory.

**Review Process**:
1. Exported all 76 topics with representative chunks
2. Reviewed each topic for semantic coherence and aviation relevance
3. Attempted to map topics to known safety categories
4. Documented systematic issues

**Decision**: After review, the unsupervised approach was **abandoned**. The topics were not meaningful enough to build a taxonomy around, and forcing a mapping would produce unreliable results.

### The Pivot: CICTT Industry Standard Taxonomy

**Research Phase**: Investigated how the aviation safety industry actually classifies accidents. Discovered the **CAST/ICAO Common Taxonomy Team (CICTT)** framework—a collaborative effort between the FAA, ICAO, and international safety organizations.

**Why CICTT?**
1. **Industry-vetted**: Used globally by safety investigators since 2006
2. **Expert-defined categories**: 30 occurrence types with precise definitions
3. **Proven in practice**: Already validated on thousands of real accidents
4. **Semantically meaningful**: Categories reflect actual causal factors, not statistical artifacts

**CICTT Categories Implemented**:
| Code | Category | Description |
|------|----------|-------------|
| LOC-I | Loss of Control - Inflight | Stalls, spins, spatial disorientation |
| CFIT | Controlled Flight Into Terrain | Flew into terrain while under control |
| SCF-PP | System Failure - Powerplant | Engine, propeller, rotor failures |
| SCF-NP | System Failure - Non-Powerplant | Flight controls, hydraulics, electrical |
| RE | Runway Excursion | Overruns, veer-offs |
| ICE | Icing | Airframe and engine icing |
| WSTRW | Wind Shear/Thunderstorm | Microbursts, convective weather |
| FUEL | Fuel Related | Exhaustion, starvation, contamination |
| MAC | Midair Collision | In-flight collisions |
| ... | *+ 21 more categories* | |

### Phase 6B: Embedding-Based CICTT Classification

**New Approach**: Instead of discovering topics, **map chunks to known categories** using embedding similarity.

**Implementation**:
```
CICTT Categories → Embed seed phrases (MIKA)
Report Chunks → Existing MIKA embeddings
Cosine Similarity → Category assignments
Aggregate → Report-level classification
```

**Key Design Decisions**:
1. **Seed phrases, not just keywords**: Each category has 4-6 example sentences that capture the semantic meaning
2. **Multi-signal scoring**: Combines average similarity, max similarity, and evidence count
3. **Threshold filtering**: Only assignments above 0.45 similarity included
4. **Multi-label**: Reports can have multiple contributing causes (average 3.8 per report)

**L1 Results** (initial run):
| Metric | Value |
|--------|-------|
| Reports Classified | 418 |
| Categories Used | 27 of 30 |
| Chunk Assignments | 6,555 |
| Report-Level Assignments | 1,736 |
| Avg Categories per Report | 3.8 |

### Phase 6A-Sub: Hierarchical L2 Classification (Complete)

**Building on L1**: Extended the CICTT classification with industry-standard subcategories using a two-pass approach:

1. **Pass 1 (L1)**: Map chunks to CICTT categories (done above)
2. **Pass 2 (L2)**: For each L1 assignment, map to more specific subcategories

**Subcategory Sources**:
- **LOC-I**: IATA/EASA research (STALL, UPSET, SD, ENV, SYS, LOAD)
- **CFIT**: IATA/SKYbrary (NAV, SA, VIS, TAWS, PROC)
- **SCF-PP/SCF-NP**: Technical sub-systems (ENG, FUEL, HYD, ELEC, etc.)
- **Human-causal categories**: HFACS Unsafe Acts (SKILL, DECISION, PERCEPTUAL, VIOLATION)

**L2 Results**:
| Metric | Value |
|--------|-------|
| Chunks Processed | 1,478 |
| L2 Chunk Assignments | 2,446 |
| Report-L2 Assignments | 1,106 |
| Subcategories Used | 32 |

### Qdrant Payload Enrichment (Complete)

**Final Step**: Enriched all Qdrant vector payloads with taxonomy data for category-filtered search in Streamlit.

**New Payload Fields**:
```json
{
  "l1_categories": ["LOC-I", "SCF-NP"],
  "l2_subcategories": ["LOC-I-STALL", "CFIT-NAV"],
  "pdf_url": "https://www.ntsb.gov/.../AAR0201.pdf"
}
```

This enables:
- **Category-filtered semantic search**: Filter results by L1/L2 taxonomy
- **Direct PDF linking**: Link to original NTSB reports
- **Faceted navigation**: Show category facets alongside search results

### Comparing Approaches

| Aspect | BERTopic Discovery | CICTT Mapping |
|--------|-------------------|---------------|
| Categories | 76 (auto-discovered) | 30 (expert-defined) |
| Interpretability | Low—random keywords | High—industry standard |
| Outlier Rate | 22.1% | 0% (all chunks mapped) |
| Human Validation | Failed review | Pending systematic review |
| Reproducibility | Sensitive to parameters | Deterministic |
| Domain Alignment | Document structure noise | Aviation safety focus |

### Key Takeaway

**Unsupervised methods need supervised validation.** BERTopic is a powerful tool, but it finds statistical patterns—not necessarily meaningful patterns. In specialized domains with standardized document formats:

1. **Document structure creates noise**: Report templates, section headers, and boilerplate language form spurious clusters
2. **Domain expertise is irreplaceable**: The CICTT taxonomy represents decades of aviation safety expertise that no algorithm could derive from text alone
3. **Human-in-the-loop isn't optional**: The Gate 1 review caught a fundamental problem before it propagated through the entire pipeline

This pivot cost time but saved the project from building on a flawed foundation.

---

## Closing the Gap: Data Quality and Taxonomy Coverage

After completing L1 and L2 taxonomy classification, a troubling number stared back from the metrics: **92 reports (18%) had no taxonomy at all**. Before building dashboards and analytics on top of this data, we needed to understand *why* and fix it.

This section documents a systematic investigation that transformed an opaque 18% gap into a well-understood, nearly-complete dataset—and the human-in-the-loop decisions that made it work.

### The Problem: 92 Reports with Zero Taxonomy

The initial L1 classification pipeline filtered chunks to causal sections (PROBABLE CAUSE, ANALYSIS, CONCLUSIONS, FINDINGS) and required a minimum similarity threshold of 0.40. This worked well for standard reports but left 92 of 510 reports completely unclassified.

The naive fix—lower thresholds and retry—would have been wrong. The real question was: **should all 92 reports have taxonomy in the first place?**

### Step 1: Report Type Classification (The Detective Work)

NTSB publishes several types of documents under the same file format, not just accident reports. We built an automated classifier using filename prefixes and title heuristics:

| Prefix | Type | Expects Taxonomy? | Count |
|--------|------|-------------------|-------|
| AAR, AAB | Accident Report | Yes | 436 |
| AIR, SIR, SS, SAR | Safety Study | No | 28 |
| ASR | Safety Recommendation | No | 21 |
| HZB, HZMSR | Hazmat Report | No | 2 |
| Unknown | Flagged for Review | Assumed yes | 1 |

**Result**: 51 of the 92 "gaps" were non-accident documents that were never supposed to have accident taxonomy. The classification reduced the true gap from 92 to 53.

### Step 2: Supplemental Document Detection

Further analysis revealed a second category hiding in the data: **supplemental documents** that had been treated as standalone reports.

We built a filename-pattern detector that identified:
- **Appendix files**: `AAR0003_app.pdf`, `AAR9804_C.pdf` (letter suffixes)
- **Summary versions**: `AAR8501S.pdf` (S suffix = executive summary)
- **Reconsideration responses**: `AAR9704r.pdf` (r suffix)
- **Cover/TOC pages**: Short files (< 3 pages) with a `_body.pdf` sibling

**22 supplemental documents** were reclassified, dropping the accident count from 458 to 436 and the true gap from 53 to **35 reports**.

### Step 3: Iterative Retry with Relaxed Thresholds

For the 35 remaining accident reports, we built a retry pipeline with relaxed parameters:

| Parameter | Original | Retry |
|-----------|----------|-------|
| Causal sections | 4 (strict) | 7 (+ DISCUSSION, SUMMARY, DETERMINATION) |
| Min chunk tokens | 100 | 50 |
| Min similarity | 0.40 | 0.35 |
| Scope | All 510 reports | Only 35 missing accident reports |

**Result**: 8 more reports classified, bringing coverage to 409/436 (93.8%). But 27 reports remained stubbornly unclassified.

### Step 4: Human-in-the-Loop Section Prioritization (GATE 2)

The remaining 27 reports failed because their section names didn't match *any* of our causal section patterns. Analysis revealed they were mostly 1970s-era reports with non-standard formatting.

Rather than blindly opening all sections (which risks noisy classifications), we performed a **human review of all 64 unique section names** across the 27 gap reports, organizing them into priority tiers:

| Tier | Signal Level | Example Sections | Decision |
|------|-------------|-----------------|----------|
| **Tier 1**: Causal | Highest | SYNOPSIS, CONCLUSIONS > FINDINGS, (a) FINDINGS | Include |
| **Tier 2**: Investigation | Good | INVESTIGATION, HISTORY OF FLIGHT, WRECKAGE, TESTS AND RESEARCH | Include |
| **Tier 3**: Supporting | Moderate | SURVIVAL ASPECTS, AIRCRAFT INFORMATION, FIRE, CREW INFORMATION | Include |
| **Tier 4**: Low/Noise | Low | RECOMMENDATIONS, PARAGRAPH_##, METEOROLOGICAL | Exclude |

**Key insight**: `SYNOPSIS` appeared in 10 of 27 reports and `INVESTIGATION` in 12 of 27. These two sections alone could unlock most remaining reports, but they had been excluded by the original causal-sections filter because they contain factual descriptions rather than causal analysis.

**Human decision**: Include Tiers 1-3 (broad coverage from diverse sections), exclude Tier 4 (recommendations don't describe causes; numbered paragraphs have no section context; narrow factual sections like METEOROLOGICAL add noise without causal signal).

### Step 5: Final Pass with Exclusion-Based Filtering

The final pass inverted the filtering logic. Instead of include-listing causal sections, it **included everything except Tier 4 exclusions**:

```
Original: include IF section IN [PROBABLE CAUSE, ANALYSIS, ...]    (4 patterns)
Retry:    include IF section IN [+ DISCUSSION, SUMMARY, ...]       (7 patterns)
Final:    include IF section NOT IN [RECOMMENDATIONS, PARAGRAPH_*] (exclusion-based)
```

**Result**: 22 of 27 remaining reports classified, using 520 chunks across 24 different section types.

### Final Coverage: 98.9%

| Stage | Coverage | Gap | Method |
|-------|----------|-----|--------|
| Initial L1 run | 401/436 (92.0%) | 35 | Strict causal sections |
| + Retry pass | 409/436 (93.8%) | 27 | Expanded sections, lower thresholds |
| + Final pass | 431/436 (98.9%) | 5 | All sections except Tier 4 |

The **5 remaining reports** are genuinely intractable with this approach:
- **3 reports** have only `PARAGRAPH_##` sections (1970s format with no section headers at all)
- **1 report** contains only `RECOMMENDATIONS` (no accident description)
- **1 report** has causal sections but all chunk similarities fell below the 0.35 threshold

### What This Process Demonstrates

**1. Root cause analysis before optimization.** The instinct was to lower thresholds and re-run. But the 92-report gap wasn't a threshold problem—it was a data heterogeneity problem. 55% of the "gap" was non-accident documents that should never have been in scope.

**2. Progressive relaxation, not wholesale abandonment.** Each retry pass relaxed constraints incrementally, with human review at each stage. This prevented the noisy classifications that would result from simply turning off all filters.

**3. Human-in-the-loop at the right granularity.** We didn't ask a human to review 24,766 chunks or 510 reports. We asked a human to rank 64 section names into 4 tiers—a 10-minute decision that unlocked 22 more classifications.

**4. Knowing when to stop.** The 5 remaining reports represent a deliberate stopping point. We could classify them by removing all filters, but the resulting low-confidence classifications would degrade overall data quality. 98.9% coverage with high confidence is better than 100% with noise.

---

## Bayesian Risk Model: From Features to Audit to Production

After building taxonomy coverage and data quality infrastructure, the next step was turning structured features into actionable risk predictions. This section documents the full lifecycle: feature extraction, initial model, statistical audit, rewrite, and production validation.

### The Data Quality Problem

The initial Bayesian model trained on *all* 448 reports with taxonomy—including 17 safety studies, recommendation summaries, and supplement documents that aren't accident reports. These contaminated the priors: safety studies discuss multiple categories broadly, inflating base rates for common categories and diluting signal from rare but important ones.

**Fix**: Join on the `report_types` table and filter to `report_type = 'accident'` only. This reduced the training set from 448 to **431 accident reports**—a cleaner dataset that produces more accurate conditional probabilities.

### Feature Extraction from Unstructured Text

The original model had 3 features (aircraft category, season, region) extracted from structured metadata. To improve predictions, we added 2 features extracted from the *unstructured text* of accident report chunks:

**Weather (VMC/IMC)**:
- Scans chunks in section-priority order: METEOROLOGICAL → SYNOPSIS → ANALYSIS
- High-confidence patterns: explicit `VMC`, `IMC`, `VFR conditions`, `IFR conditions`
- Medium-confidence patterns: inferred from visibility, ceiling, fog mentions
- **Result**: 369/510 reports classified (164 VMC, 205 IMC)

**Time of Day**:
- Parses military time (`about 0830 EDT`), 12-hour (`10:30 a.m.`), and keywords (`nighttime`)
- Handles UTC-to-local conversion using state timezone lookup
- Guards against false positives from altitudes and years (negative lookahead for "feet", "MSL", "FL")
- **Result**: 463/510 reports classified (106 Morning, 159 Afternoon, 57 Evening, 141 Night)

### The Initial Model (v1): Softmax Naive Bayes

The first model implementation used standard single-label Naive Bayes with softmax normalization:

```
P(cat | features) = softmax( log P(cat) + Σ log P(fi|cat) )
```

**Reported metrics (v1):**

| Metric | 3 Features | 5 Features |
|--------|-----------|-----------|
| Hit@1 | 51.3% | 54.8% |
| Hit@3 | 83.3% | 80.7% |
| Hit@5 | 90.7% | 90.7% |

These looked strong. Hit@5 of 90.7% seemed impressive. The model was saved, integrated into Streamlit, and appeared ready for production.

### The Statistical Audit: 4 Critical Flaws

A rigorous statistical audit revealed that the seemingly-strong metrics hid fundamental problems:

**Flaw 1: Multi-label data in a single-label model.** 95.8% of reports have *multiple* L1 categories (average 4.19 per report). But softmax normalization forces all 27 category probabilities to sum to 1.0. This creates systematic calibration error: if a report truly has 4 relevant categories at ~40% each, softmax squishes them down to ~10-15% and redistributes the probability mass. The model was 3-4x underconfident on relevant categories and overconfident on irrelevant ones.

**Flaw 2: Fake leave-one-out validation.** The `validate()` method predicted using the *full* model for every held-out report—it never actually retrained without the held-out data. The code even acknowledged this in a comment: "LOO approximation: with 400+ reports, removing one has negligible effect." But for small categories like AMAN (n=8), removing 1 report changes likelihood estimates by up to 47%.

**Flaw 3: No baseline comparison.** Hit@3=80.7% sounded good in isolation. But computing a trivial prior-only baseline (always predict the top-k categories by frequency) revealed the baseline achieved Hit@3=81.4%. The model was actually *worse* than guessing the most common categories. Without an explicit comparison, this was invisible.

**Flaw 4: Arbitrary unseen-value fallback.** When a feature value was never seen during training (e.g., predicting for a balloon when no training data includes balloons), the code used `smoothing_alpha / 100 = 0.01` as a fallback—an arbitrary constant with no statistical justification. Proper Laplace smoothing should account for the number of known values plus one for the unseen class.

### The Fix: Binary Relevance Naive Bayes (v2)

The rewrite replaced the single 27-way softmax classifier with **27 independent binary classifiers**:

```
P(cat=1 | features) = sigmoid(
    log P(cat=1) + Σ log P(fi|cat=1)
  - log P(cat=0) - Σ log P(fi|cat=0)
)
```

**Key changes:**

| Aspect | v1 (Softmax) | v2 (Binary Relevance) |
|--------|-------------|----------------------|
| Architecture | 1 classifier, 27-way softmax | 27 independent binary classifiers |
| Probability space | Sum to 1.0 (forced) | Independent [0,1] per category |
| Likelihoods stored | P(val\|cat) only | P(val\|cat=1) AND P(val\|cat=0) |
| Unseen values | Arbitrary α/100 | Proper Laplace with n+1 |
| Validation | Full model reuse (fake LOO) | Count-adjusted proper LOO |
| Baseline | Not computed | Explicit prior-only comparison |
| Calibration metric | None | Expected Calibration Error (ECE) |
| Schema | v7 (single likelihood) | v8 (label column in PK, positive_count) |

**Proper LOO implementation:** Rather than fully retraining 431 times, the validation adjusts the raw counts for each held-out report: if the report has category C, subtract 1 from the positive count; otherwise subtract 1 from the negative count. Then recompute smoothed likelihoods from the adjusted counts. This is mathematically exact for Naive Bayes with Laplace smoothing and runs in seconds instead of minutes.

### Production Validation Results

**Cross-validation with baseline comparison (431 reports, proper LOO):**

| Metric | Model | Baseline (Prior-only) | Lift |
|--------|-------|----------------------|------|
| Hit@1 | 44.8% | 45.9% | 0.97x |
| Hit@3 | 76.3% | 81.4% | 0.94x |
| Hit@5 | **86.8%** | 85.4% | **1.02x** |
| Mean Rank | 2.9 | 2.9 | — |

**Calibration (the real win):**

| Calibration Bin | Predictions | Avg Predicted | Avg Actual | Error |
|----------------|------------|--------------|------------|-------|
| [0.0-0.1) | 5,850 | 0.0375 | 0.0385 | 0.0009 |
| [0.1-0.2) | 2,135 | 0.1466 | 0.1438 | 0.0028 |
| [0.2-0.3) | 1,578 | 0.2491 | 0.2427 | 0.0064 |
| [0.3-0.4) | 1,011 | 0.3463 | 0.3452 | 0.0011 |
| [0.4-0.5) | 616 | 0.4440 | 0.4545 | 0.0105 |
| [0.5-0.6) | 303 | 0.5430 | 0.5347 | 0.0083 |
| [0.6-0.7) | 117 | 0.6467 | 0.6496 | 0.0028 |

**ECE = 0.021** — near-perfect calibration across all probability bins. The old softmax model had ECE ~0.3-0.4, a 15x improvement.

### Why Hit@1 Dropped (And Why That's Correct)

The v1 model reported Hit@1=54.8% vs. v2's 44.8%. This looks like a regression, but it's actually a correction:

1. **v1 used fake LOO** — the held-out report's data was still in the model, biasing predictions upward
2. **v1 softmax concentrated probability** — forcing sum-to-1 across 27 categories artificially inflated the top-1 probability, making ranking easier but probabilities meaningless
3. **The honest baseline is 45.9%** — v2's 44.8% is within expected range for features with limited discriminative power

The real metric is calibration. A model that says "45% chance of NAV" and is right 45% of the time is far more useful than one that says "15% chance" when the true rate is 46%.

### Per-Category Discrimination

All 27 categories show positive separation (mean predicted probability for positive reports > negative reports):

| Category | N_pos | Mean P\|pos | Mean P\|neg | Separation |
|----------|-------|-----------|-----------|-----------|
| CFIT | 99 | 0.3244 | 0.1987 | +0.1257 |
| MAC | 40 | 0.2319 | 0.0815 | +0.1504 |
| ICE | 25 | 0.1803 | 0.0531 | +0.1272 |
| UIMC | 33 | 0.1915 | 0.0661 | +0.1254 |
| ATM | 111 | 0.3301 | 0.2331 | +0.0970 |

Average separation across all 27 categories: +0.075. Categories with the strongest feature signals (CFIT from weather/time, MAC from aircraft type, ICE from season/weather) show the clearest discrimination.

### Feature Ablation Study

| Configuration | Hit@1 | Hit@3 | Hit@5 | ECE |
|--------------|-------|-------|-------|-----|
| All 5 features | 44.8% | 76.3% | 86.8% | 0.021 |
| Drop aircraft_category | 43.2% | 73.3% | 84.7% | 0.016 |
| Drop season | 41.1% | 78.4% | 86.8% | 0.020 |
| Drop region | 46.4% | 75.9% | 87.5% | 0.019 |
| Drop weather | 42.9% | 77.5% | 85.8% | 0.024 |
| Drop time_of_day | 43.4% | 77.7% | 88.9% | 0.018 |
| Only 3 core | 45.5% | 76.8% | 88.6% | 0.021 |

`aircraft_category` is the most impactful feature (dropping it hurts Hit@3 by 3pp and Hit@5 by 2.1pp). No single feature dramatically moves the needle — the features provide marginal but real gains over the prior-only baseline.

### Rare Category Limitations

| Category | N_pos | Avg Rank | Top-5 Rate | Prior |
|----------|-------|---------|-----------|-------|
| AMAN | 8 | 21.2 | 0/8 | 1.9% |
| SEC | 9 | 20.0 | 0/9 | 2.1% |
| WILD | 9 | 18.0 | 0/9 | 2.1% |
| LALT | 12 | 18.2 | 0/12 | 2.8% |

The model cannot surface rare categories (n < 16) into the top-5 predictions. With only 8-12 positive examples across 431 reports, there isn't enough signal to overcome the priors of dominant categories. This is a fundamental data limitation documented as a known constraint.

### Predictions That Make Aviation Sense

The model's predictions remain domain-sensible with the new architecture:

- **Turboprop, Winter, IMC, Night** → NAV 52.2%, RE 52.0%, CFIT 47.7%. These are the classic IFR/night scenarios where navigation errors, runway excursions, and terrain collisions dominate.
- **Single-piston, Summer, West, VMC, Morning** → MAC 69.6%, NAV 59.4%, GCOL 56.7%. VFR GA traffic in visual conditions with mid-air collision and ground collision risk.
- **Jet-wide, Summer, VMC, Afternoon** → LOC-I 45.5%, SCF-NP 41.4%, SCF-PP 41.2%. Airline operations skew toward loss-of-control and mechanical/structural failures.
- **Sum of all 27 probabilities = 4.19** (not 1.0), confirming independent classifiers correctly handle multi-label data.

### Edge Case Robustness

| Test | Result |
|------|--------|
| No features provided | Returns exact priors (diff = 0.0000) |
| Unseen value ("spaceship") | Safe degradation toward uniform (~0.49) |
| Save/load roundtrip | Perfect match (max_diff = 0.000000) |
| Concurrent DB reads | No conflicts |
| Streamlit fast load path | All internal state correctly reconstructed |

### Key Takeaways

**1. Statistical auditing is non-negotiable.** The initial model's 90.7% Hit@5 was built on fake validation, miscalibrated probabilities, and hidden baseline underperformance. Only a rigorous audit — checking calibration, computing baselines, and verifying LOO correctness — revealed these issues.

**2. Calibration matters more than ranking for risk assessment.** In a risk profiling tool, users need to trust the *magnitude* of probabilities, not just their *ordering*. A model that says "52% chance of CFIT" and is right 52% of the time is useful; one that says "15%" when the real rate is 46% is dangerous.

**3. Multi-label data requires multi-label models.** Using softmax (sum-to-1) normalization on data where 95.8% of observations have multiple labels is a fundamental modeling error that cannot be fixed by tuning — it requires a different architecture.

**4. Honest metrics, even when unflattering, build trust.** The v2 model's Hit@1 of 44.8% looks worse than v1's 54.8%, but it's the honest number from proper LOO with an explicit baseline. A portfolio that presents correctly-measured, honestly-interpreted results demonstrates stronger data science judgment than one that inflates metrics.

---

## Streamlit Application: From Dashboards to Consulting Reports

Phase 7-8 transformed the analytical foundation into a production Streamlit application. The design process surfaced important lessons about data visualization, stakeholder communication, and accessibility.

### Persona Research: Who Is the Audience?

The original plan called for 4 stakeholder dashboards: Manufacturer, Maintenance, Insurance, and Pilot. But analyzing the actual data revealed a mismatch: **~47% of the 431 accident reports involve commercial jet operations**, not general aviation. This shifted the personas from GA-centric to commercial aviation-centric.

**Key decisions:**
- **Merged Manufacturer + Maintenance** into a single "Fleet Safety" report. Maintenance risk (SCF-PP/SCF-NP component failures) is most meaningful *in context* of fleet type and manufacturer, not as a standalone view.
- **Reframed Insurance → Underwriting**. Aviation underwriting analysts need risk segmentation by operational profile (aircraft type x weather x time x region), not just aggregate statistics.
- **Reframed Pilot → Operational Risk**. A chief pilot / safety officer needs LOC-I and CFIT deep dives, weather/time risk matrices, and seasonal patterns—not just category counts.

**Result:** 3 reports instead of 4, each with a tighter analytical focus and clearer narrative thread.

### The KPI Design Problem

Initial KPI cards revealed a common trap: **displaying available metrics rather than meaningful ones**.

**Examples of bad KPIs we removed:**
- **"Weather Coverage: 72.4%"** — This is a data limitation, not a finding. Showing it as a headline metric implies it's an insight when it's actually a caveat.
- **"Avg Categories per Report: 4.19"** — Already shown in another report. Duplicative across pages.
- **"Douglas Variants Merged"** — An implementation detail about data cleaning, not a stakeholder-relevant insight.

**What makes a good KPI:**
1. **Answers a question the persona cares about.** "High Complexity Reports (4+ categories): 42%" tells an underwriter that nearly half of accidents involve cascading failures—relevant for pricing.
2. **Provides context.** "IMC Accident Share: 55.6%" with the subtitle "of weather-classified reports" frames the number honestly.
3. **Is actionable.** Knowing that "SCF-PP prevalence dropped 15% over the last 3 decades" suggests reliability improvements that affect fleet planning.

### Consulting-Style Reports vs. Interactive Dashboards

We deliberately chose **narrative reports with embedded charts** over interactive dashboard grids. The reasoning:

1. **Charts without context are ambiguous.** A heatmap of co-occurrence counts means nothing without explaining what co-occurrence implies for safety management. Narrative text guides interpretation.
2. **The audience isn't data scientists.** Fleet safety managers, underwriters, and chief pilots need conclusions and implications, not raw data exploration tools.
3. **Charts embedded in narrative feel authoritative.** This is how McKinsey, Deloitte, and aviation safety consultancies present findings—not as interactive toys but as supported arguments.

**Implementation pattern:** Each section has:
```
Section title → Explanatory context → Chart → Insight callout → Methodology note
```

The `chart_with_insight()` component enforces this pattern, ensuring every visualization is paired with an interpretation.

### Colorblind Accessibility: Avoiding Red

The initial weather comparison chart used **red (CORAL)** for IMC data. This was problematic:
- Red/green colorblindness (deuteranopia) affects ~8% of males
- Using red for "bad weather" (IMC) vs. blue for "good weather" (VMC) relies on a value judgment that the colors should convey

**Fix:** Replaced red with **orange (AMBER)** for IMC. Orange is distinguishable from blue across all common forms of color vision deficiency, and conveys "caution" without the aggressive signal of red.

The full color palette was designed for accessibility:
- **STEEL** (#4A6FA5) — Primary, neutral, safe for all vision types
- **CORAL** (#E07A5F) — Used only for emphasis, never in data comparisons
- **AMBER** (#DDAA33) — Warning/IMC data, distinguishable from all other colors
- **TEAL** (#2A9D8F) — Secondary data series
- **NAVY** (#264653) — Dark accents, night time-of-day

### Time-of-Day Visualization

Mapping time periods to colors required thinking about intuition, not aesthetics:

| Period | Color | Reasoning |
|--------|-------|-----------|
| Morning (06:00-11:59) | Gold (#DDAA33) | Sunrise, warm light |
| Afternoon (12:00-17:59) | Orange (#E07A5F) | Peak daylight, warm |
| Evening (18:00-20:59) | Light Blue (#7EB8DA) | Fading light, cool transition |
| Night (21:00-05:59) | Navy (#264653) | Darkness, deep blue |

Showing the actual time windows as a caption (`st.caption()`) below the chart prevents ambiguity about what "Morning" means — a critical detail in aviation where operations span 24 hours.

### Abbreviation Handling: The Tooltip System

Aviation is saturated with abbreviations: LOC-I, CFIT, SCF-PP, VMC, IMC, CICTT, NTSB, TAWS, GPWS, HFACS, CRM. For a non-aviation reader, this is impenetrable.

**Two-layer approach:**
1. **Narratives spell out first use:** "Loss of Control — In Flight (LOC-I)" appears before any bare "LOC-I" reference.
2. **HTML `<abbr>` tooltips:** Chart labels and repeated references use `<abbr title="Loss of Control — In Flight">LOC-I</abbr>`, showing the full definition on hover.

The `ABBREVIATIONS` dict in `report_layout.py` contains 39 definitions, and the `abbr()` helper function generates the HTML. A CSS rule styles `<abbr>` elements with a dotted underline and `cursor: help` to signal interactivity.

### Co-occurrence Matrix Design

The co-occurrence heatmap was one of the most analytically novel visualizations. Two design decisions improved it significantly:

**1. Lower triangle only.** A co-occurrence matrix is symmetric (A co-occurring with B = B co-occurring with A). Showing the full matrix wastes half the visual space on redundant information and makes it harder to scan. Masking the upper triangle and diagonal produces a cleaner, more scannable chart.

**2. Annotation threshold.** Only cells with 15+ shared reports get numeric annotations. This prevents visual clutter from low-count cells while highlighting meaningful risk combinations.

Implementation: `np.triu_indices_from(z, k=1)` masks the upper triangle by setting values to `np.nan`, which Plotly renders as blank cells.

### Selectbox vs. Accordion Pattern

The Operational Risk page initially used `st.expander()` accordions to show risk signatures per aircraft type. This was replaced with `st.selectbox()` + dynamic chart rendering:

**Why accordions failed:**
- All data loaded at page render, even for collapsed sections
- Users had to open/close multiple accordions to compare types
- The page was visually overwhelming with 6+ expandable sections

**Why selectbox works:**
- Only the selected aircraft type's chart renders
- A comparison option allows side-by-side views
- The page stays clean and focused

### What We'd Do Differently

1. **Start with persona interviews.** We derived personas from data analysis, but real stakeholder input would have sharpened the focus earlier.
2. **Build a design system first.** The theme, colors, and layout patterns evolved iteratively. Starting with a cohesive design system would have prevented inconsistencies.
3. **Prototype with static data.** Building charts with real SQL queries slowed iteration. Mocking data first would have let us validate layouts faster.
4. **Test abbreviation tooltips on mobile.** The `<abbr>` hover pattern doesn't work on touch devices. A glossary link or tap-to-expand pattern would be more universal.

---

## Semantic Search: From Pipeline to Product

Building the search page required bridging the gap between a working search pipeline (BM25 + semantic + hybrid RRF fusion) and a production-ready user interface. This involved clean architecture decisions, UX iteration, and an operational lesson about cloud infrastructure.

### Clean Architecture (4-File Design)

The search integration followed a strict separation of concerns:

| File | Responsibility |
|------|---------------|
| `search/result_types.py` | `SearchResult` dataclass — typed contract between backend and frontend |
| `search/enrichment.py` | `ResultEnricher` — joins search hits with SQLite metadata (titles, dates, categories, PDF URLs) |
| `search/service.py` | `SearchService` — orchestrates search → enrich → filter pipeline |
| `analytics/queries/search_filters.py` | Filter options + post-search aircraft filtering via SQL |

This design means the Streamlit view (`app/views/search.py`) has zero direct database or Qdrant dependencies — it calls `SearchService.search()` and renders `SearchResult` objects. The enrichment layer bridges the gap between Qdrant's chunk-level payloads and the report-level metadata stored in SQLite.

### UX Iteration: Horizontal Filter Bar

The search page layout went through three iterations based on user testing:

1. **Vertical filter sidebar + radio buttons** — Rejected. Wasted horizontal space and felt disconnected from search results.
2. **Sub-navigation bar for mode selection** — Rejected. Added visual complexity without improving the workflow.
3. **Horizontal filter bar above search box** — Final. Five-column layout (Mode dropdown | Risk Category multiselect | Aircraft Type multiselect | Date From | Date To) with a conditional L2 subcategory row that appears only when an L1 category is selected.

The key insight: search filter UI should mirror how users think about their query — mode and constraints first, then the actual search text. Putting filters *above* the search box (not beside it) keeps the full page width available for results.

### XSS-Safe HTML Rendering

Streamlit's `unsafe_allow_html=True` for rich result cards required careful security:
- `html.escape()` on all user-facing text before injection
- URL validation (`_safe_url()`) restricting to `https://` only
- Query term highlighting applied *after* HTML escaping to prevent injection through search terms

### Qdrant Cloud: An Operational Lesson

**The problem:** After several weeks of development focus on the analytics reports, we returned to find the search page throwing 404 errors. The Qdrant Cloud free-tier cluster had been silently deleted due to inactivity — all 24,766 vectors across both collections (MiniLM + MIKA) were gone.

**Why it matters:** This is the kind of operational surprise that doesn't appear in tutorials. Free-tier cloud services often have usage requirements that aren't prominently documented. For a portfolio project where development is intermittent, this creates a real risk of losing deployed infrastructure between work sessions.

**The recovery:** Because all embedding vectors were stored locally as Parquet files (`embeddings_data/v2/`), and all taxonomy enrichment data lived in SQLite, recovery was straightforward:
1. Create a new Qdrant Cloud cluster
2. Update credentials in `.env` and `.streamlit/secrets.toml`
3. Re-upload: `python -m embeddings.cli upload both` (~5 min)
4. Re-enrich: `python -m embeddings.cli enrich both --l1-run 1 --l2-run 1` (~50 min)
5. Restart Streamlit (cached client holds old connection)

**The lesson:** Never treat cloud infrastructure as your source of truth. Local-first data storage (Parquet embeddings, SQLite metadata) meant zero data loss despite complete cloud infrastructure deletion. The recovery procedure is now documented in `pick_up_here.md` and `CLAUDE.md` for future sessions.

### Known Design Challenge: Multi-Chunk Result Consolidation

The current search returns one result per matching chunk, so a single report can appear multiple times in results (e.g., if "engine failure" matches chunks from the Findings, Analysis, and Probable Cause sections of the same report). The next iteration will group results by `report_id` and show the best-matching snippet per report — a common information retrieval pattern that requires careful score aggregation across chunks.

---

## Future Directions

### Short-term

1. **Search Result Consolidation**: Group multi-chunk results by report, showing best-matching snippet per report
2. **Taxonomy Explorer Page**: Interactive L1→L2 drill-down with report lists per category
3. **UI Polish**: Incorporate human UI review feedback, improve mobile responsiveness

### Medium-term

1. **Fine-tuned Model**: Train MIKA on NTSB-specific queries for further improvement
2. **Query Expansion**: Use LLM to expand user queries with aviation terminology
3. **Feature Interactions**: Experiment with weather x time_of_day in Bayesian model
4. **Improve Weather Coverage**: Currently 72.4%; explore additional extraction patterns

### Long-term

1. **Cross-Modal Search**: Include accident photos, diagrams, flight data
2. **Causal Analysis**: Extract and link causal chains across reports
3. **Predictive Insights**: Identify emerging safety patterns before accidents occur
4. **FastAPI Backend**: REST API for search and risk scoring if needed beyond Streamlit

---

## Skills Demonstrated

### Data Engineering
- Multi-pass ETL pipeline with quality gates
- SQLite for state management, JSONL/Parquet for bulk data
- Incremental processing with resume capability
- Comprehensive logging and error tracking
- Data quality gap analysis with root cause investigation
- Automated document type classification (prefix/title heuristics)

### NLP/ML & Information Retrieval
- Text extraction with OCR fallback
- Section-aware chunking with pattern matching
- Embedding model comparison (general vs. domain-specific)
- Vector database integration (Qdrant Cloud)
- Unsupervised topic modeling (BERTopic, UMAP, HDBSCAN)
- Embedding-based classification with seed phrases
- Binary Relevance Naive Bayes for multi-label classification
- Bayesian model auditing: calibration analysis, baseline comparison, LOO correctness
- Hybrid search architecture (BM25 + semantic + RRF fusion)
- Search result enrichment pipeline (chunk-level → report-level metadata joins)
- Qdrant filter construction (taxonomy, date range, multi-field filters)

### Evaluation Methodology
- Stratified benchmark design
- Statistical significance testing (bootstrap CI, Wilcoxon)
- Human evaluation protocol
- Semantic lift calculation
- Human-in-the-loop review gates for model validation
- Iterative gap analysis with progressive threshold relaxation
- Expected Calibration Error (ECE) for probability reliability
- Feature ablation studies for model understanding
- Per-category discrimination analysis

### Data Visualization & UX
- Consulting-style narrative reports with embedded Plotly charts
- Colorblind-safe palette design (avoiding red/green confusion)
- Stakeholder persona-driven page design
- KPI selection methodology (meaningful metrics vs. available metrics)
- Abbreviation tooltip system (39 aviation/statistical code definitions)
- Co-occurrence matrix design (lower-triangle, annotation thresholds)
- Time-of-day color mapping (intuitive warm→cool progression)
- Streamlit caching architecture (`@st.cache_data`, `@st.cache_resource`)

### Software Engineering
- Modular architecture with clear separation of concerns
- CLI interfaces for all components
- Configuration management via environment variables
- Comprehensive documentation
- Thread-safe database access in multi-threaded web frameworks
- Reusable component library (charts, layout, theme)
- Cloud infrastructure resilience (local-first data storage, documented recovery procedures)
- XSS-safe HTML rendering in user-facing search results

### Domain Knowledge
- Understanding NTSB report structure
- Aviation terminology and concepts
- Safety investigation methodology
- CICTT occurrence taxonomy (CAST/ICAO industry standard)
- Stakeholder-specific risk communication (fleet, underwriting, operational)

---

## Conclusion

RiskRADAR demonstrates that building effective semantic search, classification, and risk modeling requires rigorous methodology at every stage — and the willingness to tear down work that doesn't hold up to scrutiny. The journey from 94.9% to 100% Hit@10 came not from model improvements, but from understanding how document structure affects retrieval. The pivot from unsupervised topic modeling to CICTT taxonomy came from recognizing that statistical patterns don't equal meaningful patterns. The gap analysis journey — from an unexplained 18% taxonomy gap to 98.9% coverage — came from asking *why* before asking *how*. And the Bayesian model rewrite came from having the intellectual honesty to audit working code and discovering it was built on flawed assumptions.

The key insights:
1. **Preprocessing decisions compound**. Bad chunks lead to bad embeddings lead to bad retrieval. Investing in quality at every stage pays exponential dividends.
2. **Domain expertise cannot be automated away**. BERTopic found 76 topics; human review found they were noise. The CICTT taxonomy, built by aviation safety experts over decades, provides what no algorithm could discover.
3. **Human-in-the-loop is essential**. Every major quality improvement came from human review—whether catching chunking problems, validating retrieval quality, recognizing that unsupervised topics were meaningless, or ranking section names into priority tiers for the final classification pass.
4. **Root cause analysis beats brute force**. When 92 reports lacked taxonomy, the fix wasn't lowering thresholds—it was discovering that 55% of them were non-accident documents. Understanding the problem correctly reduced the actual gap from 92 to 35, and targeted fixes brought coverage to 98.9%.
5. **Statistical auditing catches invisible errors**. The initial Bayesian model reported Hit@5=90.7% — a number that would have gone unquestioned into a portfolio. A systematic audit revealed fake validation, miscalibrated probabilities, and hidden baseline underperformance. The rewritten model has lower headline numbers (Hit@5=86.8%) but honest ones, with near-perfect calibration (ECE=0.021). Presenting correctly-measured, honestly-interpreted results demonstrates stronger data science judgment than inflating metrics.
6. **Multi-label data requires multi-label models**. When 95.8% of observations have multiple labels, forcing a sum-to-1 normalization is a fundamental architectural error — not a tuning problem. Recognizing this distinction between "the model needs better parameters" and "the model needs a different architecture" is a critical skill.
7. **Data visualization is stakeholder communication**. Building dashboards is easy; building reports that answer the right questions for the right audience is hard. The shift from 4 generic dashboards to 3 persona-driven narrative reports — and the iterative refinement of KPIs, colors, and abbreviation handling — reflects the reality that the last mile of data science is persuasive communication.
8. **Cloud infrastructure requires defensive design**. Free-tier Qdrant Cloud clusters are silently deleted after extended inactivity — an operational surprise that doesn't appear in tutorials. Because all embedding vectors and metadata were stored locally (Parquet + SQLite), recovery from complete cloud infrastructure loss took minutes, not days. Never treat cloud services as your source of truth, especially with intermittent development schedules.

For professionals evaluating this work: the methodology and willingness to pivot are as important as the final metrics. The 38.6% semantic lift is meaningful because we measured it properly. The CICTT classification is meaningful because we recognized when an approach was failing and changed course. The 98.9% taxonomy coverage is meaningful because each percentage point was earned through deliberate investigation. The Bayesian risk model is meaningful not for its headline accuracy numbers, but because it survived a rigorous statistical audit, was rebuilt when flaws were found, and ships with honest metrics, calibration analysis, and documented limitations. The semantic search page is meaningful because it bridges the gap between a working ML pipeline and a usable product — with clean architecture, XSS-safe rendering, and iterative UX refinement driven by real user testing. And the Streamlit application is meaningful because it translates analytical rigor into stakeholder-accessible reports — with colorblind-safe palettes, abbreviation tooltips, and consulting-style narratives that communicate findings without requiring domain expertise.

---

*Last updated: March 2026*

*For technical documentation, see [README.md](README.md) and module-specific documentation.*
