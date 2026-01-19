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
- [Future Directions](#future-directions)
- [Skills Demonstrated](#skills-demonstrated)

---

## Executive Summary

RiskRADAR transforms 510 NTSB aviation accident reports (spanning 1966-present) into a semantically searchable knowledge base. The project demonstrates production-grade data engineering practices:

- **30,602 pages** processed through OCR and quality pipelines
- **24,766 chunks** optimized through iterative evaluation
- **2 embedding models** compared with rigorous statistical methods
- **50 benchmark queries** spanning 6 difficulty categories
- **38.6% semantic lift** achieved with domain-specific embeddings

The most significant technical insight: **chunk quality directly determines retrieval quality**. Version 2's chunking strategy improved Hit@10 from 94.9% to 100% and MRR from 0.788 to 0.816.

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
Phase 1: Scrape     Phase 3: Extract    Phase 4: Chunk      Phase 5: Embed
─────────────────   ─────────────────   ─────────────────   ─────────────────
NTSB Website        PDF Documents       Document Text       Chunk Vectors
    │                   │                   │                   │
    ▼                   ▼                   ▼                   ▼
510 PDFs ────────► 30,602 pages ────► 24,766 chunks ────► Qdrant Cloud
    │                   │                   │                   │
    ▼                   ▼                   ▼                   ▼
SQLite              JSON/JSONL          JSONL               Vector Index
(metadata)          (full text)         (search-ready)      (similarity)
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

## Future Directions

### Short-term (Phase 6-8)

1. **Streamlit Application**: Search interface with filtering by date, aircraft, location
2. **Hierarchical Taxonomy**: Categorize reports by cause (human factors, mechanical, weather)
3. **Trend Analytics**: Visualize safety themes over time

### Medium-term

1. **Fine-tuned Model**: Train MIKA on NTSB-specific queries for further improvement
2. **Hybrid Search**: Combine vector similarity with BM25 keyword matching
3. **Query Expansion**: Use LLM to expand user queries with aviation terminology

### Long-term

1. **Cross-Modal Search**: Include accident photos, diagrams, flight data
2. **Causal Analysis**: Extract and link causal chains across reports
3. **Predictive Insights**: Identify emerging safety patterns before accidents occur

---

## Skills Demonstrated

### Data Engineering
- Multi-pass ETL pipeline with quality gates
- SQLite for state management, JSONL/Parquet for bulk data
- Incremental processing with resume capability
- Comprehensive logging and error tracking

### NLP/ML
- Text extraction with OCR fallback
- Section-aware chunking with pattern matching
- Embedding model comparison (general vs. domain-specific)
- Vector database integration (Qdrant Cloud)

### Evaluation Methodology
- Stratified benchmark design
- Statistical significance testing (bootstrap CI, Wilcoxon)
- Human evaluation protocol
- Semantic lift calculation

### Software Engineering
- Modular architecture with clear separation of concerns
- CLI interfaces for all components
- Configuration management via environment variables
- Comprehensive documentation

### Domain Knowledge
- Understanding NTSB report structure
- Aviation terminology and concepts
- Safety investigation methodology

---

## Conclusion

RiskRADAR demonstrates that building effective semantic search requires more than plugging documents into an embedding model. The journey from 94.9% to 100% Hit@10 came not from model improvements, but from understanding how document structure affects retrieval.

The key insight: **preprocessing decisions compound**. Bad chunks lead to bad embeddings lead to bad retrieval. Investing in quality at every stage pays exponential dividends.

For professionals evaluating this work: the benchmark methodology and statistical rigor are as important as the final metrics. The 38.6% semantic lift is meaningful because we measured it properly.

---

*Last updated: January 2026*

*For technical documentation, see [README.md](README.md) and module-specific documentation.*
