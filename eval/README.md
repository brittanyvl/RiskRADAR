# RiskRADAR Benchmark Framework

> **IMPORTANT: Read Before Using Results**
>
> This benchmark has two evaluation modes:
> 1. **Automated evaluation** - SQL-based ground truth. Works well for incident lookup, section queries, and aircraft queries. Results are objective and reproducible.
> 2. **Human evaluation** - Required for conceptual and comparative queries where keyword matching cannot capture semantic relevance. Without human review, automated metrics for these categories are **LOWER BOUNDS** on true performance.
>
> **Statistical note:** Do not rely on t-test p-values. Use bootstrap confidence intervals and win/loss/tie counts for inference. See [Statistical Methodology](#statistical-methodology) for details.

---

## Table of Contents

1. [Glossary: Key Terms Explained](#glossary-key-terms-explained)
2. [What This Is](#what-this-is)
3. [Why We're Doing This](#why-were-doing-this)
   - [The Core Question](#the-core-question)
   - [What "Better" Means](#what-better-means)
4. [The 50 Test Queries](#the-50-test-queries)
   - [Category 1: Incident Lookup (10 queries)](#category-1-incident-lookup-10-queries)
   - [Category 2: Conceptual Queries (12 queries)](#category-2-conceptual-queries-12-queries)
   - [Category 3: Section-Targeted Queries (10 queries)](#category-3-section-targeted-queries-10-queries)
   - [Category 4: Comparative/Analytical Queries (8 queries)](#category-4-comparativeanalytical-queries-8-queries)
   - [Category 5: Aircraft-Type Queries (6 queries)](#category-5-aircraft-type-queries-6-queries)
   - [Category 6: Flight Phase Queries (4 queries)](#category-6-flight-phase-queries-4-queries)
5. [How We Measure Success](#how-we-measure-success)
   - [Primary Metrics](#primary-metrics)
   - [Example: Understanding MRR](#example-understanding-mrr)
   - [Performance Metrics](#performance-metrics)
6. [How Ground Truth Works](#how-ground-truth-works)
7. [Statistical Methodology](#statistical-methodology)
   - [Why Standard Tests Are Problematic Here](#why-standard-tests-are-problematic-here)
   - [Recommended Approach](#recommended-approach-in-order-of-reliability)
   - [Interpreting Results Correctly](#interpreting-results-correctly)
   - [Statistical Power Consideration](#statistical-power-consideration)
   - [Multiple Comparisons](#multiple-comparisons)
8. [Critical Limitation: Semantic Evaluation Requires Human Review](#critical-limitation-semantic-evaluation-requires-human-review)
   - [The Problem with Keyword-Based Ground Truth](#the-problem-with-keyword-based-ground-truth)
   - [What This Means for Our Benchmark](#what-this-means-for-our-benchmark)
   - [Human-in-the-Loop Evaluation Protocol](#human-in-the-loop-evaluation-protocol)
   - [Recommended Evaluation Workflow](#recommended-evaluation-workflow)
   - [Why This Matters](#why-this-matters)
9. [Running the Benchmark](#running-the-benchmark)
   - [Prerequisites](#prerequisites)
   - [Complete Evaluation Workflow](#complete-evaluation-workflow)
   - [CLI Commands Reference](#cli-commands-reference)
   - [Quick Commands](#quick-commands)
   - [Expected Time Investment](#expected-time-investment)
10. [Output Files](#output-files)
    - [Results Directory Structure](#results-directory-structure)
    - [JSON Output Schema](#json-output-schema)
    - [Parquet Schema (for Streamlit)](#parquet-schema-for-streamlit)
11. [Expected Results](#expected-results)
    - [Where MIKA Should Excel](#where-mika-should-excel)
    - [Where MiniLM Should Be Competitive](#where-minilm-should-be-competitive)
    - [Decision Framework](#decision-framework)
12. [Streamlit Integration](#streamlit-integration)
    - [Available Parquet Files](#available-parquet-files)
    - [Example Streamlit Code](#example-streamlit-code)
13. [Files Reference](#files-reference)
    - [Source Files](#source-files)
    - [Generated Files (Automated Benchmark)](#generated-files-automated-benchmark)
    - [Generated Files (Human Review Export)](#generated-files-human-review-export)
    - [Human Review File Structure](#human-review-file-structure)
    - [Generated Files (After Human Review Import)](#generated-files-after-human-review-import)
    - [Final Report Files](#final-report-files)
    - [Parquet Schema Reference](#parquet-schema-reference)
14. [Troubleshooting](#troubleshooting)
15. [Summary: Evaluation Approach by Category](#summary-evaluation-approach-by-category)

---

## Glossary: Key Terms Explained

New to information retrieval or machine learning? Here's a plain-English guide to the terminology used in this document.

### Core Concepts

| Term | Plain English Explanation |
|------|---------------------------|
| **Embedding** | A way to convert text into numbers (a list of decimal values called a "vector"). Similar texts get similar numbers, so we can find related content by comparing numbers instead of matching exact words. |
| **Embedding Model** | The AI that converts text to numbers. Different models produce different quality embeddings. We're comparing two: MiniLM (general-purpose) and MIKA (aviation-specialized). |
| **Vector Database** | A specialized database (we use Qdrant) that stores embeddings and can quickly find "similar" vectors. When you search, your query becomes a vector and the database finds the closest matches. |
| **Chunk** | A piece of a document. We split our 510 PDF reports into 28,321 smaller pieces (chunks) because embedding models work better on shorter text. Each chunk is ~500-700 words. |
| **Semantic Search** | Finding results based on *meaning*, not just keyword matching. "Pilot was exhausted" should match a search for "fatigue" even though the word "fatigue" doesn't appear. |
| **Ground Truth** | The "correct answers" we know ahead of time. For each test query, we define which chunks SHOULD be returned so we can measure if the model found them. |

### Retrieval Metrics (How We Score Results)

| Term | Plain English Explanation |
|------|---------------------------|
| **MRR (Mean Reciprocal Rank)** | How quickly do we find the first good result? If the first result is relevant, MRR=1.0. If the second is first relevant, MRR=0.5. If fifth, MRR=0.2. Higher is better. |
| **Hit@K** | Did we find ANY relevant result in the top K? Hit@10 asks "Is there at least one good result in the first 10?" It's a yes/no measure. |
| **Precision@K** | What fraction of the top K results are relevant? If 3 of your top 5 results are good, Precision@5 = 0.6 (60%). Measures result quality. |
| **Recall@K** | What fraction of ALL relevant content did we find in top K? If there are 20 relevant chunks total and we found 10 in our top results, Recall = 0.5 (50%). Measures completeness. |
| **nDCG (Normalized Discounted Cumulative Gain)** | A fancy ranking score that rewards putting the best results first. Finding a relevant result at position 1 is worth more than finding it at position 10. Ranges 0-1, higher is better. |
| **Section Accuracy** | For queries targeting specific report sections (like "probable cause"), what fraction of results came from the correct section? Uses flexible matching: "(b) PROBABLE CAUSE" matches "PROBABLE CAUSE". |
| **Latency** | How long the search takes, measured in milliseconds. Lower is better. |

### Statistical Terms

| Term | Plain English Explanation |
|------|---------------------------|
| **Bootstrap Confidence Interval** | A way to estimate how reliable our measurements are. We resample our data 1000 times and see how much the results vary. A "95% CI of [0.02, 0.12]" means we're 95% confident the true difference is between 0.02 and 0.12. |
| **Confidence Interval (CI)** | A range that likely contains the true value. If the CI for "MIKA minus MiniLM" is [0.02, 0.12], MIKA is reliably better. If it's [-0.05, 0.08], we can't be sure which is better. |
| **Win/Loss/Tie** | Simply counting: on how many queries did Model A beat Model B? Easy to understand and robust to weird data. |
| **Wilcoxon Signed-Rank Test** | A statistical test that compares paired measurements without assuming the data follows a bell curve. More appropriate for our bounded MRR scores than a t-test. |
| **Statistical Significance** | Can we trust that a difference is real, not just random chance? We use bootstrap CIs: if the interval excludes zero, the difference is likely real. |
| **Effect Size** | How BIG is the difference in practical terms? A "statistically significant" difference of 0.001 MRR doesn't matter. A difference of 0.10 MRR is meaningful. |

### Human Evaluation Terms

| Term | Plain English Explanation |
|------|---------------------------|
| **Keyword Match** | A result that matches our SQL ground truth - it contains the expected keywords. This is what traditional search would find. |
| **Semantic Match** | A result that's relevant but DOESN'T match keywords. The model found something good that keyword search would miss. This is the value of semantic search! |
| **False Positive** | A result that looks related but isn't actually relevant to the query. The model made a mistake. |
| **Semantic Lift** | The percentage of results that are semantic matches (relevant but no keyword overlap). This measures how much value the embedding model adds over simple keyword search. |
| **Semantic Precision** | (Keyword Matches + Semantic Matches) / Total Results. The overall "correctness" rate including both types of good results. |

### File Formats

| Term | Plain English Explanation |
|------|---------------------------|
| **JSON** | A text file format for structured data. Human-readable, used for detailed results and configuration. |
| **Parquet** | A compressed binary file format optimized for data analysis. Loads fast into pandas/Streamlit. Used for visualization-ready data. |
| **YAML** | A text file format that's easy for humans to read and write. Used for query definitions and human review forms. |

---

## What This Is

This is the evaluation system for Phase 5 of RiskRADAR. We have 28,321 text chunks from 510 NTSB aviation accident reports, and we need to make them searchable. To do that, we convert text into numerical vectors (embeddings) and store them in a vector database (Qdrant). When a user searches, we convert their query into a vector and find the most similar chunks.

**The problem:** There are many embedding models available. Which one should we use?

**The solution:** Run both models on the same 50 carefully designed test queries and measure which one returns better results.

---

## Why We're Doing This

### The Core Question

We have two embedding model options:

| Model | Dimensions | Background |
|-------|------------|------------|
| **MiniLM** | 384 | General-purpose model trained on web text. Fast and small. |
| **MIKA** | 768 | Specialized model from NASA trained on aerospace/aviation documents. Larger and slower. |

**Hypothesis:** MIKA should perform better on aviation-specific queries because it was trained on similar content. But is it worth the extra size and computational cost?

### What "Better" Means

When someone searches "hydraulic system failure," we want the search results to:
1. **Be relevant** - Return chunks that actually discuss hydraulic failures
2. **Be ranked correctly** - The most relevant chunks should appear first
3. **Be complete** - Find all the relevant chunks, not just some
4. **Be fast** - Return results quickly

We measure all of these with standard information retrieval metrics.

---

## The 50 Test Queries

We designed 50 test queries across 6 categories. Each query tests a different aspect of the search system.

### Category 1: Incident Lookup (10 queries)

**Purpose:** Test if the system can find specific, known accidents when users search for them by name.

**Why this matters:** Users often know about famous accidents and want to read the official report. The system should easily find these.

**Difficulty:** Easy - these queries have high keyword overlap with the target documents.

| ID | Query | Target Report | What We're Testing |
|----|-------|---------------|-------------------|
| INC-001 | Alaska Airlines flight 261 horizontal stabilizer jackscrew failure | AAR0201.pdf | Famous 2000 accident - jackscrew wear caused crash |
| INC-002 | Aloha Airlines flight 243 explosive decompression fuselage | AAR8903.pdf | 1988 structural failure - roof tore off mid-flight |
| INC-003 | USAir flight 427 rudder power control unit malfunction Pittsburgh | AAR9901.pdf | 1994 crash that led to 737 rudder redesign |
| INC-004 | ValuJet flight 592 cargo fire oxygen generators Everglades | AAR9706.pdf | 1996 cargo fire - led to major safety regulations |
| INC-005 | American Airlines flight 587 vertical stabilizer separation Queens | AAR0404.pdf | 2001 A300 crash - composite structure failure |
| INC-006 | Colgan Air flight 3407 stall stick pusher Buffalo | AAR1001.pdf | 2009 crash - led to pilot training reform |
| INC-007 | Asiana Airlines flight 214 San Francisco visual approach automation | AAR1401.pdf | 2013 777 crash - automation confusion |
| INC-008 | TWA flight 800 center fuel tank explosion Long Island | AAR0003.pdf | 1996 747 explosion - most expensive NTSB investigation |
| INC-009 | United Airlines flight 232 Sioux City DC-10 hydraulic failure | AAR9006.pdf | 1989 crash - heroic emergency landing |
| INC-010 | Air France flight 447 pitot tube icing Atlantic stall | N/A | **Negative test** - this was a French investigation, not NTSB |

### Category 2: Conceptual Queries (12 queries)

**Purpose:** Test if the system understands aviation concepts, not just keywords.

**Why this matters:** Users often search for concepts like "ice on wings" when reports might say "leading edge contamination." The model needs to understand these are related.

**Difficulty:** Medium to Hard - requires semantic understanding beyond keyword matching.

**How we validate these:** Unlike incident lookups (which target ONE specific report), conceptual queries should find content across MANY reports. For each query, we define:
1. **Verification SQL** - Finds all chunks in our corpus that discuss this concept
2. **Expected Report IDs** - Specific reports we KNOW discuss this topic (used as anchors)
3. **Minimum Relevant Reports** - How many different reports should have relevant content
4. **Relevance Signals** - Keywords that indicate a chunk is truly relevant

A result is considered relevant if it:
- Comes from one of the expected_report_ids, OR
- Matches the verification_sql pattern, OR
- Contains multiple relevance_signals

| ID | Query | Validation Details |
|----|-------|--------------------|
| CON-001 | uncommanded flight control surface movement | **SQL:** `chunk_text LIKE '%uncommanded%' AND (LIKE '%rudder%' OR '%aileron%' OR '%elevator%')` <br> **Expected Reports:** AAR0101, AAR9901, AAR0404 <br> **Min Reports:** 3 <br> **Signals:** uncommanded, hardover, runaway, control surface |
| CON-002 | ice contamination on wing leading edge aerodynamic degradation | **SQL:** `chunk_text LIKE '%icing%' AND LIKE '%wing%'` <br> **Expected Reports:** AAR1001, AAB0001, AAB0202, AAB0205 <br> **Min Reports:** 5 <br> **Signals:** icing, deicing, leading edge, lift, stall |
| CON-003 | controlled flight into terrain warning system GPWS TAWS | **SQL:** `chunk_text LIKE '%gpws%' OR '%ground proximity%' OR '%terrain%warning%'` <br> **Expected Reports:** AAR0001, AAR0701 <br> **Min Reports:** 10 <br> **Signals:** gpws, taws, terrain, cfit, warning |
| CON-004 | crew resource management breakdown communication failure | **SQL:** `chunk_text LIKE '%crew resource management%' OR '%crm%'` <br> **Expected Reports:** AAB0203, AAB0401, AAB0603 <br> **Min Reports:** 8 <br> **Signals:** crm, communication, coordination, cockpit, assertiveness |
| CON-005 | pilot fatigue circadian rhythm duty time regulations | **SQL:** `chunk_text LIKE '%fatigue%' AND LIKE '%pilot%'` <br> **Expected Reports:** AAB0102, AAB0201, AAB0202, AAR1001 <br> **Min Reports:** 10 <br> **Signals:** fatigue, duty time, rest, sleep, circadian |
| CON-006 | metal fatigue crack propagation structural inspection | **SQL:** `chunk_text LIKE '%fatigue%crack%' OR '%crack%propagation%'` <br> **Expected Reports:** AAR8903, AAR0201 <br> **Min Reports:** 15 <br> **Signals:** fatigue, crack, inspection, propagation, fracture |
| CON-007 | automation mode confusion glass cockpit display | **SQL:** `chunk_text LIKE '%automation%' AND (LIKE '%confusion%' OR '%mode%')` <br> **Expected Reports:** AAR1401 <br> **Min Reports:** 5 <br> **Signals:** automation, autoflight, mode, display, awareness |
| CON-008 | turbine engine uncontained failure blade separation | **SQL:** `chunk_text LIKE '%uncontained%' OR (LIKE '%turbine%' AND '%blade%fail%')` <br> **Expected Reports:** AAR9006 <br> **Min Reports:** 8 <br> **Signals:** uncontained, blade, turbine, disk, fan |
| CON-009 | wake turbulence vortex encounter separation standards | **SQL:** `chunk_text LIKE '%wake%turbulence%' OR '%vortex%'` <br> **Expected Reports:** AAR0404 <br> **Min Reports:** 10 <br> **Signals:** wake, vortex, turbulence, separation, heavy |
| CON-010 | spatial disorientation vestibular illusion instrument flight | **SQL:** `chunk_text LIKE '%spatial disorientation%' OR '%vestibular%'` <br> **Expected Reports:** None specified <br> **Min Reports:** 3 <br> **Signals:** disorientation, vestibular, illusion, instrument, attitude |
| CON-011 | fuel system contamination water ice crystals | **SQL:** `chunk_text LIKE '%fuel%' AND (LIKE '%contamination%' OR '%water%ice%')` <br> **Expected Reports:** None specified <br> **Min Reports:** 5 <br> **Signals:** fuel, contamination, water, ice, filter |
| CON-012 | loss of control inflight upset recovery training | **SQL:** `chunk_text LIKE '%loss of control%' OR '%upset%recovery%'` <br> **Expected Reports:** AAR1001, AAR1401 <br> **Min Reports:** 20 <br> **Signals:** loss of control, upset, recovery, stall, unusual attitude |

**Example Validation for CON-001:**
```sql
-- Find ground truth chunks for "uncommanded flight control surface movement"
SELECT chunk_id, report_id, section_name, LEFT(chunk_text, 100) as preview
FROM chunks
WHERE lower(chunk_text) LIKE '%uncommanded%'
  AND (lower(chunk_text) LIKE '%rudder%'
       OR lower(chunk_text) LIKE '%aileron%'
       OR lower(chunk_text) LIKE '%elevator%');

-- This might return 25 chunks across 8 reports
-- If search returns chunks from 5 of those 8 reports in top 10, that's good recall
```

### Category 3: Section-Targeted Queries (10 queries)

**Purpose:** Test if the system returns results from the correct sections of reports.

**Why this matters:** NTSB reports have a standard structure. Someone searching for "probable cause" should get chunks from the Probable Cause section, not random mentions of the word "cause."

**Difficulty:** Medium - tests understanding of document structure.

**How we validate these:** Unlike other categories, we don't just check if the content is relevant - we check if results come from the **correct section**. We use a special metric called **Section Accuracy**:

```
Section Accuracy = (chunks from expected sections) / (total chunks returned)
```

For example, if we search "probable cause" and get 10 results:
- 6 are from "PROBABLE CAUSE" section
- 2 are from "CONCLUSIONS" section
- 2 are from "ANALYSIS" section

Section Accuracy = 8/10 = **0.80** (PROBABLE CAUSE and CONCLUSIONS are both valid)

**Universal Valid Sections:** Some sections are valid for ANY section query because they summarize content from multiple sections:
- SYNOPSIS
- EXECUTIVE SUMMARY
- SUMMARY
- ABSTRACT
- BRIEF
- OVERVIEW
- APPENDIX / APPENDICES

For example, a SYNOPSIS might mention the probable cause, so finding probable cause content in SYNOPSIS is valid - it's not penalized.

Each query defines:
1. **Primary Sections** - The specific sections that should contain the answer
2. **Universal Sections** - Automatically included (summary sections valid for all queries)
3. **Section Chunk Count** - How many chunks exist in those sections (total pool)
4. **Minimum Section Accuracy** - Threshold for acceptable performance (typically 40-60%)

| ID | Query | Validation Details |
|----|-------|--------------------|
| SEC-001 | what was determined to be the probable cause | **Target Sections:** PROBABLE CAUSE, CONCLUSIONS <br> **Total Chunks in Sections:** 557 <br> **Min Accuracy:** 50% <br> **Why:** The definitive cause determination |
| SEC-002 | safety recommendations issued to the FAA | **Target Sections:** RECOMMENDATIONS, SAFETY RECOMMENDATIONS <br> **Total Chunks:** 1,621 <br> **Min Accuracy:** 60% <br> **Why:** Actionable safety improvements |
| SEC-003 | weather conditions temperature visibility at accident time | **Target Sections:** METEOROLOGICAL INFORMATION <br> **Total Chunks:** 600 <br> **Min Accuracy:** 50% <br> **Why:** Standard section in every report |
| SEC-004 | wreckage distribution debris field impact marks | **Target Sections:** WRECKAGE AND IMPACT INFORMATION <br> **Total Chunks:** 700 <br> **Min Accuracy:** 50% <br> **Why:** Physical evidence documentation |
| SEC-005 | flight data recorder parameters cockpit voice recorder transcript | **Target Sections:** FLIGHT RECORDERS, TESTS AND RESEARCH <br> **Total Chunks:** 1,513 <br> **Min Accuracy:** 40% <br> **Why:** FDR/CVR data is critical evidence |
| SEC-006 | pilot certificate ratings flight experience hours | **Target Sections:** PERSONNEL INFORMATION, PILOT INFORMATION, FLIGHT CREW INFORMATION <br> **Total Chunks:** 500 <br> **Min Accuracy:** 40% <br> **Why:** Crew qualifications |
| SEC-007 | aircraft maintenance records inspection history airworthiness | **Target Sections:** AIRCRAFT INFORMATION, MAINTENANCE RECORDS <br> **Total Chunks:** 1,300 <br> **Min Accuracy:** 40% <br> **Why:** Mechanical history |
| SEC-008 | fire damage burn patterns post-crash fire | **Target Sections:** FIRE, WRECKAGE AND IMPACT INFORMATION <br> **Total Chunks:** 616 <br> **Min Accuracy:** 40% <br> **Why:** Fire-specific findings |
| SEC-009 | survival factors injuries occupant protection evacuation | **Target Sections:** SURVIVAL ASPECTS, MEDICAL AND PATHOLOGICAL INFORMATION <br> **Total Chunks:** 829 <br> **Min Accuracy:** 40% <br> **Why:** Survivability analysis |
| SEC-010 | tests conducted simulations research findings | **Target Sections:** TESTS AND RESEARCH, ADDITIONAL INFORMATION <br> **Total Chunks:** 2,815 <br> **Min Accuracy:** 40% <br> **Why:** Post-accident testing |

**Example: Why Section Accuracy Matters**

Query: "what was determined to be the probable cause"

**Bad Result (Section Accuracy = 0.20):**
1. Chunk from SYNOPSIS mentioning "cause" - wrong section
2. Chunk from ANALYSIS discussing causes - wrong section
3. Chunk from PROBABLE CAUSE - correct!
4. Chunk from HISTORY OF FLIGHT - wrong section
5. Chunk from RECOMMENDATIONS mentioning "cause" - wrong section

**Good Result (Section Accuracy = 0.80):**
1. Chunk from PROBABLE CAUSE - correct!
2. Chunk from PROBABLE CAUSE - correct!
3. Chunk from CONCLUSIONS - correct!
4. Chunk from PROBABLE CAUSE - correct!
5. Chunk from ANALYSIS - wrong section

### Category 4: Comparative/Analytical Queries (8 queries)

**Purpose:** Test if the system can find patterns across multiple reports.

**Why this matters:** Safety researchers want to find trends like "accidents caused by maintenance failures." This requires understanding that different reports discuss similar themes in different words.

**Difficulty:** Hard - pure semantic understanding required, minimal keyword overlap.

**How we validate these:** These are the hardest queries because they require the model to understand THEMES, not just find keywords. We validate by:
1. **Verification SQL** - Often combines content matching with section filtering (e.g., "maintenance" in "probable cause" section)
2. **Minimum Relevant Reports** - These queries should return content from MANY reports (20-50)
3. **Relevance Signals** - Multiple related terms that indicate the chunk discusses this theme

The key difference from conceptual queries: analytical queries often filter by SECTION (e.g., "probable cause") because we want chunks where this theme was identified as a CAUSE, not just mentioned.

| ID | Query | Validation Details |
|----|-------|--------------------|
| CMP-001 | accidents caused by inadequate maintenance procedures | **SQL:** `chunk_text LIKE '%maintenance%' AND section_name = 'probable cause'` <br> **Min Reports:** 20 <br> **Signals:** maintenance, inspection, oversight, procedure <br> **Why section matters:** We want chunks where maintenance was the CAUSE, not just discussed |
| CMP-002 | accidents where pilots failed to respond to warnings | **SQL:** `chunk_text LIKE '%warning%' AND LIKE '%fail%'` <br> **Min Reports:** 15 <br> **Signals:** warning, ignored, response, alert <br> **Challenge:** "Failed to respond" is semantic - no exact keyword match |
| CMP-003 | recurring design defects in Boeing aircraft | **SQL:** `chunk_text LIKE '%boeing%' AND LIKE '%design%'` <br> **Min Reports:** 30 <br> **Signals:** boeing, design, defect, flaw, modification <br> **Tests:** Can model find design issues across different Boeing models? |
| CMP-004 | accidents during night visual flight rules operations | **SQL:** `chunk_text LIKE '%night%' AND LIKE '%vfr%'` <br> **Min Reports:** 10 <br> **Signals:** night, vfr, visual, darkness <br> **Challenge:** Must understand "night VFR" as a risky flight condition |
| CMP-005 | regulatory failures that contributed to accidents | **SQL:** `chunk_text LIKE '%faa%' AND LIKE '%oversight%'` <br> **Min Reports:** 25 <br> **Signals:** faa, regulatory, oversight, certification <br> **Tests:** Can model find systemic issues, not just individual errors? |
| CMP-006 | training deficiencies identified after accidents | **SQL:** `chunk_text LIKE '%training%' AND section_name IN ('analysis', 'probable cause')` <br> **Min Reports:** 30 <br> **Signals:** training, proficiency, inadequate, deficient <br> **Why section matters:** Training as a FINDING, not general discussion |
| CMP-007 | accidents involving regional airlines commuter operations | **SQL:** `chunk_text LIKE '%regional%' OR LIKE '%commuter%'` <br> **Min Reports:** 40 <br> **Signals:** regional, commuter, part 135, scheduled <br> **Tests:** Broad category - should find many reports |
| CMP-008 | post-accident safety improvements and industry changes | **SQL:** `section_name = 'recommendations'` <br> **Min Reports:** 50 <br> **Signals:** recommendation, improvement, require, mandate <br> **Unique:** This query should pull from RECOMMENDATIONS sections across all reports |

**Example Validation for CMP-001:**
```sql
-- Find chunks where maintenance was identified as a CAUSE
SELECT chunk_id, report_id, section_name, LEFT(chunk_text, 100) as preview
FROM chunks
WHERE lower(chunk_text) LIKE '%maintenance%'
  AND lower(section_name) = 'probable cause';

-- This might return 45 chunks across 22 reports
-- A good model should surface these even when query says "inadequate procedures"
-- instead of exact phrase "maintenance"
```

**Why these are "hard":** The query "regulatory failures that contributed to accidents" has almost no keyword overlap with a chunk that says "The FAA's oversight of the operator's maintenance program was inadequate." A good semantic model understands these mean the same thing.

### Category 5: Aircraft-Type Queries (6 queries)

**Purpose:** Test if the system can filter by aircraft type effectively.

**Why this matters:** Users often want to find all accidents involving a specific aircraft model. The model should understand that "737" and "Boeing 737-800" refer to the same family.

**Difficulty:** Medium - lexical cues available but semantic understanding helps.

**How we validate these:** We know approximately how many reports mention each aircraft type. We validate by:
1. **Verification SQL** - Pattern match for aircraft type mentions
2. **Expected Report Count** - Approximate number of reports that should match
3. **Relevance Signals** - Related terms (model variants, manufacturer names, etc.)

| ID | Query | Validation Details |
|----|-------|--------------------|
| ACF-001 | Boeing 737 rudder system problems and accidents | **SQL:** `chunk_text LIKE '%boeing 737%' AND LIKE '%rudder%'` <br> **Expected Reports:** ~76 Boeing 737 reports in corpus <br> **Signals:** 737, rudder, pcu, yaw damper <br> **Challenge:** Should find 737-200, 737-800, etc. as same family |
| ACF-002 | DC-10 cargo door and structural issues | **SQL:** `chunk_text LIKE '%dc-10%' AND LIKE '%cargo%door%'` <br> **Expected Reports:** ~65 DC-10 reports <br> **Signals:** dc-10, cargo door, latch, structural <br> **Note:** Famous design flaw - multiple accidents |
| ACF-003 | Cessna small aircraft accidents general aviation | **SQL:** `chunk_text LIKE '%cessna%'` <br> **Expected Reports:** ~147 Cessna reports <br> **Signals:** cessna, general aviation, single engine, private <br> **Tests:** Largest aircraft category - should find many |
| ACF-004 | Boeing 747 jumbo jet accidents and incidents | **SQL:** `chunk_text LIKE '%boeing 747%' OR '%747%jumbo%'` <br> **Expected Reports:** ~59 Boeing 747 reports <br> **Signals:** 747, jumbo, wide body, four engine <br> **Challenge:** "jumbo jet" is colloquial, not in reports |
| ACF-005 | turboprop aircraft propeller related accidents | **SQL:** `chunk_text LIKE '%turboprop%' OR (LIKE '%propeller%' AND '%turb%')` <br> **Expected Reports:** ~100 turboprop reports <br> **Signals:** turboprop, propeller, prop, pt6 <br> **Tests:** Can model understand turboprop as aircraft category? |
| ACF-006 | helicopter rotor system failures and accidents | **SQL:** `chunk_text LIKE '%helicopter%' AND LIKE '%rotor%'` <br> **Expected Reports:** ~50 helicopter reports <br> **Signals:** helicopter, rotor, blade, rotorcraft <br> **Tests:** Different aircraft category entirely |

### Category 6: Flight Phase Queries (4 queries)

**Purpose:** Test if the system understands flight phases.

**Why this matters:** Most accidents happen during specific flight phases (takeoff, approach, landing). Users researching approach accidents shouldn't get results about cruise-phase incidents.

**Difficulty:** Medium - requires understanding flight phase terminology.

**How we validate these:** NTSB reports typically identify the phase of flight when the accident occurred. We validate by:
1. **Verification SQL** - Pattern match for phase-related phrases
2. **Expected Report Count** - Approximate number of reports for each phase
3. **Relevance Signals** - Phase-specific terminology

| ID | Query | Validation Details |
|----|-------|--------------------|
| PHS-001 | accidents during takeoff roll and initial climb | **SQL:** `chunk_text LIKE '%during takeoff%' OR '%takeoff roll%'` <br> **Expected Reports:** ~153 takeoff-related reports <br> **Signals:** takeoff, v1, rotate, initial climb, runway <br> **Note:** High-risk phase - many accidents occur here |
| PHS-002 | approach and landing accidents runway environment | **SQL:** `chunk_text LIKE '%during approach%' OR '%during landing%'` <br> **Expected Reports:** ~128 approach/landing reports <br> **Signals:** approach, landing, runway, touchdown, go-around <br> **Note:** Most common accident phase |
| PHS-003 | in-flight emergencies during cruise altitude | **SQL:** `chunk_text LIKE '%during cruise%' OR '%cruise altitude%'` <br> **Expected Reports:** ~41 cruise-related reports <br> **Signals:** cruise, altitude, level flight, en route <br> **Tests:** Less common but includes catastrophic failures |
| PHS-004 | ground operations taxi incidents ramp accidents | **SQL:** `chunk_text LIKE '%during taxi%' OR '%ramp%'` <br> **Expected Reports:** ~59 ground ops reports <br> **Signals:** taxi, ramp, ground, apron, gate <br> **Tests:** Non-flight accidents still in NTSB corpus |

**Why flight phase matters for safety research:**
- **Takeoff + Approach/Landing** = ~80% of all accidents but only ~6% of flight time
- Researchers studying "approach accidents" need results filtered to that phase
- A good model should understand "final approach" relates to "approach and landing"

---

## How We Measure Success

### Primary Metrics

| Metric | What It Measures | Interpretation |
|--------|------------------|----------------|
| **MRR** (Mean Reciprocal Rank) | Position of first relevant result | 1.0 = first result is always relevant; 0.5 = relevant result is typically second |
| **Hit@K** | Did we find ANY relevant result in top K? | "Did the user find what they needed in the first K results?" |
| **Precision@K** | What fraction of top K results are relevant? | Higher = less noise in results |
| **Recall@K** | What fraction of ALL relevant chunks did we find in top K? | Higher = more complete results |
| **nDCG@10** | Ranking quality with graded relevance | Penalizes relevant results appearing lower in the list |

### Example: Understanding MRR

If we run 3 queries and the first relevant result appears at positions 1, 2, and 5:
- Query 1: First relevant at position 1 → Reciprocal Rank = 1/1 = 1.0
- Query 2: First relevant at position 2 → Reciprocal Rank = 1/2 = 0.5
- Query 3: First relevant at position 5 → Reciprocal Rank = 1/5 = 0.2

MRR = (1.0 + 0.5 + 0.2) / 3 = **0.567**

A model with MRR of 0.8 is much better than one with MRR of 0.5.

### Performance Metrics

| Metric | What It Measures |
|--------|------------------|
| **Embed Latency** | Time to convert query text to vector |
| **Search Latency** | Time for Qdrant to find similar vectors |
| **Total Latency** | End-to-end time for the full search |

MiniLM (384d) should be faster than MIKA (768d) because smaller vectors = faster math.

---

## How Ground Truth Works

For each query, we have SQL that verifies which chunks SHOULD be returned:

```yaml
- id: "INC-001"
  query: "Alaska Airlines flight 261 horizontal stabilizer jackscrew failure"
  ground_truth:
    verification_sql: "WHERE lower(chunk_text) LIKE '%alaska%' AND lower(chunk_text) LIKE '%261%' AND lower(chunk_text) LIKE '%jackscrew%'"
    expected_report_ids: ["AAR0201.pdf"]
    relevance_signals: ["jackscrew", "horizontal stabilizer", "acme nut", "trim system"]
```

This means:
1. Run the SQL against `chunks.parquet` to find all chunks that mention "Alaska," "261," and "jackscrew"
2. Those chunks are the ground truth - they ARE relevant
3. When we run the search, check if those chunks appear in the top K results

This approach is **reproducible** - anyone can re-run the SQL and verify the results.

---

## Statistical Methodology

### Why Standard Tests Are Problematic Here

**MRR scores are NOT normally distributed.** They are:
- **Bounded** - values fall in [0, 1]
- **Discrete** - only takes values like 1, 0.5, 0.333, 0.25, 0.2, etc. (reciprocals of integers)
- **Zero-inflated** - many queries may have MRR=0 (no relevant result found)

This violates assumptions of the paired t-test. **Do not rely on t-test p-values.**

### Recommended Approach (In Order of Reliability)

| Method | What It Does | Why It's Better |
|--------|--------------|-----------------|
| **Bootstrap 95% CI** | Resamples 1000x to estimate confidence interval on mean difference | Makes NO distributional assumptions. If CI excludes 0, difference is reliable. |
| **Win/Loss/Tie Count** | Count queries where each model wins | Intuitive, robust to outliers, shows practical difference |
| **Wilcoxon Signed-Rank** | Non-parametric test for paired differences | Doesn't assume normality, but still assumes symmetric differences |
| **Effect Size** | Mean difference in practical units (MRR points) | Tells you if the difference MATTERS, not just if it's non-zero |

### Interpreting Results Correctly

**Example Output:**
```
MIKA mean MRR:   0.72
MiniLM mean MRR: 0.65
Difference:     +0.07
Bootstrap 95% CI: [0.02, 0.12]
Win/Loss/Tie: MIKA 28 / MiniLM 15 / Tie 7
```

**Correct interpretation:**
- The bootstrap CI [0.02, 0.12] does NOT include 0, so MIKA is reliably better
- MIKA wins on 28/50 queries (56%), MiniLM wins on 15/50 (30%)
- Effect size is +0.07 MRR points - this is MEANINGFUL (moves first relevant result up ~1 position on average)

**What NOT to say:**
- ❌ "p < 0.05 so MIKA is significantly better" (t-test assumptions violated)
- ❌ "MIKA is 10% better" (0.72 is not 10% more than 0.65 in any meaningful sense)

### Statistical Power Consideration

With 50 queries, we can reliably detect MRR differences of ~0.05 or larger. Smaller differences may be real but undetectable with this sample size. This is acceptable because:
- Differences < 0.05 MRR are unlikely to matter in practice
- Adding more queries would require human labeling effort

### Multiple Comparisons

We compute metrics across 6 categories and 3 difficulty levels. When making many comparisons, some will appear significant by chance. **Interpret per-category differences cautiously** - focus on overall metrics for the primary conclusion.

---

## Critical Limitation: Semantic Evaluation Requires Human Review

### The Problem with Keyword-Based Ground Truth

Our SQL-based validation has a **fundamental limitation**: it measures how well semantic search mimics keyword search, NOT semantic search's actual value.

**Example of the problem:**

```
Query: "pilot fatigue circadian rhythm duty time regulations"
SQL Ground Truth: WHERE chunk_text LIKE '%fatigue%' AND '%pilot%'

Chunk A (MATCHES SQL - counted as relevant):
  "Pilot fatigue was cited as a contributing factor..."

Chunk B (DOESN'T MATCH SQL - NOT counted as relevant):
  "The captain had been awake for 18 hours prior to the accident.
   His reaction time during the emergency was significantly degraded,
   consistent with performance decrements observed in sleep deprivation studies."
```

**Chunk B is HIGHLY relevant** - it describes fatigue effects without using the word "fatigue." A good semantic model SHOULD find this. But our keyword-based ground truth would score it as a FALSE POSITIVE, penalizing the model for being smart.

### What This Means for Our Benchmark

| Query Category | Keyword Validation Works? | Why |
|----------------|---------------------------|-----|
| Incident Lookup | YES | Looking for specific report IDs - exact match |
| Section Queries | YES | Looking for specific section names - exact match |
| Aircraft Queries | MOSTLY | Aircraft names are usually explicit (Boeing 737) |
| Phase Queries | MOSTLY | Flight phases usually stated explicitly |
| Conceptual Queries | PARTIALLY | Some keyword overlap, but semantic understanding matters |
| Comparative Queries | POORLY | Themes often expressed without exact keywords |

**For Categories 2 and 4 (Conceptual and Comparative), our automated metrics are LOWER BOUNDS on true performance.** The models may be better than we're measuring.

### Human-in-the-Loop Evaluation Protocol

To properly evaluate semantic understanding, we need human review. Here's the protocol:

#### Phase 1: Automated Benchmark (What We Have)
Run `python -m eval.benchmark run` to get baseline metrics.

#### Phase 2: Human Review of "Surprise Retrievals"

For each conceptual/comparative query, a human reviews the model's top-10 results and marks:

```
For each result:
[ ] KEYWORD MATCH - matches SQL ground truth (already counted as relevant)
[ ] SEMANTIC MATCH - relevant but doesn't match SQL (model found something smart)
[ ] FALSE POSITIVE - not relevant (model made a mistake)
```

**"Semantic Match" is the key metric** - it measures the model's ability to find relevant content that keyword search would miss.

**Auto-Fill Feature:** When you run `export-review`, the benchmark system automatically:
1. Loads the actual chunk text from `chunks.parquet`
2. Checks each result against the SQL ground truth and relevance signals
3. Pre-fills `KEYWORD_MATCH` for results that match
4. Leaves `judgment` empty for results that need human evaluation

This reduces human effort by ~50% - you only need to judge results where the automated system couldn't determine relevance.

#### Human Review Template

```yaml
query_id: CON-005
query: "pilot fatigue circadian rhythm duty time regulations"
reviewer: [name]
date: [date]

results:
  - rank: 1
    chunk_id: AAR1001.pdf_chunk_042
    judgment: KEYWORD_MATCH
    notes: "Explicitly mentions pilot fatigue"

  - rank: 2
    chunk_id: AAR9006.pdf_chunk_018
    judgment: SEMANTIC_MATCH
    notes: "Discusses crew alertness and duty period without using 'fatigue'"

  - rank: 3
    chunk_id: AAB0401.pdf_chunk_007
    judgment: FALSE_POSITIVE
    notes: "About maintenance fatigue (metal), not pilot fatigue"
```

#### Computing Semantic Precision

After human review, compute:

```
Semantic Precision@10 = (KEYWORD_MATCH + SEMANTIC_MATCH) / 10
Semantic Lift = SEMANTIC_MATCH / 10  # Value-add over keyword search
False Positive Rate = FALSE_POSITIVE / 10
```

**Semantic Lift is what differentiates a good embedding model from keyword search.** If MIKA has higher Semantic Lift than MiniLM, it's finding more relevant content that keywords would miss.

### Recommended Evaluation Workflow

```
1. Run automated benchmark
   └── Produces baseline MRR, Precision, Recall for all queries

2. For conceptual/comparative queries (20 queries):
   └── Human reviews top-10 results for each model
   └── ~400 judgments total (20 queries × 10 results × 2 models)
   └── Time estimate: 2-3 hours

3. Compute adjusted metrics:
   └── Automated metrics (categories 1, 3, 5, 6)
   └── Human-adjusted metrics (categories 2, 4)

4. Final comparison:
   └── Report BOTH automated and human-adjusted metrics
   └── Highlight Semantic Lift as key differentiator
```

### Why This Matters

If we ONLY use automated keyword-based evaluation, we might conclude:
- "MiniLM and MIKA have similar performance"

But with human evaluation, we might find:
- "MIKA finds 30% more relevant content that keywords would miss"

This is the VALUE of a domain-specific model - and we can't measure it without human review.

---

## Running the Benchmark

### Prerequisites

1. Embeddings generated for both models:
   ```bash
   python -m embeddings.cli embed both
   ```

2. Embeddings uploaded to Qdrant:
   ```bash
   python -m embeddings.cli upload both
   ```

3. Environment variables set in `.env`:
   ```
   QDRANT_URL=https://your-cluster.qdrant.io:6333
   QDRANT_API_KEY=your_api_key
   ```

### Complete Evaluation Workflow

The benchmark has 5 steps. Use `python -m eval.benchmark status` to see current progress.

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Run Automated Benchmark                                │
│  python -m eval.benchmark run                                   │
│  → Runs 50 queries against both models                         │
│  → Outputs: eval/results/benchmark_*.json                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Export for Human Review                                │
│  python -m eval.benchmark export-review                         │
│  → Creates YAML files for 20 semantic queries (per model)      │
│  → Outputs: eval/human_reviews/review_*.yaml                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Complete Human Reviews (MANUAL)                        │
│  Open each YAML file and for each result assign:               │
│    - KEYWORD_MATCH: Matches SQL ground truth                   │
│    - SEMANTIC_MATCH: Relevant but doesn't match SQL            │
│    - FALSE_POSITIVE: Not relevant                              │
│  Set review_complete: true when done                           │
│  Time estimate: 2-3 hours for 40 files (20 queries × 2 models)│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Import Human Reviews                                   │
│  python -m eval.benchmark import-review                         │
│  → Reads completed YAML files                                  │
│  → Computes semantic precision and semantic lift               │
│  → Outputs: eval/results/human_review_*.json                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Generate Final Report                                  │
│  python -m eval.benchmark final-report                          │
│  → Combines automated + human metrics                          │
│  → Generates recommendation                                     │
│  → Outputs: eval/final_benchmark_report.md                     │
└─────────────────────────────────────────────────────────────────┘
```

### CLI Commands Reference

| Command | Description |
|---------|-------------|
| `status` | Show current progress and next steps |
| `run` | Run automated benchmark on both/one model |
| `export-review` | Export results for human evaluation |
| `import-review` | Import completed human reviews |
| `final-report` | Generate combined final report |
| `report` | Generate automated-only report |
| `validate` | Validate query definitions |

### Quick Commands

```bash
# Check status and next steps
python -m eval.benchmark status

# Run automated benchmark
python -m eval.benchmark run                # Both models
python -m eval.benchmark run -m minilm      # Single model

# Human review workflow
python -m eval.benchmark export-review      # Export for review
python -m eval.benchmark import-review      # Import completed reviews

# Generate reports
python -m eval.benchmark report             # Automated only
python -m eval.benchmark final-report       # Combined with human review

# Validation
python -m eval.benchmark validate
```

### Expected Time Investment

| Step | Time | Notes |
|------|------|-------|
| Automated benchmark | ~1 minute | Requires Qdrant connection |
| Export for review | ~5 seconds | Creates 40 YAML files with auto-filled judgments |
| Human reviews | 1-2 hours | ~200 judgments (auto-fill reduces by ~50%) |
| Import reviews | ~5 seconds | Reads YAML files |
| Final report | ~10 seconds | Generates markdown + Streamlit Parquet files |
| **Total** | **1-2 hours** | Auto-fill reduces human review effort |

**Note:** The export step auto-fills KEYWORD_MATCH judgments by checking each result against SQL ground truth and relevance signals. You only need to manually judge results where the automated system couldn't determine relevance (~50% of results).

---

## Output Files

### Results Directory Structure

```
eval/results/
├── minilm_results.json      # Full results for MiniLM
├── minilm_results.parquet   # Streamlit-ready DataFrame
├── mika_results.json        # Full results for MIKA
├── mika_results.parquet     # Streamlit-ready DataFrame
└── comparison_report.json   # Head-to-head comparison
```

### JSON Output Schema

```json
{
  "model_name": "minilm",
  "collection_name": "riskradar_minilm",
  "total_queries": 50,
  "timestamp": "2026-01-13T10:30:00",
  "metrics": {
    "mean_mrr": 0.72,
    "std_mrr": 0.18,
    "mean_hit_at_1": 0.58,
    "mean_hit_at_5": 0.82,
    "mean_hit_at_10": 0.90,
    "mean_precision_at_5": 0.45,
    "mean_ndcg_at_10": 0.68,
    "mean_embed_latency_ms": 45.2,
    "mean_search_latency_ms": 32.1
  },
  "category_metrics": {
    "incident_lookup": {
      "mean_mrr": 0.92,
      "count": 10
    },
    "conceptual_queries": {
      "mean_mrr": 0.58,
      "count": 12
    }
  },
  "difficulty_metrics": {
    "easy": {"mean_mrr": 0.88, "count": 9},
    "medium": {"mean_mrr": 0.72, "count": 27},
    "hard": {"mean_mrr": 0.52, "count": 14}
  },
  "query_results": [
    {
      "query_id": "INC-001",
      "query_text": "Alaska Airlines flight 261...",
      "mrr": 1.0,
      "hit_at_10": true,
      "top_results": [...]
    }
  ]
}
```

### Parquet Schema (for Streamlit)

| Column | Type | Description |
|--------|------|-------------|
| query_id | string | Unique identifier (INC-001, CON-002, etc.) |
| query_text | string | The actual search query |
| category | string | incident_lookup, conceptual_queries, etc. |
| difficulty | string | easy, medium, hard |
| intent | string | lookup, exploratory, analytical |
| mrr | float | Reciprocal rank of first relevant result |
| hit_at_1 | bool | Was top result relevant? |
| hit_at_5 | bool | Any relevant result in top 5? |
| hit_at_10 | bool | Any relevant result in top 10? |
| precision_at_5 | float | Fraction of top 5 that are relevant |
| recall_at_10 | float | Fraction of relevant found in top 10 |
| ndcg_at_10 | float | Normalized discounted cumulative gain |
| embed_latency_ms | float | Time to embed query |
| search_latency_ms | float | Time for Qdrant search |
| total_latency_ms | float | Total end-to-end time |

---

## Expected Results

Based on the model characteristics:

### Where MIKA Should Excel
- **Conceptual queries** - Aviation terminology (CON-001 through CON-012)
- **Hard queries** - Requires deep semantic understanding
- **Aerospace-specific terms** - "uncommanded," "GPWS," "CRM," etc.

### Where MiniLM Should Be Competitive
- **Incident lookups** - High keyword overlap makes this easier
- **Easy queries** - Both models should do well
- **Speed** - MiniLM should be 2x faster (384 vs 768 dimensions)

### Decision Framework

| If... | Then... |
|-------|---------|
| MIKA significantly better (bootstrap CI excludes 0, wins majority of queries) | Use MIKA for production |
| Models statistically tied (CI includes 0, similar win counts) | Use MiniLM (faster, smaller) |
| MiniLM better on some categories | Consider hybrid approach |

---

## Streamlit Integration

The final report generates multiple Parquet files optimized for Streamlit visualization:

### Available Parquet Files

| File | Description | Best Used For |
|------|-------------|---------------|
| `query_comparison.parquet` | Per-query metrics for both models side-by-side | Scatter plots, per-query tables |
| `aggregate_metrics.parquet` | Summary metrics for both models | Dashboard metric cards |
| `category_metrics.parquet` | Breakdown by category | Grouped bar charts |
| `difficulty_metrics.parquet` | Breakdown by difficulty | Difficulty analysis |
| `human_review_details.parquet` | Per-query human evaluation (if available) | Semantic lift analysis |
| `benchmark_decision.parquet` | Final decision and recommendation | Summary cards |

### Example Streamlit Code

```python
import pandas as pd
import streamlit as st

# Load benchmark comparison data
comparison = pd.read_parquet("eval/results/query_comparison.parquet")
aggregates = pd.read_parquet("eval/results/aggregate_metrics.parquet")
categories = pd.read_parquet("eval/results/category_metrics.parquet")
decision = pd.read_parquet("eval/results/benchmark_decision.parquet")

# Show recommendation
st.header("Model Recommendation")
st.metric("Winner", decision['recommendation'].iloc[0])

# Compare MRR side-by-side
st.header("MRR by Model")
col1, col2 = st.columns(2)
with col1:
    minilm = aggregates[aggregates['model'] == 'MiniLM'].iloc[0]
    st.metric("MiniLM MRR", f"{minilm['mean_mrr']:.3f}")
with col2:
    mika = aggregates[aggregates['model'] == 'MIKA'].iloc[0]
    st.metric("MIKA MRR", f"{mika['mean_mrr']:.3f}")

# Bar chart by category
st.header("MRR by Category")
st.bar_chart(categories.pivot(index='category', columns='model', values='mrr'))

# Per-query scatter plot
st.header("Per-Query Comparison")
st.scatter_chart(comparison, x="minilm_mrr", y="mika_mrr", color="category")

# Winner distribution
st.header("Win/Loss/Tie")
win_counts = comparison['winner'].value_counts()
st.bar_chart(win_counts)

# Semantic lift (if human review completed)
try:
    human = pd.read_parquet("eval/results/human_review_details.parquet")
    st.header("Semantic Lift by Model")
    st.bar_chart(human.groupby('model')['semantic_lift'].mean())
except FileNotFoundError:
    st.info("Human review not yet completed")
```

---

## Files Reference

### Source Files

| File | Purpose |
|------|---------|
| `gold_queries.yaml` | All 50 test queries with SQL ground truth |
| `benchmark.py` | Benchmark runner CLI (~1600 lines) |
| `__init__.py` | Package marker |
| `README.md` | This documentation |

### Generated Files (Automated Benchmark)

After running `python -m eval.benchmark run`:

```
eval/results/
├── benchmark_minilm_20260113_103000.json    # Full MiniLM results
├── benchmark_minilm_20260113_103000.parquet # Streamlit-ready DataFrame
├── benchmark_mika_20260113_103100.json      # Full MIKA results
└── benchmark_mika_20260113_103100.parquet   # Streamlit-ready DataFrame
```

### Generated Files (Human Review Export)

After running `python -m eval.benchmark export-review`:

```
eval/human_reviews/
├── review_CON-001_minilm.yaml   # Conceptual query 1, MiniLM results
├── review_CON-001_mika.yaml     # Conceptual query 1, MIKA results
├── review_CON-002_minilm.yaml
├── ...
├── review_CMP-001_minilm.yaml   # Comparative query 1, MiniLM results
├── review_CMP-001_mika.yaml
└── ...                          # 40 files total (20 queries × 2 models)
```

### Human Review File Structure

Each exported file has this structure. **Note: KEYWORD_MATCH judgments are auto-filled where possible.**

```yaml
metadata:
  query_id: CON-001
  model: minilm
  category: conceptual_queries
  reviewer: ""           # YOUR NAME
  review_date: ""        # DATE
  review_complete: false # SET TO true WHEN DONE

query:
  text: "uncommanded flight control surface movement"
  sql_ground_truth: "WHERE lower(chunk_text) LIKE '%uncommanded%'..."
  expected_report_ids: [AAR0101, AAR9901, AAR0404]
  relevance_signals: [uncommanded, hardover, runaway]

instructions:
  keyword_match: "Already filled in by automated check..."
  semantic_match: "YOU FILL THIS: Result is relevant but doesn't match keywords..."
  false_positive: "YOU FILL THIS: Result is NOT relevant to the query..."

summary:
  auto_keyword_matches: 6    # How many were auto-filled
  needs_human_review: 4      # How many YOU need to judge

results:
  - rank: 1
    chunk_id: "AAR9901.pdf_chunk_042"
    report_id: "AAR9901.pdf"
    section_name: "ANALYSIS"
    score: 0.847
    text_preview: "The uncommanded rudder hardover caused the aircraft to..."
    judgment: "KEYWORD_MATCH"      # AUTO-FILLED - matches relevance signals
    notes: "[AUTO] Matches relevance signals/SQL pattern"
  - rank: 2
    chunk_id: "AAR0201.pdf_chunk_018"
    report_id: "AAR0201.pdf"
    section_name: "CONCLUSIONS"
    score: 0.812
    text_preview: "The flight control system exhibited anomalous behavior..."
    judgment: ""                   # EMPTY - needs human review
    notes: ""                      # You explain your judgment here
  - rank: 3
    ...
```

**Your task:** Only review results where `judgment` is empty. For each:
- Assign `SEMANTIC_MATCH` if the chunk is relevant but doesn't match keywords
- Assign `FALSE_POSITIVE` if the chunk is not relevant
- You can override auto-filled `KEYWORD_MATCH` judgments if they're wrong

### Generated Files (After Human Review Import)

After running `python -m eval.benchmark import-review`:

```
eval/results/
├── human_review_minilm.json    # Aggregated human judgments for MiniLM
└── human_review_mika.json      # Aggregated human judgments for MIKA
```

### Final Report Files

After running `python -m eval.benchmark final-report`:

```
eval/
├── final_benchmark_report.md   # Combined report with recommendation
└── results/
    ├── combined_metrics.json          # Machine-readable combined metrics
    │
    │   # Streamlit-Ready Parquet Files:
    ├── query_comparison.parquet       # Per-query metrics (both models)
    ├── aggregate_metrics.parquet      # Summary dashboard metrics
    ├── category_metrics.parquet       # By-category breakdown
    ├── difficulty_metrics.parquet     # By-difficulty breakdown
    ├── human_review_details.parquet   # Human eval per-query (if available)
    └── benchmark_decision.parquet     # Final recommendation
```

### Parquet Schema Reference

**query_comparison.parquet** - One row per query (50 rows)
| Column | Type | Description |
|--------|------|-------------|
| query_id | string | Query identifier |
| query_text | string | The search query |
| category | string | Query category |
| difficulty | string | easy/medium/hard |
| minilm_mrr | float | MiniLM MRR score |
| mika_mrr | float | MIKA MRR score |
| mrr_diff | float | MIKA - MiniLM |
| winner | string | "MIKA", "MiniLM", or "Tie" |
| ... | | All metrics for both models |

**aggregate_metrics.parquet** - One row per model (2 rows)
| Column | Type | Description |
|--------|------|-------------|
| model | string | "MiniLM" or "MIKA" |
| mean_mrr | float | Overall MRR |
| mean_hit_at_10 | float | Overall Hit@10 |
| mean_latency_ms | float | Average latency |
| semantic_lift | float | Human-eval semantic lift (nullable) |

**benchmark_decision.parquet** - Single row with final decision
| Column | Type | Description |
|--------|------|-------------|
| recommendation | string | Winner and reason |
| mika_auto_advantage | bool | MIKA better on automated? |
| mika_semantic_advantage | bool | MIKA better on semantic? |
| mrr_difference | float | MIKA - MiniLM MRR |
| semantic_lift_difference | float | MIKA - MiniLM lift |

---

## Troubleshooting

### "Collection not found" Error
```bash
# Verify collections exist in Qdrant
python -m embeddings.cli verify minilm
python -m embeddings.cli verify mika
```

### "Connection refused" Error
Check `.env` has correct Qdrant credentials:
```bash
cat .env | grep QDRANT
```

### "No results returned" for a Query
Some queries may legitimately return no results if:
- The query tests a negative case (like INC-010 Air France 447)
- The verification SQL is too restrictive

### Slow Performance
- Qdrant Cloud latency depends on your location vs. cluster location
- First query may be slow due to model loading (subsequent queries cached)

---

## Summary: Evaluation Approach by Category

| Category | Queries | Automated Eval | Human Eval Needed? | Key Metric |
|----------|---------|----------------|-------------------|------------|
| **Incident Lookup** | 10 | Report ID matching | NO | MRR (first relevant report) |
| **Section Queries** | 10 | Section name matching | NO | Section Accuracy@10 |
| **Aircraft Queries** | 6 | Aircraft name in text | Minimal | Recall@10 |
| **Phase Queries** | 4 | Flight phase in text | Minimal | Recall@10 |
| **Conceptual Queries** | 12 | Keyword patterns | **YES** | Semantic Lift |
| **Comparative Queries** | 8 | Keyword + section | **YES** | Semantic Lift |
| **TOTAL** | 50 | 30 fully automated | 20 need human review | |

### Decision Framework

```
IF automated metrics show clear winner (>0.10 MRR difference, CI excludes 0):
   └── Winner is likely correct, human review confirms but unlikely to change outcome

IF automated metrics are close (<0.05 MRR difference):
   └── Human review is CRITICAL - Semantic Lift will determine winner
   └── Model with higher Semantic Lift finds more content keywords would miss

IF human review shows MIKA has higher Semantic Lift:
   └── MIKA is better for semantic search even if automated metrics are similar
   └── Use MIKA for production (unless latency is critical)

IF human review shows similar Semantic Lift:
   └── Models are equivalent for this corpus
   └── Use MiniLM (faster, smaller) for production
```
