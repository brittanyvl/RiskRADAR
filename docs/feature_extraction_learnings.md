# Feature Extraction: Learnings & Decisions

**Date:** 2026-01-27
**Phase:** Risk Profiler Development

## Summary

This document captures the learnings from the manual validation of aircraft extraction for the RiskRADAR Bayesian Risk Model.

## Results

| Metric | Before Validation | After Manual Review | After Pattern Improvements |
|--------|-------------------|---------------------|---------------------------|
| Aircraft Coverage | 372/510 (72.9%) | 461/510 (90.4%) | 473/510 (92.7%) |
| Reports with Aircraft | 372 | 461 | 473 |
| Critical Gap (taxonomy but no aircraft) | 90 | 1 | 1 |
| Multi-aircraft Reports Identified | 0 | 10 | 10 |
| Duplicate Reports Flagged | 0 | 1 | 1 |
| Total Patterns in Database | ~200 | ~397 | 686 |
| Auto-Extraction Rate | ~44% | ~59% | ~65% |

## Key Findings from Manual Review

### 1. Boeing/Douglas Confusion (Major Issue)

**Problem:** Many reports incorrectly labeled as "Boeing" were actually McDonnell Douglas aircraft.

**Root Cause:** The pattern matcher found "Boeing" in titles that mentioned both manufacturers, or the title format was ambiguous.

**Examples:**
- AAR7302: Labeled Boeing → Actually McDonnell Douglas DC-10
- AAR7303: Labeled Boeing → Actually McDonnell Douglas DC-9
- AAR7115: Labeled Boeing → Actually Douglas DC-9-32
- AAR8307: Labeled Boeing → Actually McDonnell Douglas DC-8
- AAR8605: Labeled Boeing → Actually Lockheed L-1011

**Fix Applied:** 23 reports manually corrected to proper manufacturer.

### 2. Missing Model Numbers (Common Issue)

**Problem:** Make was extracted but specific model numbers were missed.

**Affected Aircraft Types:**
- **Learjet**: 15+ reports had "Learjet" but missed models like 23, 24, 25, 25D, 35A
- **Cessna Citation**: Missed 500, 501, 551
- **Gulfstream**: Missed G-II, G-IV, G650
- **King Air**: Got "King Air" but missed 100, 200
- **Boeing**: Got "Boeing" but missed 707, 727, 737, 747

**Lesson:** Model number extraction should use regex for numeric patterns following make.

### 3. Foreign Aircraft Not Recognized

**Missing Patterns:**
- British Aerospace J-3101 (Jetstream)
- BAE Systems BAE-J3201
- CASA C-212
- ATR 72-212
- Nord 262
- Aerospatiale SA365N1
- Eurocopter EC-135, AS350
- de Havilland DHC-3, DHC-3T, DH-104

**Fix Applied:** Added to manual corrections; patterns should be added for future extraction.

### 4. Helicopters Under-Represented

**Missing:**
- Eurocopter EC-135, AS350 B2, AS350 B3
- Sikorsky S-61L
- Bell UH-1B
- Agusta A109E
- Aerospatiale SA365N1

**Lesson:** Helicopter patterns need expansion, including Eurocopter → Airbus Helicopters transition.

### 5. Multi-Aircraft Reports (New Discovery)

**Finding:** 10 reports involved multiple aircraft (collisions, runway incursions, near-misses).

**Database Change:** Created `report_aircraft` table with `aircraft_sequence` field to support 1:N relationship.

**Examples:**
| Report | Event Type | Aircraft 1 | Aircraft 2 |
|--------|------------|------------|------------|
| AAR7603 | Near collision | Douglas DC-10 | Lockheed L-1011 |
| AAR8210 | Midair collision | F-111D | Cessna TU-206G |
| AAR8703 | Midair collision | DHC-7 | Bell 206B |
| AIR2401 | Runway incursion | Boeing 777 | Boeing 737 |

**Indicators:** "collision", "midair", "near-miss", "runway incursion"

### 6. Aircraft Info in Body, Not Title

**Finding:** Some reports have generic titles like "Aviation Accident Report AAR-81-04" with aircraft info only in the body (History of Flight, Abstract).

**Lesson:** For complete extraction, need to parse first 1-3 pages of report body.

### 7. Duplicate Reports

**Finding:** Report AAR8803S.pdf is a duplicate of AAR8803.pdf (same accident).

**Database Change:** Added `is_duplicate` and `related_report_id` columns to `report_features`.

**Lesson:** For statistical analysis, filter out duplicates to avoid double-counting accidents.

## Schema Changes Made

### New Table: `report_aircraft`
```sql
CREATE TABLE report_aircraft (
    id INTEGER PRIMARY KEY,
    report_id TEXT NOT NULL,
    aircraft_sequence INTEGER DEFAULT 1,  -- 1=primary, 2=secondary
    aircraft_make TEXT,
    aircraft_model TEXT,
    aircraft_model_series TEXT,
    aircraft_category TEXT,
    aircraft_registration TEXT,
    source TEXT,        -- 'title', 'abstract', 'body', 'manual'
    confidence TEXT,    -- 'high', 'medium', 'low', 'manual'
    extraction_notes TEXT,
    UNIQUE(report_id, aircraft_sequence)
);
```

### New Columns in `report_features`
- `is_duplicate` (INTEGER): Flag for duplicate reports
- `related_report_id` (TEXT): Link to original report
- `duplicate_notes` (TEXT): Explanation
- `extraction_notes` (TEXT): Notes about extraction quality

## Pattern Improvements (Phase 2)

### SQL Case Sensitivity Fix

**Problem:** SQL LIKE is case-sensitive by default, missing matches like "CESSNA" vs "Cessna".

**Fix Applied:** Changed lookup query to use UPPER() for case-insensitive matching:
```python
# Before
WHERE ? LIKE '%' || pattern || '%'

# After
WHERE UPPER(?) LIKE '%' || UPPER(pattern) || '%'
```

### Additional Patterns Added

Added 289 new patterns covering gaps identified in manual review:

| Category | Patterns Added | Examples |
|----------|---------------|----------|
| Dirty Text Variants | 11 | DC8, MD80, MD11 (without hyphens) |
| Short Brothers | 7 | Skyvan, SD3-30, Sherpa |
| Britten-Norman | 5 | Islander, BN-2, Trislander |
| Cessna Singles | 20 | 150-210, P210, TU206 |
| Cessna Twins | 21 | 303-441, Citations |
| Beechcraft | 35 | 18 series, Baron, King Air, 99/1900 |
| Learjet | 25 | Models 23-60, Gates Learjet |
| De Havilland | 20 | DHC-2 through DHC-8, Beaver, Otter |
| Helicopters | 20 | S-61, UH-1, EC-135, AS350 |
| Regional Jets | 30 | BAe, Fokker, SAAB, Embraer |
| Turboprops | 40 | ATR, CASA, Nord, Metroliner |

### Fuzzy Matching

Added fallback fuzzy matching for edge cases using trigram-based Jaccard similarity:
- Threshold: 0.85 (85% similarity required)
- Confidence: `low` for fuzzy matches
- Catches typos and variant spellings

### Results

| Metric | Before | After |
|--------|--------|-------|
| Patterns in database | 397 | 686 |
| Auto-extracted | 302 | 314 |
| Coverage | 90.4% | 92.7% |
| Manual corrections still needed | 159 | 159 |

### Unmatched Reports Analysis

37 reports remain without aircraft - most are NOT accident reports:
- Safety Studies (AIR-24-03, AIR2201-2209)
- Safety Recommendations (ASR series)
- Hazmat Reports (HZB, HZMSR)
- Generic titles (AAR-81-04 style)

These appropriately have no aircraft as they are policy/recommendation documents.

## Recommendations for Future Work

### 1. Pattern Improvements
- Add comprehensive patterns for foreign manufacturers (BAe, CASA, ATR, Nord) [DONE]
- Add helicopter patterns (Eurocopter/Airbus Helicopters, Aerospatiale) [DONE]
- Improve model number extraction with numeric regex [PARTIAL]

### 2. Body Parsing
- For reports with no aircraft in title, parse Abstract and History of Flight sections
- Use existing chunk data to find aircraft mentions in first few pages

### 3. Multi-Aircraft Detection
- Automatically flag reports with collision/incursion keywords
- Prompt for second aircraft during review

### 4. Duplicate Detection
- Build automated check for report IDs with letter suffixes (e.g., "S")
- Cross-reference by accident date and location

## Data Quality Metrics

### Final Coverage
| Feature | Count | Coverage |
|---------|-------|----------|
| Aircraft Category | 461 | 90.4% |
| Aircraft Make | 461 | 90.4% |
| Region | 461 | 90.4% |
| Season | 489 | 95.9% |

### Confidence Distribution
| Confidence | Count | Percentage |
|------------|-------|------------|
| Manual | 155 | 33.6% |
| High | ~217 | 47.1% |
| Medium | ~89 | 19.3% |

### Multi-Aircraft Statistics
- Total multi-aircraft reports: 10
- Total aircraft entries: 165
- Reports with 2+ aircraft: 10

## Files Created/Modified

### New Files
- `risk_profiler/schema_v2.py` - Schema migrations
- `risk_profiler/data/manual_aircraft_corrections.json` - 145 corrections
- `risk_profiler/import_corrections.py` - Import script
- `risk_profiler/review/aircraft_review.html` - Interactive review page
- `risk_profiler/review/gap_analysis.html` - Gap analysis page
- `docs/feature_extraction_learnings.md` - This document

### Modified Tables
- `report_features` - Added columns, updated 155+ records
- `report_aircraft` - New table, 165 records

## Conclusion

The manual validation process improved aircraft extraction coverage from 72.9% to 90.4%, a 17.5 percentage point improvement. Key learnings include the importance of handling manufacturer confusion (Boeing vs Douglas), the need for comprehensive foreign aircraft patterns, and the discovery of multi-aircraft scenarios requiring schema changes.

The human-in-the-loop validation approach was essential for achieving high data quality and will inform future automation improvements.
