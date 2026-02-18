"""
analytics.queries.glossary_data — Structured glossary content from taxonomy sources.

Sources content from existing modules. No new content to write for L1/L2.
"""

import pandas as pd


def get_l1_glossary() -> pd.DataFrame:
    """
    DataFrame of 27 L1 CICTT categories: code, name, description.
    """
    from taxonomy.cictt import CICTT_CATEGORIES
    rows = [
        {"code": c.code, "name": c.name, "description": c.description}
        for c in CICTT_CATEGORIES
    ]
    return pd.DataFrame(rows)


def get_l2_glossary() -> pd.DataFrame:
    """
    DataFrame of all L2 subcategories: code, name, description, parent_code.
    """
    from taxonomy.subcategories import ALL_SUBCATEGORIES
    rows = [
        {
            "code": c.code,
            "name": c.name,
            "description": c.description,
            "parent_code": c.parent_code,
        }
        for c in ALL_SUBCATEGORIES
    ]
    return pd.DataFrame(rows)


def get_feature_definitions() -> dict[str, pd.DataFrame]:
    """
    Dict of DataFrames from risk_profiler.schema constants.

    Keys: "aircraft_categories", "us_regions".
    """
    from risk_profiler.schema import AIRCRAFT_CATEGORIES, US_CENSUS_REGIONS

    ac_df = pd.DataFrame([
        {"category": k, "description": v}
        for k, v in AIRCRAFT_CATEGORIES.items()
    ])

    region_rows = []
    for code, (state, region, division) in US_CENSUS_REGIONS.items():
        region_rows.append({
            "state_code": code,
            "state_name": state,
            "region": region,
            "division": division,
        })
    region_df = pd.DataFrame(region_rows)

    return {"aircraft_categories": ac_df, "us_regions": region_df}


def get_aviation_terms() -> list[dict]:
    """
    Hardcoded list of ~20 aviation terminology definitions.

    Each dict has keys: term, abbreviation (optional), definition.
    """
    return [
        {"term": "Visual Meteorological Conditions", "abbreviation": "VMC",
         "definition": "Weather conditions in which pilots have sufficient visibility to fly by visual reference. Generally requires visibility of 3 statute miles or more and a ceiling of 1,000 feet or more."},
        {"term": "Instrument Meteorological Conditions", "abbreviation": "IMC",
         "definition": "Weather conditions that require pilots to fly primarily by reference to instruments. Visibility is below VMC minimums due to clouds, fog, rain, or other obscurations."},
        {"term": "14 CFR Part 91", "abbreviation": "Part 91",
         "definition": "Federal aviation regulation governing general aviation operations, including private and non-commercial flying."},
        {"term": "14 CFR Part 121", "abbreviation": "Part 121",
         "definition": "Federal aviation regulation governing scheduled air carrier operations (airlines). Requires the highest level of safety standards."},
        {"term": "14 CFR Part 135", "abbreviation": "Part 135",
         "definition": "Federal aviation regulation governing commuter and on-demand air carrier operations, including air taxi and charter flights."},
        {"term": "Maintenance, Repair, and Overhaul", "abbreviation": "MRO",
         "definition": "The maintenance activities performed on aircraft to ensure airworthiness, ranging from line checks to complete engine overhauls."},
        {"term": "National Transportation Safety Board", "abbreviation": "NTSB",
         "definition": "Independent U.S. government agency responsible for investigating transportation accidents, determining probable cause, and issuing safety recommendations."},
        {"term": "Federal Aviation Administration", "abbreviation": "FAA",
         "definition": "U.S. government agency responsible for regulating civil aviation, including airspace management, pilot certification, and aircraft airworthiness."},
        {"term": "CAST/ICAO Common Taxonomy Team", "abbreviation": "CICTT",
         "definition": "International team that developed standard occurrence categories for classifying aviation accidents and incidents. The taxonomy used in this project."},
        {"term": "Human Factors Analysis and Classification System", "abbreviation": "HFACS",
         "definition": "Framework for analyzing human error in aviation accidents. Categorizes errors as skill-based, decision, perceptual, or violations."},
        {"term": "Terrain Awareness and Warning System", "abbreviation": "TAWS/GPWS",
         "definition": "Aircraft system that alerts pilots of potential controlled flight into terrain (CFIT) by comparing aircraft position with terrain databases. GPWS is the earlier generation."},
        {"term": "Air Traffic Control", "abbreviation": "ATC",
         "definition": "Ground-based service that directs aircraft on the ground and in the air. Provides separation between aircraft and traffic advisories."},
        {"term": "Loss of Control - In Flight", "abbreviation": "LOC-I",
         "definition": "Situation where the flight crew is unable to maintain control of the aircraft in flight, leading to an unrecoverable deviation from the intended flight path."},
        {"term": "Controlled Flight Into Terrain", "abbreviation": "CFIT",
         "definition": "Accident where an airworthy aircraft under pilot control is unintentionally flown into terrain, water, or an obstacle."},
        {"term": "Unintended Flight in IMC", "abbreviation": "UIMC",
         "definition": "Situation where a VFR-only pilot or aircraft inadvertently enters instrument meteorological conditions, often leading to spatial disorientation."},
        {"term": "Standard Operating Procedures", "abbreviation": "SOP",
         "definition": "Documented procedures that establish standard practices for aircraft operation. Designed to ensure safety through consistent crew actions."},
        {"term": "Minimum Descent Altitude", "abbreviation": "MDA",
         "definition": "The lowest altitude to which descent is authorized during a non-precision instrument approach without visual contact with the runway."},
        {"term": "Decision Altitude", "abbreviation": "DA",
         "definition": "A specified altitude on a precision approach at which a missed approach must be initiated if the required visual reference is not established."},
        {"term": "Upset Prevention and Recovery Training", "abbreviation": "UPRT",
         "definition": "Training program designed to reduce LOC-I accidents by teaching pilots to recognize, prevent, and recover from unusual aircraft attitudes."},
        {"term": "Flight Data Recorder", "abbreviation": "FDR",
         "definition": "Device installed in aircraft to record specific flight parameters. Used in accident investigation to reconstruct the sequence of events."},
    ]


def get_statistical_terms() -> list[dict]:
    """
    Hardcoded list of ~10 statistical methodology definitions.

    Each dict has keys: term, definition.
    """
    return [
        {"term": "Prevalence",
         "definition": "The proportion of reports in the dataset that contain a given category or feature. Expressed as a percentage of the accident-only population (n=431)."},
        {"term": "Co-occurrence",
         "definition": "When two or more accident categories appear together in the same report. Indicates cascading failure chains or multiple concurrent factors."},
        {"term": "Expected Calibration Error (ECE)",
         "definition": "A metric measuring how well predicted probabilities match actual outcomes. An ECE of 0.021 means predictions are on average 2.1 percentage points from reality."},
        {"term": "Bayesian Inference",
         "definition": "Statistical method that updates probability estimates as new evidence is observed. The risk model computes P(category | features) using Bayes' theorem."},
        {"term": "Calibration",
         "definition": "The degree to which predicted probabilities reflect actual frequencies. A well-calibrated model predicting 30% probability should be correct about 30% of the time."},
        {"term": "Laplace Smoothing",
         "definition": "Technique that adds a small count (alpha=1) to every observation to prevent zero-probability estimates for unseen feature combinations."},
        {"term": "Binary Relevance",
         "definition": "Multi-label classification approach where each category gets an independent binary classifier. Allows multiple categories to have high probability simultaneously."},
        {"term": "Leave-One-Out Cross-Validation (LOO-CV)",
         "definition": "Validation method where each report is held out one at a time, the model is retrained on the remaining data, and the held-out report is predicted. Provides unbiased performance estimates."},
        {"term": "Risk Ratio",
         "definition": "The rate of a condition in a specific category divided by the overall rate. A risk ratio > 1.0 indicates overrepresentation; e.g., 2.0 means twice as likely."},
        {"term": "Multi-label Classification",
         "definition": "Classification task where each report can belong to multiple categories simultaneously. Unlike single-label classification, probabilities do not sum to 1."},
    ]
