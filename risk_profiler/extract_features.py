"""
Feature extraction pipeline for Risk Profiler.

Extracts:
- Aircraft category from report titles
- State/region from location field
- Updates coverage metrics
"""

import re
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Tuple

from .aircraft_data import lookup_aircraft


# State extraction patterns
# Match "City, ST" or "City, State" or just "ST" at end
STATE_PATTERNS = [
    # Two-letter state code after comma
    r",\s*([A-Z]{2})\s*$",
    # Two-letter state code in parentheses
    r"\(([A-Z]{2})\)",
    # Common "near City, State" pattern
    r"near\s+[\w\s]+,\s*([A-Z]{2})",
    # State name patterns (will be matched against region_lookup)
]


def extract_state_from_location(location: str, conn) -> Optional[Dict]:
    """
    Extract state code from location string.

    Args:
        location: Location string like "Miami, FL" or "near Dallas, Texas"
        conn: SQLite connection (for state name lookup)

    Returns:
        Dict with state_code, state_name, region, region_confidence or None
    """
    if not location:
        return None

    cursor = conn.cursor()

    # Try two-letter codes first
    for pattern in STATE_PATTERNS:
        match = re.search(pattern, location)
        if match:
            code = match.group(1).upper()
            # Validate against region_lookup
            result = cursor.execute("""
                SELECT state_code, state_name, region, division
                FROM region_lookup
                WHERE state_code = ?
            """, (code,)).fetchone()

            if result:
                return {
                    "state_code": result[0],
                    "state_name": result[1],
                    "region": result[2],
                    "division": result[3],
                    "confidence": "high"
                }

    # Try matching state names
    result = cursor.execute("""
        SELECT state_code, state_name, region, division
        FROM region_lookup
        WHERE ? LIKE '%' || state_name || '%'
        ORDER BY LENGTH(state_name) DESC
        LIMIT 1
    """, (location,)).fetchone()

    if result:
        return {
            "state_code": result[0],
            "state_name": result[1],
            "region": result[2],
            "division": result[3],
            "confidence": "medium"
        }

    return None


def extract_features_for_report(conn, report_id: str, title: str, location: str) -> Dict:
    """
    Extract all features for a single report.

    Returns dict with extracted values and confidence levels.
    """
    result = {
        "report_id": report_id,
        "aircraft_raw": None,
        "aircraft_make": None,
        "aircraft_model": None,
        "aircraft_category": None,
        "aircraft_confidence": None,
        "state_code": None,
        "state_name": None,
        "region": None,
        "region_confidence": None,
    }

    # Extract aircraft from title
    if title:
        aircraft = lookup_aircraft(conn, title)
        if aircraft:
            result["aircraft_raw"] = aircraft.get("pattern_matched")
            result["aircraft_make"] = aircraft.get("make")
            result["aircraft_model"] = aircraft.get("model")
            result["aircraft_category"] = aircraft.get("category")
            result["aircraft_confidence"] = aircraft.get("confidence")

    # Extract state/region from location
    if location:
        state = extract_state_from_location(location, conn)
        if state:
            result["state_code"] = state["state_code"]
            result["state_name"] = state["state_name"]
            result["region"] = state["region"]
            result["region_confidence"] = state["confidence"]

    return result


def run_feature_extraction(conn, verbose: bool = True) -> Dict:
    """
    Run feature extraction on all reports.

    Updates report_features table with aircraft and location features.

    Returns:
        Dict with extraction statistics
    """
    cursor = conn.cursor()

    # Start a run record
    cursor.execute("""
        INSERT INTO feature_extraction_runs
        (run_type, started_at, status, extraction_version)
        VALUES ('aircraft_location', datetime('now'), 'running', '1.0')
    """)
    run_id = cursor.lastrowid
    conn.commit()

    if verbose:
        print(f"Starting feature extraction run {run_id}...")

    # Get all reports with titles
    reports = cursor.execute("""
        SELECT r.filename, r.title, r.location
        FROM reports r
        JOIN report_features f ON r.filename = f.report_id
    """).fetchall()

    total = len(reports)
    aircraft_extracted = 0
    location_extracted = 0

    if verbose:
        print(f"  Processing {total} reports...")

    for i, (report_id, title, location) in enumerate(reports):
        features = extract_features_for_report(conn, report_id, title, location)

        # Update report_features
        cursor.execute("""
            UPDATE report_features
            SET
                aircraft_raw = ?,
                aircraft_make = ?,
                aircraft_model = ?,
                aircraft_category = ?,
                aircraft_confidence = ?,
                state_code = ?,
                state_name = ?,
                region = ?,
                region_confidence = ?,
                extraction_run_id = ?,
                extraction_version = '1.0',
                extracted_at = datetime('now')
            WHERE report_id = ?
        """, (
            features["aircraft_raw"],
            features["aircraft_make"],
            features["aircraft_model"],
            features["aircraft_category"],
            features["aircraft_confidence"],
            features["state_code"],
            features["state_name"],
            features["region"],
            features["region_confidence"],
            run_id,
            report_id
        ))

        if features["aircraft_category"]:
            aircraft_extracted += 1
        if features["region"]:
            location_extracted += 1

        # Progress update
        if verbose and (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{total}...")

    conn.commit()

    # Update run record
    cursor.execute("""
        UPDATE feature_extraction_runs
        SET
            completed_at = datetime('now'),
            status = 'completed',
            total_reports = ?,
            processed_reports = ?,
            successful_extractions = ?
        WHERE id = ?
    """, (total, total, aircraft_extracted + location_extracted, run_id))
    conn.commit()

    # Compute coverage
    stats = compute_coverage(conn, run_id)

    if verbose:
        print(f"\n=== Feature Extraction Complete (Run {run_id}) ===")
        print(f"  Total reports: {total}")
        print(f"  Aircraft extracted: {aircraft_extracted} ({aircraft_extracted/total*100:.1f}%)")
        print(f"  Location extracted: {location_extracted} ({location_extracted/total*100:.1f}%)")

    return {
        "run_id": run_id,
        "total": total,
        "aircraft_extracted": aircraft_extracted,
        "location_extracted": location_extracted,
        "stats": stats
    }


def compute_coverage(conn, run_id: int) -> Dict:
    """
    Compute and store coverage metrics for a run.
    """
    cursor = conn.cursor()

    # Get counts
    total = cursor.execute("SELECT COUNT(*) FROM report_features").fetchone()[0]

    counts = {}
    for col in ['aircraft_make', 'aircraft_category', 'state_code', 'region',
                'season', 'time_of_day', 'weather_category']:
        count = cursor.execute(f"""
            SELECT COUNT(*) FROM report_features
            WHERE {col} IS NOT NULL AND {col} != ''
        """).fetchone()[0]
        counts[col] = count

    # Aircraft confidence breakdown
    conf_counts = cursor.execute("""
        SELECT aircraft_confidence, COUNT(*)
        FROM report_features
        WHERE aircraft_confidence IS NOT NULL
        GROUP BY aircraft_confidence
    """).fetchall()
    conf_dict = {row[0]: row[1] for row in conf_counts}

    # Store coverage record
    cursor.execute("""
        INSERT INTO extraction_coverage (
            run_id, computed_at, total_reports,
            aircraft_make_count, aircraft_category_count,
            state_code_count, region_count,
            season_count, time_of_day_count, weather_category_count,
            aircraft_make_pct, aircraft_category_pct,
            state_code_pct, region_pct,
            season_pct, time_of_day_pct, weather_category_pct,
            aircraft_high_conf, aircraft_medium_conf, aircraft_low_conf
        ) VALUES (
            ?, datetime('now'), ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?
        )
    """, (
        run_id, total,
        counts.get('aircraft_make', 0), counts.get('aircraft_category', 0),
        counts.get('state_code', 0), counts.get('region', 0),
        counts.get('season', 0), counts.get('time_of_day', 0), counts.get('weather_category', 0),
        counts.get('aircraft_make', 0) / total * 100 if total > 0 else 0,
        counts.get('aircraft_category', 0) / total * 100 if total > 0 else 0,
        counts.get('state_code', 0) / total * 100 if total > 0 else 0,
        counts.get('region', 0) / total * 100 if total > 0 else 0,
        counts.get('season', 0) / total * 100 if total > 0 else 0,
        counts.get('time_of_day', 0) / total * 100 if total > 0 else 0,
        counts.get('weather_category', 0) / total * 100 if total > 0 else 0,
        conf_dict.get('high', 0), conf_dict.get('medium', 0), conf_dict.get('low', 0)
    ))
    conn.commit()

    return counts


if __name__ == "__main__":
    conn = sqlite3.connect("sqlite/riskradar.db")
    run_feature_extraction(conn, verbose=True)
    conn.close()
