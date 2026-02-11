"""
Time-of-day extraction for Risk Profiler.

Extracts time information from NTSB report chunk text and classifies
into time-of-day buckets: Morning, Afternoon, Evening, Night, Unknown.

Strategies:
  A. Keyword match (nighttime, daylight, etc.) - highest confidence
  B. Timestamp parsing (military, colon, 12-hour formats)

Section priority: SYNOPSIS > HISTORY OF FLIGHT > FACTUAL INFORMATION
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Time-of-day buckets
# ---------------------------------------------------------------------------

def classify_hour(hour: int) -> str:
    """
    Classify an integer hour (0-23) into a time-of-day bucket.

    | Category   | Hours (local) |
    |------------|---------------|
    | Morning    | 0500-1059     |
    | Afternoon  | 1100-1659     |
    | Evening    | 1700-2059     |
    | Night      | 2100-0459     |
    """
    if 5 <= hour <= 10:
        return "Morning"
    elif 11 <= hour <= 16:
        return "Afternoon"
    elif 17 <= hour <= 20:
        return "Evening"
    else:
        return "Night"


# ---------------------------------------------------------------------------
# State -> UTC offset (standard time)
# ---------------------------------------------------------------------------

STATE_TZ_OFFSET = {
    'ME': -5, 'NH': -5, 'VT': -5, 'MA': -5, 'RI': -5, 'CT': -5,
    'NY': -5, 'NJ': -5, 'PA': -5, 'DE': -5, 'MD': -5, 'DC': -5,
    'VA': -5, 'WV': -5, 'NC': -5, 'SC': -5, 'GA': -5, 'FL': -5,
    'OH': -5, 'MI': -5, 'IN': -5, 'KY': -5, 'TN': -5,
    'AL': -6, 'MS': -6, 'AR': -6, 'LA': -6, 'MO': -6,
    'WI': -6, 'IL': -6, 'MN': -6, 'IA': -6, 'ND': -6, 'SD': -6,
    'NE': -6, 'KS': -6, 'OK': -6, 'TX': -6,
    'MT': -7, 'WY': -7, 'CO': -7, 'NM': -7, 'AZ': -7,
    'UT': -7, 'ID': -7,
    'WA': -8, 'OR': -8, 'CA': -8, 'NV': -8,
    'AK': -9, 'HI': -10,
}

# Timezone abbreviation -> UTC offset
TZ_ABBREV_OFFSET = {
    'EST': -5, 'EDT': -4,
    'CST': -6, 'CDT': -5,
    'MST': -7, 'MDT': -6,
    'PST': -8, 'PDT': -7,
    'AKST': -9, 'AKDT': -8,
    'HST': -10,
    'UTC': 0, 'Z': 0,
}


# ---------------------------------------------------------------------------
# Keyword patterns (Strategy A)
# ---------------------------------------------------------------------------

NIGHT_KEYWORDS = [
    r'\bnighttime\b',
    r'\bafter dark\b',
    r'\bdarkness\b',
    r'\bnight\s+conditions\b',
    r'\bnight\s+visual\b',
]

DAY_KEYWORDS = [
    r'\bdaylight\b',
    r'\bdaytime\b',
    r'\bduring the day\b',
]


# ---------------------------------------------------------------------------
# Timestamp patterns (Strategy B)
# ---------------------------------------------------------------------------

# Negative lookbehind/lookahead to avoid matching altitudes, years, etc.
# Don't match if preceded by "altitude", "FL", or followed by "feet", "ft", "MSL", "AGL"
_NOT_ALTITUDE_BEFORE = r'(?<!altitude\s)(?<!FL)(?<!FL\s)'
_NOT_ALTITUDE_AFTER = r'(?!\s*(?:feet|ft|MSL|AGL|msl|agl))'

# Military time: "about 0830 EDT", "at 1445 CST", "approximately 2130 local"
TIME_MIL = re.compile(
    r'(?:about|at|approximately|around)\s+'
    + _NOT_ALTITUDE_BEFORE
    + r'(\d{4})'
    + _NOT_ALTITUDE_AFTER
    + r'\s*(EDT|EST|CDT|CST|PDT|PST|MDT|MST|AKDT|AKST|HST|local|UTC|Z|hours?)?',
    re.IGNORECASE,
)

# Colon military: "14:30", "08:15 EST"
TIME_COLON_MIL = re.compile(
    _NOT_ALTITUDE_BEFORE
    + r'\b(\d{1,2}):(\d{2})'
    + _NOT_ALTITUDE_AFTER
    + r'\s*(EDT|EST|CDT|CST|PDT|PST|MDT|MST|AKDT|AKST|HST|UTC|Z)?'
    + r'(?!\s*(?:a\.?m\.?|p\.?m\.?))',   # avoid double-matching 12h times
    re.IGNORECASE,
)

# 12-hour: "10:30 a.m.", "2:15 p.m."
TIME_12H = re.compile(
    r'(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)',
    re.IGNORECASE,
)

# Compiled keyword patterns
_NIGHT_RE = [re.compile(p, re.IGNORECASE) for p in NIGHT_KEYWORDS]
_DAY_RE = [re.compile(p, re.IGNORECASE) for p in DAY_KEYWORDS]


# ---------------------------------------------------------------------------
# Section priority ordering
# ---------------------------------------------------------------------------

SECTION_PRIORITY = [
    'synopsis',
    'history of flight',
    'factual information',
]


def _section_rank(section_name: Optional[str]) -> int:
    """Return priority rank for a section (lower = higher priority)."""
    if not section_name:
        return 99
    lower = section_name.strip().lower()
    for i, key in enumerate(SECTION_PRIORITY):
        if key in lower:
            return i
    return 99


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

def load_chunks_from_jsonl(
    jsonl_path: str = "extraction/json_data/chunks.jsonl",
) -> Dict[str, List[dict]]:
    """
    Load chunks from JSONL, grouped by report_id.

    Returns:
        Dict mapping report_id -> list of chunk dicts.
    """
    chunks_by_report: Dict[str, List[dict]] = {}
    path = Path(jsonl_path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            rid = chunk.get("report_id", "")
            chunks_by_report.setdefault(rid, []).append(chunk)
    return chunks_by_report


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _convert_utc_to_local(hour: int, minute: int, state_code: Optional[str]) -> int:
    """Convert a UTC hour to local hour using state timezone offset."""
    offset = STATE_TZ_OFFSET.get(state_code, -5) if state_code else -5
    local_hour = (hour + offset) % 24
    return local_hour


def _parse_military(text: str) -> Optional[tuple]:
    """Try to parse a 4-digit military time string. Returns (hour, minute) or None."""
    val = int(text)
    hour, minute = divmod(val, 100)
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return (hour, minute)
    return None


def _is_utc(tz_str: Optional[str]) -> bool:
    """Check if timezone string indicates UTC/Zulu."""
    if not tz_str:
        return False
    return tz_str.upper() in ('UTC', 'Z')


def _is_local_tz(tz_str: Optional[str]) -> bool:
    """Check if timezone string is an explicit local abbreviation."""
    if not tz_str:
        return False
    upper = tz_str.upper()
    return upper in TZ_ABBREV_OFFSET and upper not in ('UTC', 'Z')


def _try_timestamp_extraction(text: str, state_code: Optional[str]):
    """
    Try all timestamp patterns on a text block.

    Returns (hour_local, raw_match_text) or (None, None).
    """
    # Strategy B1: Military time ("about 0830 EDT")
    m = TIME_MIL.search(text)
    if m:
        parsed = _parse_military(m.group(1))
        if parsed:
            hour, minute = parsed
            tz_str = m.group(2)
            if _is_utc(tz_str):
                hour = _convert_utc_to_local(hour, minute, state_code)
            # If explicit local tz or "local" or "hours" or None -> treat as local
            return hour, m.group(0).strip()

    # Strategy B2: Colon military ("14:30 EST")
    m = TIME_COLON_MIL.search(text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            tz_str = m.group(3)
            if _is_utc(tz_str):
                hour = _convert_utc_to_local(hour, minute, state_code)
            return hour, m.group(0).strip()

    # Strategy B3: 12-hour time ("10:30 a.m.")
    m = TIME_12H.search(text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3).lower().replace('.', '')
        if 1 <= hour <= 12 and 0 <= minute <= 59:
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            return hour, m.group(0).strip()

    return None, None


def extract_time_for_report(
    report_id: str,
    chunks: List[dict],
    state_code: Optional[str] = None,
) -> Optional[Dict]:
    """
    Extract time-of-day from chunk text for a single report.

    Strategies (in order):
      A. Keyword match in priority sections (nighttime, daylight, etc.)
      B. Timestamp parsing in priority sections

    Returns:
        Dict with time_of_day, time_raw, time_confidence, or None.
    """
    # Sort chunks by section priority
    sorted_chunks = sorted(chunks, key=lambda c: _section_rank(c.get("section_name")))

    # --- Strategy A: Keyword match ---
    for chunk in sorted_chunks:
        text = chunk.get("chunk_text", "")
        if not text:
            continue
        section = chunk.get("section_name") or ""
        is_priority = _section_rank(section) < 99

        for pat in _NIGHT_RE:
            m = pat.search(text)
            if m:
                confidence = "high" if is_priority else "medium"
                return {
                    "time_of_day": "Night",
                    "time_raw": m.group(0).strip(),
                    "time_confidence": confidence,
                }

        for pat in _DAY_RE:
            m = pat.search(text)
            if m:
                # "daylight" alone doesn't specify Morning/Afternoon;
                # we'll fall through to timestamp if possible
                break
        else:
            continue
        break  # found a day keyword; try timestamp next for better specificity

    # --- Strategy B: Timestamp parsing ---
    for chunk in sorted_chunks:
        text = chunk.get("chunk_text", "")
        if not text:
            continue
        section = chunk.get("section_name") or ""
        is_priority = _section_rank(section) < 99

        hour, raw = _try_timestamp_extraction(text, state_code)
        if hour is not None:
            confidence = "high" if is_priority else "medium"
            return {
                "time_of_day": classify_hour(hour),
                "time_raw": raw,
                "time_confidence": confidence,
            }

    # --- Fallback: if we found a day keyword but no timestamp ---
    for chunk in sorted_chunks:
        text = chunk.get("chunk_text", "")
        if not text:
            continue
        section = chunk.get("section_name") or ""
        is_priority = _section_rank(section) < 99

        for pat in _DAY_RE:
            m = pat.search(text)
            if m:
                return {
                    "time_of_day": "Afternoon",  # best generic guess for "daylight"
                    "time_raw": m.group(0).strip(),
                    "time_confidence": "low",
                }

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_time_extraction(
    conn: sqlite3.Connection,
    jsonl_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run time-of-day extraction on all reports.

    Steps:
      1. Load chunks from JSONL
      2. Look up state_code per report from report_features
      3. Create a feature_extraction_runs record
      4. Extract time for each report
      5. UPDATE report_features SET time_of_day, time_raw, time_confidence
      6. Print summary stats

    Args:
        conn: SQLite connection to riskradar.db
        jsonl_path: Override path to chunks JSONL
        verbose: Print progress

    Returns:
        Dict with run_id and extraction stats.
    """
    if jsonl_path is None:
        jsonl_path = "extraction/json_data/chunks.jsonl"

    if verbose:
        print("Loading chunks from JSONL...")
    chunks_by_report = load_chunks_from_jsonl(jsonl_path)
    if verbose:
        print(f"  Loaded chunks for {len(chunks_by_report)} reports")

    cursor = conn.cursor()

    # Get state_code per report
    rows = cursor.execute(
        "SELECT report_id, state_code FROM report_features"
    ).fetchall()
    state_map = {r[0]: r[1] for r in rows}

    # Create run record
    cursor.execute("""
        INSERT INTO feature_extraction_runs
        (run_type, started_at, status, extraction_version)
        VALUES ('time_of_day', datetime('now'), 'running', '1.0')
    """)
    run_id = cursor.lastrowid
    conn.commit()

    if verbose:
        print(f"Starting time extraction run {run_id}...")

    total = 0
    extracted = 0
    bucket_counts: Dict[str, int] = {}
    confidence_counts: Dict[str, int] = {}

    for report_id, chunks in chunks_by_report.items():
        total += 1
        state_code = state_map.get(report_id)
        result = extract_time_for_report(report_id, chunks, state_code)

        if result:
            extracted += 1
            tod = result["time_of_day"]
            conf = result["time_confidence"]
            bucket_counts[tod] = bucket_counts.get(tod, 0) + 1
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

            cursor.execute("""
                UPDATE report_features
                SET time_of_day = ?,
                    time_raw = ?,
                    time_confidence = ?
                WHERE report_id = ?
            """, (
                result["time_of_day"],
                result["time_raw"],
                result["time_confidence"],
                report_id,
            ))

        if verbose and total % 100 == 0:
            print(f"  Processed {total} reports...")

    conn.commit()

    # Update run record
    cursor.execute("""
        UPDATE feature_extraction_runs
        SET completed_at = datetime('now'),
            status = 'completed',
            total_reports = ?,
            processed_reports = ?,
            successful_extractions = ?
        WHERE id = ?
    """, (total, total, extracted, run_id))
    conn.commit()

    stats = {
        "run_id": run_id,
        "total": total,
        "extracted": extracted,
        "unknown": total - extracted,
        "buckets": bucket_counts,
        "confidence": confidence_counts,
    }

    if verbose:
        pct = extracted / total * 100 if total else 0
        print(f"\n=== Time Extraction Complete (Run {run_id}) ===")
        print(f"  Total reports:  {total}")
        print(f"  Time extracted: {extracted} ({pct:.1f}%)")
        print(f"  Unknown:        {total - extracted}")
        print()
        print("  Bucket breakdown:")
        for bucket in ["Morning", "Afternoon", "Evening", "Night"]:
            cnt = bucket_counts.get(bucket, 0)
            print(f"    {bucket:12s}: {cnt:4d}")
        print()
        print("  Confidence breakdown:")
        for conf in ["high", "medium", "low"]:
            cnt = confidence_counts.get(conf, 0)
            print(f"    {conf:8s}: {cnt:4d}")

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conn = sqlite3.connect("sqlite/riskradar.db")
    try:
        run_time_extraction(conn, verbose=True)
    finally:
        conn.close()
