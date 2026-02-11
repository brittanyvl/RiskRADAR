"""
Weather condition extraction from NTSB report chunks.

Classifies each report as VMC (Visual Meteorological Conditions)
or IMC (Instrument Meteorological Conditions) by scanning chunk text
from the JSONL file.

Uses section-priority scanning and regex patterns with confidence levels.
"""

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from riskradar.config import PROJECT_ROOT


# Default JSONL path for chunk text
DEFAULT_JSONL_PATH = PROJECT_ROOT / "extraction" / "json_data" / "chunks.jsonl"

# Section priority order (higher priority = scanned first)
SECTION_PRIORITY = {
    "METEOROLOGICAL INFORMATION": 0,
    "METEOROLOGICAL": 0,
    "WEATHER": 0,
    "SYNOPSIS": 1,
    "PROBABLE CAUSE": 2,
    "ANALYSIS": 2,
    "FACTUAL INFORMATION": 3,
}

# --- Regex patterns with confidence and category ---

# High confidence VMC
HIGH_VMC_PATTERNS = [
    re.compile(r"\bV\.?M\.?C\.?\b"),
    re.compile(r"\bvisual meteorological conditions?\b", re.IGNORECASE),
    re.compile(r"\bVFR\s+(?:conditions?|weather)\b", re.IGNORECASE),
]

# High confidence IMC
HIGH_IMC_PATTERNS = [
    re.compile(r"\bI\.?M\.?C\.?\b"),
    re.compile(r"\binstrument meteorological conditions?\b", re.IGNORECASE),
    re.compile(r"\bIFR\s+(?:conditions?|weather)\b", re.IGNORECASE),
]

# Medium confidence IMC (inferred from conditions)
MEDIUM_IMC_PATTERNS = [
    re.compile(r"\bfog\b", re.IGNORECASE),
    re.compile(
        r"\bvisibility\s+(?:was\s+)?(?:less than|below|reduced\s+to)\s+\d",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bceiling\s+(?:was\s+)?(?:at\s+)?\d{2,3}\s+feet",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:overcast|obscured)\b.*\b(?:at|ceiling)\s+\d{2,3}\b",
        re.IGNORECASE,
    ),
]

# Medium confidence VMC (inferred from conditions)
MEDIUM_VMC_PATTERNS = [
    re.compile(
        r"\bclear\b.*\bvisibility\b.*\b(?:\d{2,}|unlimited)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bvisibility\s+(?:was\s+)?(?:\d{2,}|unlimited)\b",
        re.IGNORECASE,
    ),
]


def _get_section_priority(section_name: Optional[str]) -> int:
    """Return priority rank for a section name (lower = higher priority)."""
    if not section_name:
        return 99

    upper = section_name.upper().strip()
    for key, priority in SECTION_PRIORITY.items():
        if key in upper:
            return priority
    return 99


def _extract_snippet(text: str, match: re.Match, max_len: int = 100) -> str:
    """Extract a text snippet around a regex match, up to max_len chars."""
    start = max(0, match.start() - 30)
    end = min(len(text), match.end() + 30)
    snippet = text[start:end].strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len]
    return snippet


def _determine_confidence(section_priority: int, pattern_tier: str) -> str:
    """
    Determine confidence level based on section and pattern tier.

    Args:
        section_priority: 0=METEO, 1=SYNOPSIS, 2=ANALYSIS, 3+=other
        pattern_tier: 'high' or 'medium'
    """
    if pattern_tier == "high" and section_priority <= 0:
        return "high"
    if pattern_tier == "high" and section_priority <= 1:
        return "medium"
    if pattern_tier == "medium" and section_priority <= 0:
        return "medium"
    return "low"


def load_chunks_from_jsonl(
    jsonl_path: Optional[Path] = None,
) -> Dict[str, List[dict]]:
    """
    Load chunks from JSONL, grouped by report_id.

    Args:
        jsonl_path: Path to chunks JSONL file. Defaults to chunks_v2.jsonl.

    Returns:
        Dict mapping report_id -> list of chunk dicts
    """
    if jsonl_path is None:
        jsonl_path = DEFAULT_JSONL_PATH

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    chunks_by_report: Dict[str, List[dict]] = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            report_id = chunk.get("report_id")
            if report_id:
                chunks_by_report[report_id].append(chunk)

    return dict(chunks_by_report)


def extract_weather_for_report(
    report_id: str, chunks: List[dict]
) -> Optional[Dict]:
    """
    Scan chunk text to classify weather as VMC, IMC, or Unknown.

    Scans sections in priority order. First match from the highest-tier
    pattern wins.

    Args:
        report_id: The report identifier
        chunks: List of chunk dicts with section_name and chunk_text

    Returns:
        Dict with weather_category, weather_raw, weather_confidence
        or None if no weather info found
    """
    # Sort chunks by section priority
    sorted_chunks = sorted(
        chunks,
        key=lambda c: _get_section_priority(c.get("section_name")),
    )

    # Try patterns in priority order: high VMC/IMC first, then medium
    pattern_groups = [
        ("VMC", "high", HIGH_VMC_PATTERNS),
        ("IMC", "high", HIGH_IMC_PATTERNS),
        ("IMC", "medium", MEDIUM_IMC_PATTERNS),
        ("VMC", "medium", MEDIUM_VMC_PATTERNS),
    ]

    best_result = None
    best_confidence_rank = 999  # lower = better

    confidence_ranks = {"high": 0, "medium": 1, "low": 2}

    for chunk in sorted_chunks:
        text = chunk.get("chunk_text", "")
        if not text:
            continue

        section_priority = _get_section_priority(chunk.get("section_name"))

        for category, tier, patterns in pattern_groups:
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    confidence = _determine_confidence(section_priority, tier)
                    conf_rank = confidence_ranks.get(confidence, 99)

                    if conf_rank < best_confidence_rank:
                        best_confidence_rank = conf_rank
                        best_result = {
                            "weather_category": category,
                            "weather_raw": _extract_snippet(text, match),
                            "weather_confidence": confidence,
                        }

                    # If we found a high-confidence match, return immediately
                    if confidence == "high":
                        return best_result

    return best_result


def run_weather_extraction(
    conn: sqlite3.Connection,
    jsonl_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run weather extraction pipeline on all reports.

    Loads chunks from JSONL, extracts VMC/IMC classification,
    and updates report_features table.

    Args:
        conn: SQLite connection to riskradar.db
        jsonl_path: Path to chunks JSONL (default: chunks_v2.jsonl)
        verbose: Print progress messages

    Returns:
        Dict with extraction statistics
    """
    cursor = conn.cursor()

    # Create a run record
    cursor.execute("""
        INSERT INTO feature_extraction_runs
        (run_type, started_at, status, extraction_version)
        VALUES ('weather', datetime('now'), 'running', '1.0')
    """)
    run_id = cursor.lastrowid
    conn.commit()

    if verbose:
        print(f"Starting weather extraction run {run_id}...")

    # Load chunks from JSONL
    if verbose:
        print("  Loading chunks from JSONL...")
    chunks_by_report = load_chunks_from_jsonl(jsonl_path)
    if verbose:
        print(f"  Loaded chunks for {len(chunks_by_report)} reports")

    # Get all report_ids from report_features
    report_ids = [
        row[0]
        for row in cursor.execute(
            "SELECT report_id FROM report_features"
        ).fetchall()
    ]

    total = len(report_ids)
    vmc_count = 0
    imc_count = 0
    unknown_count = 0
    no_chunks_count = 0

    if verbose:
        print(f"  Processing {total} reports...")

    for i, report_id in enumerate(report_ids):
        chunks = chunks_by_report.get(report_id, [])
        if not chunks:
            no_chunks_count += 1
            continue

        result = extract_weather_for_report(report_id, chunks)

        if result:
            cursor.execute("""
                UPDATE report_features
                SET
                    weather_category = ?,
                    weather_raw = ?,
                    weather_confidence = ?,
                    extraction_run_id = ?,
                    extracted_at = datetime('now')
                WHERE report_id = ?
            """, (
                result["weather_category"],
                result["weather_raw"],
                result["weather_confidence"],
                run_id,
                report_id,
            ))

            if result["weather_category"] == "VMC":
                vmc_count += 1
            elif result["weather_category"] == "IMC":
                imc_count += 1
        else:
            unknown_count += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{total}...")

    conn.commit()

    # Update run record
    successful = vmc_count + imc_count
    cursor.execute("""
        UPDATE feature_extraction_runs
        SET
            completed_at = datetime('now'),
            status = 'completed',
            total_reports = ?,
            processed_reports = ?,
            successful_extractions = ?
        WHERE id = ?
    """, (total, total - no_chunks_count, successful, run_id))
    conn.commit()

    stats = {
        "run_id": run_id,
        "total": total,
        "vmc": vmc_count,
        "imc": imc_count,
        "unknown": unknown_count,
        "no_chunks": no_chunks_count,
    }

    if verbose:
        print(f"\n=== Weather Extraction Complete (Run {run_id}) ===")
        print(f"  Total reports:    {total}")
        print(f"  VMC classified:   {vmc_count} ({vmc_count/total*100:.1f}%)")
        print(f"  IMC classified:   {imc_count} ({imc_count/total*100:.1f}%)")
        print(f"  Unknown:          {unknown_count} ({unknown_count/total*100:.1f}%)")
        print(f"  No chunks found:  {no_chunks_count}")

    return stats


if __name__ == "__main__":
    conn = sqlite3.connect(str(PROJECT_ROOT / "sqlite" / "riskradar.db"))
    try:
        run_weather_extraction(conn, verbose=True)
    finally:
        conn.close()
