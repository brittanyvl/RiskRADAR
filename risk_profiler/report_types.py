"""
risk_profiler/report_types.py
-----------------------------
Classify NTSB reports by type (accident, safety_study, recommendation, hazmat, unknown).

Uses filename prefix as the primary signal, with title heuristics as fallback
for ambiguous or unknown cases.
"""

import re
import sqlite3
from datetime import datetime, timezone


# Prefix -> (report_type, expects_taxonomy)
PREFIX_MAP = {
    "AAR": ("accident", 1),
    "AAB": ("accident", 1),
    "AIR": ("safety_study", 0),
    "SIR": ("safety_study", 0),
    "SS": ("safety_study", 0),
    "SAR": ("safety_study", 0),
    "ASR": ("recommendation", 0),
    "HZB": ("hazmat", 0),
    "HZMSR": ("hazmat", 0),
}


def _detect_supplemental(filename: str, title: str = "") -> dict | None:
    """
    Detect supplemental documents: appendices, summaries, reconsiderations,
    and cover/TOC pages that are not standalone accident reports.

    Returns classification dict if supplemental, None otherwise.
    """
    stem = filename.replace(".pdf", "")

    # Appendix files: AAR0003_app, AAR9804_C, AAR9804_D, etc.
    if re.search(r'_[A-Za-z]+$', stem):
        suffix = re.search(r'_([A-Za-z]+)$', stem).group(1).lower()
        # _body is the actual report content, not supplemental
        if suffix == "body":
            return None
        return {
            "report_type": "supplemental",
            "report_prefix": re.match(r'^([A-Z]+)', filename).group(1) if re.match(r'^([A-Z]+)', filename) else None,
            "classification_source": "filename_suffix",
            "expects_taxonomy": 0,
            "notes": f"Appendix/supplement (_{suffix} suffix)",
        }

    # Summary files: AAR8501S, AAR8601S, etc. (digit + S before .pdf)
    if re.search(r'\d{2}S\.pdf$', filename):
        return {
            "report_type": "supplemental",
            "report_prefix": re.match(r'^([A-Z]+)', filename).group(1) if re.match(r'^([A-Z]+)', filename) else None,
            "classification_source": "filename_suffix",
            "expects_taxonomy": 0,
            "notes": "Summary version (S suffix)",
        }

    # Reconsideration files: AAR9704r
    if re.search(r'\d{2}r\.pdf$', filename):
        return {
            "report_type": "supplemental",
            "report_prefix": re.match(r'^([A-Z]+)', filename).group(1) if re.match(r'^([A-Z]+)', filename) else None,
            "classification_source": "filename_suffix",
            "expects_taxonomy": 0,
            "notes": "Reconsideration response (r suffix)",
        }

    return None


def classify_single(filename: str, title: str = "", page_count: int = None) -> dict:
    """
    Classify a single report by filename prefix and title heuristics.

    Args:
        filename: Report filename (e.g., 'AAR7302.pdf')
        title: Report title for heuristic fallback
        page_count: Number of pages (used to detect cover/TOC files)

    Returns:
        Dict with keys: report_type, report_prefix, classification_source,
        expects_taxonomy, notes
    """
    # Check for supplemental documents first (appendices, summaries, etc.)
    supplemental = _detect_supplemental(filename, title)
    if supplemental is not None:
        return supplemental

    # Extract prefix from filename
    match = re.match(r'^([A-Z]+)', filename)
    prefix = match.group(1) if match else None

    # Cover/TOC detection: if a _body version exists, this is just a cover page
    # (handled by caller passing page_count - very short files with a _body sibling)

    # Try prefix lookup
    if prefix and prefix in PREFIX_MAP:
        report_type, expects_taxonomy = PREFIX_MAP[prefix]
        return {
            "report_type": report_type,
            "report_prefix": prefix,
            "classification_source": "prefix",
            "expects_taxonomy": expects_taxonomy,
            "notes": None,
        }

    # Title heuristics for ambiguous/unknown cases
    title_upper = (title or "").upper()

    if any(phrase in title_upper for phrase in ["SAFETY STUDY", "SPECIAL INVESTIGATION", "REVIEW OF"]):
        return {
            "report_type": "safety_study",
            "report_prefix": prefix,
            "classification_source": "title_heuristic",
            "expects_taxonomy": 0,
            "notes": "Classified via title heuristic",
        }

    if "RECOMMENDATION" in title_upper and title_upper.index("RECOMMENDATION") < len(title_upper) // 2:
        return {
            "report_type": "recommendation",
            "report_prefix": prefix,
            "classification_source": "title_heuristic",
            "expects_taxonomy": 0,
            "notes": "Classified via title heuristic",
        }

    # Unknown - flag for review but assume accident (expects taxonomy)
    return {
        "report_type": "unknown",
        "report_prefix": prefix,
        "classification_source": "unresolved",
        "expects_taxonomy": 1,
        "notes": "Unknown prefix, flagged for review",
    }


def classify_all_reports(conn: sqlite3.Connection, dry_run: bool = False) -> dict:
    """
    Classify all reports in the database by type.

    Args:
        conn: SQLite connection
        dry_run: If True, classify but don't write to DB

    Returns:
        Dict with classification summary stats
    """
    cursor = conn.execute("SELECT filename, title FROM reports")
    reports = cursor.fetchall()

    # Build set of filenames to detect _body siblings
    all_filenames = {r[0] for r in reports}

    # Get page counts for cover-page detection
    page_counts = {}
    for row in conn.execute("SELECT report_id, COUNT(*) FROM pages GROUP BY report_id"):
        page_counts[row[0]] = row[1]

    results = []
    for filename, title in reports:
        # Detect cover/TOC pages: very short file that has a _body sibling
        stem = filename.replace(".pdf", "")
        body_sibling = stem + "_body.pdf"
        if body_sibling in all_filenames and page_counts.get(filename, 0) <= 3:
            classification = {
                "report_type": "supplemental",
                "report_prefix": re.match(r'^([A-Z]+)', filename).group(1) if re.match(r'^([A-Z]+)', filename) else None,
                "classification_source": "cover_page",
                "expects_taxonomy": 0,
                "notes": f"Cover/TOC page ({page_counts.get(filename, '?')} pages, _body sibling exists)",
            }
        else:
            classification = classify_single(filename, title or "")
        classification["report_id"] = filename
        results.append(classification)

    # Summary
    type_counts = {}
    source_counts = {}
    for r in results:
        type_counts[r["report_type"]] = type_counts.get(r["report_type"], 0) + 1
        source_counts[r["classification_source"]] = source_counts.get(r["classification_source"], 0) + 1

    now = datetime.now(timezone.utc).isoformat()

    if not dry_run:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS report_types (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL,
                report_prefix TEXT,
                classification_source TEXT,
                expects_taxonomy INTEGER DEFAULT 1 CHECK(expects_taxonomy IN (0, 1)),
                notes TEXT,
                classified_at TEXT,
                FOREIGN KEY (report_id) REFERENCES reports(filename)
            )
        """)

        conn.executemany(
            """INSERT OR REPLACE INTO report_types
               (report_id, report_type, report_prefix, classification_source,
                expects_taxonomy, notes, classified_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    r["report_id"],
                    r["report_type"],
                    r["report_prefix"],
                    r["classification_source"],
                    r["expects_taxonomy"],
                    r["notes"],
                    now,
                )
                for r in results
            ],
        )
        conn.commit()

    # Print summary
    print("\n" + "=" * 60)
    print("REPORT TYPE CLASSIFICATION" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 60)
    print(f"Total reports: {len(results)}")
    print("\nBy type:")
    for rtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {rtype:20s} {count:>5}")
    print("\nBy source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source:20s} {count:>5}")
    print("=" * 60)

    return {
        "total": len(results),
        "type_counts": type_counts,
        "source_counts": source_counts,
        "dry_run": dry_run,
    }
