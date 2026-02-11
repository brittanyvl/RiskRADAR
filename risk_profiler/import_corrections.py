"""
Import manual corrections from validation review.

This script:
1. Loads corrections from JSON
2. Updates report_features with corrected aircraft
3. Populates report_aircraft table for multi-aircraft reports
4. Marks duplicate reports
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime


def import_corrections(conn, json_path: str, verbose: bool = True):
    """
    Import corrections from JSON file.
    """
    cursor = conn.cursor()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {
        "corrections_applied": 0,
        "multi_aircraft_added": 0,
        "duplicates_marked": 0,
        "errors": []
    }

    if verbose:
        print(f"Importing corrections from {json_path}")
        print(f"  Found {len(data.get('corrections', []))} corrections")
        print(f"  Found {len(data.get('multi_aircraft_reports', []))} multi-aircraft reports")
        print(f"  Found {len(data.get('duplicate_reports', []))} duplicate reports")
        print()

    # 1. Apply single-aircraft corrections
    if verbose:
        print("Applying corrections...")

    for correction in data.get("corrections", []):
        report_id = correction["report_id"]
        make = correction.get("make")
        model = correction.get("model")
        category = correction.get("category")
        notes = correction.get("notes", "")

        # Update report_features
        try:
            cursor.execute("""
                UPDATE report_features
                SET
                    aircraft_make = ?,
                    aircraft_model = ?,
                    aircraft_category = ?,
                    aircraft_confidence = 'manual',
                    extraction_notes = ?,
                    extracted_at = datetime('now')
                WHERE report_id = ?
            """, (make, model, category, notes, report_id))

            # Also insert into report_aircraft table
            cursor.execute("""
                INSERT OR REPLACE INTO report_aircraft
                (report_id, aircraft_sequence, aircraft_make, aircraft_model,
                 aircraft_category, source, confidence, extraction_notes)
                VALUES (?, 1, ?, ?, ?, 'manual', 'manual', ?)
            """, (report_id, make, model, category, notes))

            stats["corrections_applied"] += 1
        except Exception as e:
            stats["errors"].append(f"{report_id}: {e}")

    conn.commit()

    if verbose:
        print(f"  Applied {stats['corrections_applied']} corrections")

    # 2. Handle multi-aircraft reports
    if verbose:
        print("Processing multi-aircraft reports...")

    for multi in data.get("multi_aircraft_reports", []):
        report_id = multi["report_id"]
        notes = multi.get("notes", "")

        for aircraft in multi.get("aircraft", []):
            seq = aircraft.get("sequence", 1)
            make = aircraft.get("make")
            model = aircraft.get("model")
            category = aircraft.get("category")

            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO report_aircraft
                    (report_id, aircraft_sequence, aircraft_make, aircraft_model,
                     aircraft_category, source, confidence, extraction_notes)
                    VALUES (?, ?, ?, ?, ?, 'manual', 'manual', ?)
                """, (report_id, seq, make, model, category, notes))

                stats["multi_aircraft_added"] += 1
            except Exception as e:
                stats["errors"].append(f"{report_id} seq {seq}: {e}")

        # Update report_features with primary aircraft (sequence 1)
        primary = next((a for a in multi.get("aircraft", []) if a.get("sequence") == 1), None)
        if primary:
            cursor.execute("""
                UPDATE report_features
                SET
                    aircraft_make = ?,
                    aircraft_model = ?,
                    aircraft_category = ?,
                    aircraft_confidence = 'manual',
                    extraction_notes = ?
                WHERE report_id = ?
            """, (primary["make"], primary["model"], primary["category"],
                  f"Multi-aircraft: {notes}", report_id))

    conn.commit()

    if verbose:
        print(f"  Added {stats['multi_aircraft_added']} multi-aircraft entries")

    # 3. Mark duplicate reports
    if verbose:
        print("Marking duplicate reports...")

    for dup in data.get("duplicate_reports", []):
        report_id = dup["report_id"]
        related = dup.get("related_to")
        notes = dup.get("notes", "")

        try:
            cursor.execute("""
                UPDATE report_features
                SET
                    is_duplicate = 1,
                    related_report_id = ?,
                    duplicate_notes = ?
                WHERE report_id = ?
            """, (related, notes, report_id))

            stats["duplicates_marked"] += 1
        except Exception as e:
            stats["errors"].append(f"Duplicate {report_id}: {e}")

    conn.commit()

    if verbose:
        print(f"  Marked {stats['duplicates_marked']} duplicates")

    # Print errors if any
    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for err in stats["errors"][:10]:
            print(f"  - {err}")

    return stats


def verify_import(conn, verbose: bool = True):
    """
    Verify the import was successful.
    """
    cursor = conn.cursor()

    if verbose:
        print("\n" + "=" * 50)
        print("VERIFICATION")
        print("=" * 50)

    # Count report_features with manual confidence
    manual_count = cursor.execute("""
        SELECT COUNT(*) FROM report_features
        WHERE aircraft_confidence = 'manual'
    """).fetchone()[0]

    # Count report_aircraft entries
    aircraft_entries = cursor.execute("""
        SELECT COUNT(*) FROM report_aircraft
    """).fetchone()[0]

    # Count multi-aircraft reports
    multi_count = cursor.execute("""
        SELECT COUNT(DISTINCT report_id) FROM report_aircraft
        WHERE aircraft_sequence > 1
    """).fetchone()[0]

    # Count duplicates
    dup_count = cursor.execute("""
        SELECT COUNT(*) FROM report_features
        WHERE is_duplicate = 1
    """).fetchone()[0]

    # Total coverage
    total = cursor.execute("SELECT COUNT(*) FROM report_features").fetchone()[0]
    with_aircraft = cursor.execute("""
        SELECT COUNT(*) FROM report_features
        WHERE aircraft_category IS NOT NULL AND aircraft_category != ''
    """).fetchone()[0]

    if verbose:
        print(f"  Manual corrections applied: {manual_count}")
        print(f"  Total aircraft entries: {aircraft_entries}")
        print(f"  Multi-aircraft reports: {multi_count}")
        print(f"  Duplicate reports: {dup_count}")
        print(f"  Total coverage: {with_aircraft}/{total} ({with_aircraft/total*100:.1f}%)")

    return {
        "manual_count": manual_count,
        "aircraft_entries": aircraft_entries,
        "multi_count": multi_count,
        "dup_count": dup_count,
        "coverage": with_aircraft / total * 100
    }


if __name__ == "__main__":
    conn = sqlite3.connect("sqlite/riskradar.db")

    json_path = "risk_profiler/data/manual_aircraft_corrections.json"
    stats = import_corrections(conn, json_path, verbose=True)
    verify_import(conn, verbose=True)

    conn.close()
