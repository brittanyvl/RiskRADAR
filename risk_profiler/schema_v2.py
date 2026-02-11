"""
Schema V2 updates for Risk Profiler.

Changes from V1:
1. New report_aircraft table (1:N relationship - multiple aircraft per report)
2. Add is_duplicate flag to report_features
3. Add related_report_id for duplicate tracking
"""

SCHEMA_VERSION = "2.0.0"

# New table for multiple aircraft per report
CREATE_REPORT_AIRCRAFT = """
CREATE TABLE IF NOT EXISTS report_aircraft (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    aircraft_sequence INTEGER DEFAULT 1,  -- 1 = primary, 2 = secondary (for collisions)

    -- Aircraft identification
    aircraft_make TEXT,
    aircraft_model TEXT,
    aircraft_model_series TEXT,           -- Full model like "727-200", "DHC-8-400"
    aircraft_category TEXT,               -- jet-wide, turboprop, etc.
    aircraft_registration TEXT,           -- N-number if available

    -- Extraction metadata
    source TEXT,                          -- 'title', 'abstract', 'body', 'manual'
    confidence TEXT,                      -- 'high', 'medium', 'low', 'manual'
    extraction_notes TEXT,

    -- Timestamps
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT,

    FOREIGN KEY (report_id) REFERENCES reports(filename),
    UNIQUE(report_id, aircraft_sequence)
);
"""

# Migration to add columns to report_features
ADD_DUPLICATE_COLUMNS = """
-- Add duplicate tracking columns (run separately, ignore if exists)
ALTER TABLE report_features ADD COLUMN is_duplicate INTEGER DEFAULT 0;
ALTER TABLE report_features ADD COLUMN related_report_id TEXT;
ALTER TABLE report_features ADD COLUMN duplicate_notes TEXT;
"""


def migrate_to_v2(conn, verbose=True):
    """
    Migrate database to V2 schema.
    """
    cursor = conn.cursor()

    if verbose:
        print("Migrating to schema V2...")

    # Create report_aircraft table
    try:
        cursor.execute(CREATE_REPORT_AIRCRAFT)
        if verbose:
            print("  Created report_aircraft table")
    except Exception as e:
        if verbose:
            print(f"  report_aircraft: {e}")

    # Add duplicate columns (may fail if already exists)
    for stmt in ADD_DUPLICATE_COLUMNS.strip().split(';'):
        if stmt.strip():
            try:
                cursor.execute(stmt)
            except Exception as e:
                pass  # Column may already exist

    if verbose:
        print("  Added duplicate tracking columns")

    conn.commit()

    # Create index for faster lookups
    try:
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_report_aircraft_report
            ON report_aircraft(report_id)
        """)
    except:
        pass

    conn.commit()

    if verbose:
        print("  Migration complete")

    return True


if __name__ == "__main__":
    import sqlite3
    conn = sqlite3.connect("sqlite/riskradar.db")
    migrate_to_v2(conn)
    conn.close()
