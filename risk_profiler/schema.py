"""
Database schema for Risk Profiler tables.

Tables created:
- report_features: Extracted features per report
- report_taxonomy: Migrated from Parquet (L1 + L2 combined)
- aircraft_lookup: Make/model → category mapping
- region_lookup: State → region mapping
- feature_extraction_runs: Track extraction pipeline runs
- feature_validation: Human review tracking
- extraction_coverage: Coverage metrics per run
"""

SCHEMA_VERSION = "1.0.0"

# SQL statements to create tables
CREATE_REPORT_FEATURES = """
CREATE TABLE IF NOT EXISTS report_features (
    report_id TEXT PRIMARY KEY,

    -- Aircraft features
    aircraft_raw TEXT,              -- Raw extracted text from title
    aircraft_make TEXT,             -- Normalized: "Boeing", "Cessna"
    aircraft_model TEXT,            -- Normalized: "747", "172"
    aircraft_category TEXT,         -- "jet-wide", "single-piston", etc.
    aircraft_confidence TEXT,       -- "high", "medium", "low", "manual"

    -- Location features
    location_raw TEXT,              -- Original location field
    state_code TEXT,                -- "FL", "NY", "AK"
    state_name TEXT,                -- "Florida", "New York"
    region TEXT,                    -- "South", "Northeast", "West"
    region_confidence TEXT,

    -- Temporal features
    accident_date TEXT,             -- From reports table
    year INTEGER,
    decade INTEGER,                 -- 1970, 1980, etc.
    month INTEGER,
    season TEXT,                    -- "Winter", "Spring", "Summer", "Fall"

    -- Time of day (extracted from text)
    time_of_day TEXT,               -- "Day", "Night", "Twilight", "Unknown"
    time_raw TEXT,                  -- Raw extracted time string
    time_confidence TEXT,

    -- Weather (extracted from text)
    weather_category TEXT,          -- "VMC", "IMC", "Icing", "Thunderstorm"
    weather_raw TEXT,               -- Raw extracted weather description
    weather_confidence TEXT,

    -- Metadata
    extraction_run_id INTEGER,
    extraction_version TEXT,
    extracted_at TEXT,
    validation_status TEXT,         -- "pending", "approved", "corrected"
    validated_by TEXT,
    validated_at TEXT,

    FOREIGN KEY (report_id) REFERENCES reports(filename)
);
"""

CREATE_REPORT_TAXONOMY = """
CREATE TABLE IF NOT EXISTS report_taxonomy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    level TEXT NOT NULL,            -- "L1" or "L2"
    category_code TEXT NOT NULL,
    category_name TEXT,
    parent_code TEXT,               -- For L2, the parent L1 code
    confidence REAL,
    rank INTEGER,                   -- Rank within report (1 = highest)
    source_run_id INTEGER,          -- Original taxonomy run ID

    FOREIGN KEY (report_id) REFERENCES reports(filename),
    UNIQUE(report_id, level, category_code)
);
"""

CREATE_AIRCRAFT_LOOKUP = """
CREATE TABLE IF NOT EXISTS aircraft_lookup (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Matching patterns
    pattern TEXT NOT NULL,          -- Regex or exact string to match
    pattern_type TEXT DEFAULT 'contains',  -- "exact", "contains", "regex"

    -- Normalized values
    make TEXT NOT NULL,             -- "Boeing", "Cessna", "Airbus"
    model TEXT,                     -- "747", "172", "A320"
    model_series TEXT,              -- "747-400", "172S"

    -- Category (based on FAA definitions)
    category TEXT NOT NULL,         -- See AIRCRAFT_CATEGORIES

    -- Metadata
    icao_code TEXT,                 -- ICAO type designator (e.g., "B744")
    source TEXT NOT NULL,           -- "FAA", "OpenFlights", "Manual"
    source_url TEXT,
    notes TEXT,

    UNIQUE(pattern, pattern_type)
);
"""

CREATE_REGION_LOOKUP = """
CREATE TABLE IF NOT EXISTS region_lookup (
    state_code TEXT PRIMARY KEY,    -- "FL", "NY", "AK"
    state_name TEXT NOT NULL,       -- "Florida", "New York"
    region TEXT NOT NULL,           -- US Census region
    division TEXT,                  -- US Census division (more granular)
    source TEXT DEFAULT 'US Census Bureau',

    CHECK(LENGTH(state_code) = 2)
);
"""

CREATE_FEATURE_EXTRACTION_RUNS = """
CREATE TABLE IF NOT EXISTS feature_extraction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,         -- "full", "aircraft", "time_of_day", etc.
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'running',  -- "running", "completed", "failed"

    -- Counts
    total_reports INTEGER,
    processed_reports INTEGER,
    successful_extractions INTEGER,
    failed_extractions INTEGER,

    -- Configuration
    extraction_version TEXT,
    config_json TEXT,               -- JSON of extraction parameters

    -- Error tracking
    error_message TEXT,

    notes TEXT
);
"""

CREATE_FEATURE_VALIDATION = """
CREATE TABLE IF NOT EXISTS feature_validation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    field_name TEXT NOT NULL,       -- "aircraft_category", "time_of_day", etc.

    -- Original extraction
    original_value TEXT,
    original_confidence TEXT,
    extraction_run_id INTEGER,

    -- Validation
    decision TEXT,                  -- "approve", "reject", "correct"
    corrected_value TEXT,
    validator_notes TEXT,

    validated_at TEXT,

    FOREIGN KEY (report_id) REFERENCES reports(filename),
    UNIQUE(report_id, field_name, extraction_run_id)
);
"""

CREATE_EXTRACTION_COVERAGE = """
CREATE TABLE IF NOT EXISTS extraction_coverage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    computed_at TEXT NOT NULL,

    -- Total reports
    total_reports INTEGER NOT NULL,

    -- Coverage by feature (count of non-null values)
    aircraft_make_count INTEGER,
    aircraft_category_count INTEGER,
    state_code_count INTEGER,
    region_count INTEGER,
    season_count INTEGER,
    time_of_day_count INTEGER,
    weather_category_count INTEGER,

    -- Coverage percentages
    aircraft_make_pct REAL,
    aircraft_category_pct REAL,
    state_code_pct REAL,
    region_pct REAL,
    season_pct REAL,
    time_of_day_pct REAL,
    weather_category_pct REAL,

    -- Confidence breakdown (count of high/medium/low)
    aircraft_high_conf INTEGER,
    aircraft_medium_conf INTEGER,
    aircraft_low_conf INTEGER,

    FOREIGN KEY (run_id) REFERENCES feature_extraction_runs(id)
);
"""

# Aircraft categories based on FAA definitions
AIRCRAFT_CATEGORIES = {
    "single-piston": "Single-engine piston aircraft",
    "multi-piston": "Multi-engine piston aircraft",
    "turboprop": "Turboprop aircraft",
    "jet-narrow": "Narrow-body jet aircraft",
    "jet-wide": "Wide-body jet aircraft",
    "jet-regional": "Regional jet aircraft",
    "helicopter": "Rotorcraft/Helicopter",
    "other": "Other aircraft types"
}

# US Census Bureau Regions (Public Domain)
US_CENSUS_REGIONS = {
    # Northeast
    "CT": ("Connecticut", "Northeast", "New England"),
    "ME": ("Maine", "Northeast", "New England"),
    "MA": ("Massachusetts", "Northeast", "New England"),
    "NH": ("New Hampshire", "Northeast", "New England"),
    "RI": ("Rhode Island", "Northeast", "New England"),
    "VT": ("Vermont", "Northeast", "New England"),
    "NJ": ("New Jersey", "Northeast", "Middle Atlantic"),
    "NY": ("New York", "Northeast", "Middle Atlantic"),
    "PA": ("Pennsylvania", "Northeast", "Middle Atlantic"),

    # Midwest
    "IL": ("Illinois", "Midwest", "East North Central"),
    "IN": ("Indiana", "Midwest", "East North Central"),
    "MI": ("Michigan", "Midwest", "East North Central"),
    "OH": ("Ohio", "Midwest", "East North Central"),
    "WI": ("Wisconsin", "Midwest", "East North Central"),
    "IA": ("Iowa", "Midwest", "West North Central"),
    "KS": ("Kansas", "Midwest", "West North Central"),
    "MN": ("Minnesota", "Midwest", "West North Central"),
    "MO": ("Missouri", "Midwest", "West North Central"),
    "NE": ("Nebraska", "Midwest", "West North Central"),
    "ND": ("North Dakota", "Midwest", "West North Central"),
    "SD": ("South Dakota", "Midwest", "West North Central"),

    # South
    "DE": ("Delaware", "South", "South Atlantic"),
    "FL": ("Florida", "South", "South Atlantic"),
    "GA": ("Georgia", "South", "South Atlantic"),
    "MD": ("Maryland", "South", "South Atlantic"),
    "NC": ("North Carolina", "South", "South Atlantic"),
    "SC": ("South Carolina", "South", "South Atlantic"),
    "VA": ("Virginia", "South", "South Atlantic"),
    "WV": ("West Virginia", "South", "South Atlantic"),
    "DC": ("District of Columbia", "South", "South Atlantic"),
    "AL": ("Alabama", "South", "East South Central"),
    "KY": ("Kentucky", "South", "East South Central"),
    "MS": ("Mississippi", "South", "East South Central"),
    "TN": ("Tennessee", "South", "East South Central"),
    "AR": ("Arkansas", "South", "West South Central"),
    "LA": ("Louisiana", "South", "West South Central"),
    "OK": ("Oklahoma", "South", "West South Central"),
    "TX": ("Texas", "South", "West South Central"),

    # West
    "AZ": ("Arizona", "West", "Mountain"),
    "CO": ("Colorado", "West", "Mountain"),
    "ID": ("Idaho", "West", "Mountain"),
    "MT": ("Montana", "West", "Mountain"),
    "NV": ("Nevada", "West", "Mountain"),
    "NM": ("New Mexico", "West", "Mountain"),
    "UT": ("Utah", "West", "Mountain"),
    "WY": ("Wyoming", "West", "Mountain"),
    "AK": ("Alaska", "West", "Pacific"),
    "CA": ("California", "West", "Pacific"),
    "HI": ("Hawaii", "West", "Pacific"),
    "OR": ("Oregon", "West", "Pacific"),
    "WA": ("Washington", "West", "Pacific"),

    # Territories
    "PR": ("Puerto Rico", "Territory", "Caribbean"),
    "VI": ("Virgin Islands", "Territory", "Caribbean"),
    "GU": ("Guam", "Territory", "Pacific Islands"),
    "AS": ("American Samoa", "Territory", "Pacific Islands"),
    "MP": ("Northern Mariana Islands", "Territory", "Pacific Islands"),
}

ALL_TABLES = [
    ("report_features", CREATE_REPORT_FEATURES),
    ("report_taxonomy", CREATE_REPORT_TAXONOMY),
    ("aircraft_lookup", CREATE_AIRCRAFT_LOOKUP),
    ("region_lookup", CREATE_REGION_LOOKUP),
    ("feature_extraction_runs", CREATE_FEATURE_EXTRACTION_RUNS),
    ("feature_validation", CREATE_FEATURE_VALIDATION),
    ("extraction_coverage", CREATE_EXTRACTION_COVERAGE),
]


def init_risk_profiler_tables(conn):
    """
    Initialize all risk profiler tables in the database.

    Args:
        conn: SQLite connection object

    Returns:
        List of created table names
    """
    cursor = conn.cursor()
    created = []

    for table_name, create_sql in ALL_TABLES:
        try:
            cursor.execute(create_sql)
            created.append(table_name)
        except Exception as e:
            print(f"Error creating {table_name}: {e}")
            raise

    conn.commit()
    return created


def populate_region_lookup(conn):
    """
    Populate the region_lookup table with US Census Bureau data.

    Source: US Census Bureau (Public Domain)
    """
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM region_lookup")

    # Insert all regions
    for state_code, (state_name, region, division) in US_CENSUS_REGIONS.items():
        cursor.execute("""
            INSERT INTO region_lookup (state_code, state_name, region, division, source)
            VALUES (?, ?, ?, ?, 'US Census Bureau')
        """, (state_code, state_name, region, division))

    conn.commit()
    return len(US_CENSUS_REGIONS)


def get_coverage_stats(conn, run_id=None):
    """
    Compute coverage statistics for extracted features.

    Args:
        conn: SQLite connection
        run_id: Optional run ID to filter by

    Returns:
        Dictionary with coverage metrics
    """
    cursor = conn.cursor()

    # Get total reports
    total = cursor.execute("SELECT COUNT(*) FROM reports").fetchone()[0]

    # Get feature counts
    stats = {"total_reports": total}

    feature_columns = [
        ("aircraft_make", "aircraft_make_count"),
        ("aircraft_category", "aircraft_category_count"),
        ("state_code", "state_code_count"),
        ("region", "region_count"),
        ("season", "season_count"),
        ("time_of_day", "time_of_day_count"),
        ("weather_category", "weather_category_count"),
    ]

    for col, stat_name in feature_columns:
        count = cursor.execute(f"""
            SELECT COUNT(*) FROM report_features
            WHERE {col} IS NOT NULL AND {col} != ''
        """).fetchone()[0]

        stats[stat_name] = count
        stats[stat_name.replace("_count", "_pct")] = round(count / total * 100, 1) if total > 0 else 0

    return stats


def print_coverage_report(conn):
    """Print a formatted coverage report."""
    stats = get_coverage_stats(conn)

    print("=" * 50)
    print("FEATURE EXTRACTION COVERAGE REPORT")
    print("=" * 50)
    print(f"Total reports: {stats['total_reports']}")
    print()

    features = [
        ("Aircraft Make", "aircraft_make"),
        ("Aircraft Category", "aircraft_category"),
        ("State", "state_code"),
        ("Region", "region"),
        ("Season", "season"),
        ("Time of Day", "time_of_day"),
        ("Weather", "weather_category"),
    ]

    for label, key in features:
        count = stats.get(f"{key}_count", 0)
        pct = stats.get(f"{key}_pct", 0)
        bar = "#" * int(pct / 2)
        print(f"{label:20s}: {count:3d} / {stats['total_reports']} ({pct:5.1f}%) {bar}")
