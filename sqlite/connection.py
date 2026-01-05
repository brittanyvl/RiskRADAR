"""
sqlite/connection.py
--------------------
Database connection management for RiskRADAR.
"""
import sqlite3
from pathlib import Path

from .schema import SCHEMA_VERSION, ALL_TABLES, INDEXES, PHASE3_TABLES, PHASE3_INDEXES


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    """
    Get a connection to the SQLite database.

    Args:
        db_path: Path to the database file

    Returns:
        sqlite3.Connection with row_factory set for dict-like access
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enables column access by name
    return conn


def init_db(db_path: Path | str) -> sqlite3.Connection:
    """
    Initialize database with schema. Safe to call multiple times.

    Creates all tables if they don't exist and sets schema version.

    Args:
        db_path: Path to the database file

    Returns:
        Initialized sqlite3.Connection
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Check current schema version
    cursor.execute("PRAGMA user_version")
    current_version = cursor.fetchone()[0]

    if current_version == 0:
        # Fresh database - create all tables
        for table_sql in ALL_TABLES:
            cursor.execute(table_sql)

        for index_sql in INDEXES:
            cursor.execute(index_sql)

        cursor.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

    elif current_version < SCHEMA_VERSION:
        # Handle migrations
        if current_version == 1 and SCHEMA_VERSION >= 2:
            # Migration v1 → v2: Add extraction tables
            print(f"Migrating database from v{current_version} to v2...")
            for table_sql in PHASE3_TABLES:
                cursor.execute(table_sql)

            for index_sql in PHASE3_INDEXES:
                cursor.execute(index_sql)

            cursor.execute("PRAGMA user_version = 2")
            conn.commit()
            print("Migration complete!")

        # Future migrations can be added here
        # elif current_version == 2 and SCHEMA_VERSION >= 3:
        #     ...

    return conn
