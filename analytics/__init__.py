"""
RiskRADAR Analytics Engine

DuckDB-based analytics layer for querying pages, documents, and chunks data.

Usage:
    # Convert JSONL to Parquet
    py -m analytics.convert

    # Launch interactive SQL shell
    py -m analytics.cli

    # Run a specific query
    py -m analytics.cli --query "SELECT COUNT(*) FROM chunks"
"""

__version__ = "1.0.0"
