"""
RiskRADAR Analytics Engine

Analytics layer for querying structured aviation accident data.

Sub-packages:
    analytics.queries    Reusable query modules (shared, fleet_safety,
                         underwriting, operational_risk, glossary_data)

Legacy DuckDB tools (for chunks/pages exploration):
    analytics.convert    Convert JSONL to Parquet
    analytics.cli        Interactive SQL shell

Usage:
    # Convert JSONL to Parquet
    python -m analytics.convert

    # Launch interactive SQL shell
    python -m analytics.cli

    # Run a specific query
    python -m analytics.cli --query "SELECT COUNT(*) FROM chunks"
"""

__version__ = "1.0.0"
