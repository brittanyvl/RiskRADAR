"""
riskradar/config.py
-------------------
Shared configuration for all RiskRADAR modules.

Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path

# Project root (parent of this file's directory)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# NAS path for PDF storage
NAS_PATH = Path(os.getenv("RISKRADAR_NAS_PATH", r"\\TRUENAS\Photos\RiskRADAR"))

# SQLite database path (inside sqlite/ directory)
DB_PATH = Path(os.getenv("RISKRADAR_DB_PATH", PROJECT_ROOT / "sqlite" / "riskradar.db"))

# Log directory
LOG_DIR = Path(os.getenv("RISKRADAR_LOG_DIR", PROJECT_ROOT / "logs"))

# NTSB Reports URL
NTSB_REPORTS_URL = "https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx"
