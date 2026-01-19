"""
riskradar/config.py
-------------------
Shared configuration for all RiskRADAR modules.

Loads settings from environment variables with sensible defaults.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()

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

# ============================================================================
# Phase 5: Qdrant Configuration
# ============================================================================

def get_qdrant_config() -> dict:
    """
    Get Qdrant configuration from environment or Streamlit secrets.

    Tries in order:
    1. Environment variables (QDRANT_URL, QDRANT_API_KEY)
    2. Streamlit secrets (if running in Streamlit)

    Returns:
        Dict with keys: url, api_key

    Raises:
        ValueError: If Qdrant config not found in any source
    """
    # Try environment variables first
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if url and api_key:
        return {"url": url, "api_key": api_key}

    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            url = st.secrets.get("QDRANT_URL")
            api_key = st.secrets.get("QDRANT_API_KEY")
            if url and api_key:
                return {"url": url, "api_key": api_key}
    except ImportError:
        logger.debug("Streamlit not installed, skipping Streamlit secrets")
    except Exception as e:
        logger.debug(f"Could not access Streamlit secrets: {e}")

    raise ValueError(
        "Qdrant configuration not found. Set QDRANT_URL and QDRANT_API_KEY "
        "environment variables or add to .streamlit/secrets.toml"
    )
