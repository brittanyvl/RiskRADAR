# riskradar - Core Configuration Module

Central configuration and shared utilities for all RiskRADAR components.

---

## Table of Contents

- [Overview](#overview)
- [Role in Pipeline](#role-in-pipeline)
- [Configuration Reference](#configuration-reference)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Limitations](#limitations)

---

## Overview

The `riskradar` module provides:

- **Centralized path configuration** for database, NAS storage, and logs
- **Environment variable management** with sensible defaults
- **Qdrant Cloud integration** with multi-source credential loading
- **Shared constants** used across all pipeline phases

This module ensures consistent configuration across all components and simplifies deployment by supporting environment-based configuration.

---

## Role in Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    riskradar/config.py                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Paths     │  │   Qdrant    │  │   Environment       │  │
│  │  DB_PATH    │  │   Config    │  │   Variables         │  │
│  │  NAS_PATH   │  │   (URL/Key) │  │   (.env support)    │  │
│  │  LOG_DIR    │  │             │  │                     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
    ┌──────────┐    ┌──────────┐         ┌──────────┐
    │ sqlite   │    │embeddings│         │ scraper  │
    │extraction│    │   eval   │         │analytics │
    └──────────┘    └──────────┘         └──────────┘
```

Every module imports from `riskradar.config` to ensure consistent paths and settings.

---

## Configuration Reference

### Path Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | Auto-detected | Absolute path to repository root |
| `DB_PATH` | `sqlite/riskradar.db` | SQLite database location |
| `NAS_PATH` | `\\TRUENAS\Photos\RiskRADAR` | PDF storage location |
| `LOG_DIR` | `logs/` | Pipeline log directory |
| `NTSB_REPORTS_URL` | NTSB website | Source URL for scraping |

### Qdrant Configuration

Retrieved via `get_qdrant_config()`:

| Key | Source | Description |
|-----|--------|-------------|
| `url` | `QDRANT_URL` | Qdrant Cloud cluster URL |
| `api_key` | `QDRANT_API_KEY` | API key for authentication |

---

## Environment Variables

All configuration can be overridden via environment variables or a `.env` file:

```bash
# Core Paths (optional - defaults usually work)
RISKRADAR_NAS_PATH=\\TRUENAS\Photos\RiskRADAR
RISKRADAR_DB_PATH=sqlite/riskradar.db
RISKRADAR_LOG_DIR=logs

# Qdrant Cloud (required for Phase 5+)
QDRANT_URL=https://your-cluster.region.cloud.qdrant.io:6333
QDRANT_API_KEY=your_api_key_here
```

### Loading Order

1. Environment variables (highest priority)
2. `.env` file in project root
3. Default values in code

---

## API Reference

### `get_qdrant_config() -> dict`

Retrieves Qdrant credentials from available sources.

**Returns:**
```python
{
    "url": "https://...",
    "api_key": "..."
}
```

**Raises:**
- `ValueError` if credentials not found in any source

**Source Priority:**
1. Environment variables (`QDRANT_URL`, `QDRANT_API_KEY`)
2. Streamlit secrets (`.streamlit/secrets.toml`)

**Example:**
```python
from riskradar.config import get_qdrant_config

try:
    config = get_qdrant_config()
    print(f"Qdrant URL: {config['url']}")
except ValueError as e:
    print(f"Qdrant not configured: {e}")
```

---

## Usage Examples

### Accessing Paths

```python
from riskradar.config import DB_PATH, NAS_PATH, LOG_DIR

# Database operations
print(f"Database: {DB_PATH}")  # sqlite/riskradar.db

# PDF storage
pdf_path = NAS_PATH / "AAR2001.pdf"

# Logging
log_file = LOG_DIR / "pipeline.log"
```

### Checking Configuration

```python
from riskradar.config import PROJECT_ROOT, DB_PATH, NAS_PATH

print(f"Project root: {PROJECT_ROOT}")
print(f"Database exists: {DB_PATH.exists()}")
print(f"NAS accessible: {NAS_PATH.exists()}")
```

### Qdrant Integration

```python
from riskradar.config import get_qdrant_config
from qdrant_client import QdrantClient

config = get_qdrant_config()
client = QdrantClient(
    url=config["url"],
    api_key=config["api_key"]
)
```

---

## Limitations

1. **NAS Path Assumption**: Default NAS path assumes Windows UNC path format. Linux/Mac users must override via environment variable.

2. **No Path Validation**: Paths are not validated at import time. Use `Path.exists()` to check accessibility.

3. **Single Qdrant Cluster**: Only one Qdrant configuration supported. Multiple environments (dev/prod) require environment variable switching.

4. **Blocking Credential Load**: `get_qdrant_config()` will raise immediately if credentials missing, not lazy-load.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All configuration and utilities |
| `__init__.py` | Package marker |

---

## See Also

- [Main README](../README.md) - Project overview
- [sqlite/README.md](../sqlite/README.md) - Database schema
- [embeddings/README.md](../embeddings/README.md) - Qdrant usage
