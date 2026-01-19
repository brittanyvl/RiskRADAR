# scraper - Web Scraping Library

A clean, composable Selenium wrapper for web scraping with built-in rate limiting and download handling.

---

## Table of Contents

- [Overview](#overview)
- [Role in Pipeline](#role-in-pipeline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Usage Patterns](#usage-patterns)
- [Rate Limiting](#rate-limiting)
- [Limitations](#limitations)

---

## Overview

The `scraper` module provides:

- **Browser configuration and setup** with headless mode support
- **Page navigation and form actions** with automatic waiting
- **Download handling** with file completion detection
- **Rate limiting** with jitter to avoid predictable patterns
- **Context managers** for clean resource management

This library handles repetitive Selenium boilerplate while keeping site-specific logic in your application code.

---

## Role in Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Phase 1: Scraping                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    NTSB Website ──────► scraper library ──────► NAS Storage             │
│                              │                      │                    │
│                              │                      │                    │
│                              ▼                      ▼                    │
│                         Rate Limiting          510 PDFs                  │
│                         (2-3s delays)         (metadata)                 │
│                                                     │                    │
│                                                     ▼                    │
│                                                SQLite DB                 │
│                                              (reports table)             │
└─────────────────────────────────────────────────────────────────────────┘
```

The scraper library was used to:
1. Navigate NTSB's paginated report listings
2. Extract metadata from report pages
3. Download 510 aviation accident PDFs
4. Store metadata in SQLite for downstream processing

---

## Installation

From the repository root:

```bash
pip install -e ./scraper
```

---

## Quick Start

```python
from pathlib import Path
from selenium.webdriver.common.by import By

from scraper.config import BrowserConfig
from scraper.browser import chrome
from scraper.actions import go_to, select_dropdown_by_value, click
from scraper.download import wait_for_new_download, move_and_rename

URL = "https://example.com/reports"
DOWNLOAD_BUTTON = (By.ID, "download-btn")

def download_report(report_id: str) -> Path:
    config = BrowserConfig()

    with chrome(config) as driver:
        go_to(driver, f"{URL}/{report_id}")
        click(driver, DOWNLOAD_BUTTON)
        downloaded_file = wait_for_new_download(config)

    return move_and_rename(
        downloaded_file,
        target_dir=Path("/storage/reports"),
        new_name=f"{report_id}.pdf"
    )
```

---

## API Reference

### Browser Configuration

```python
from scraper.config import BrowserConfig

# Default configuration (reads from environment)
config = BrowserConfig()

# Custom configuration
config = BrowserConfig(
    headless=False,                          # Show browser window
    downloads_dir=Path("/custom/downloads"), # Download location
    user_agent="CustomBot/1.0",              # User agent string
    timeout_secs=30,                         # Page load timeout
    request_delay_sec=2.0,                   # Delay between requests
    download_delay_sec=3.0,                  # Delay between downloads
    page_delay_sec=2.0                       # Delay between pages
)
```

### Browser Context Manager

```python
from scraper.browser import chrome

with chrome(config) as driver:
    # driver is a configured Chrome WebDriver
    # Automatically closes when context exits
    pass
```

### Page Actions

```python
from scraper.actions import go_to, select_dropdown_by_value, click

# Navigate to URL (waits for page load)
go_to(driver, "https://example.com")

# Select dropdown by value
select_dropdown_by_value(driver, (By.NAME, "dropdown"), "value")

# Click element (waits for clickable)
click(driver, (By.ID, "button"))
```

### Download Handling

```python
from scraper.download import wait_for_new_download, move_and_rename

# Wait for download to complete (blocks until file stable)
downloaded_file = wait_for_new_download(config)

# Move and rename with atomic operation
final_path = move_and_rename(
    downloaded_file,
    target_dir=Path("/target"),
    new_name="report.pdf"
)
```

### Rate Limiting

```python
from scraper.waits import rate_limit

# Add delay with random jitter
rate_limit(2.0)  # 2 seconds +/- random jitter

# Use config delays
rate_limit(config.request_delay_sec)
rate_limit(config.download_delay_sec)
rate_limit(config.page_delay_sec)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRAPER_HEADLESS` | `1` | Run browser in headless mode (0 for visible) |
| `SCRAPER_DOWNLOADS_DIR` | `.scraper_tmp_downloads` | Temporary download directory |
| `SCRAPER_DOWNLOAD_TIMEOUT` | `120` | Download timeout in seconds |
| `HTTP_USER_AGENT` | `RiskRADARBot/1.0` | Browser user agent string |
| `SCRAPER_PAGELOAD_TIMEOUT` | `60` | Page load timeout in seconds |
| `SCRAPER_REQUEST_DELAY` | `2.0` | Delay between requests (seconds) |
| `SCRAPER_DOWNLOAD_DELAY` | `3.0` | Delay between downloads (seconds) |
| `SCRAPER_PAGE_DELAY` | `2.0` | Delay between page navigations |

---

## Usage Patterns

### Simple Form and Download

```python
with chrome(config) as driver:
    go_to(driver, url)
    select_dropdown_by_value(driver, dropdown_locator, value)
    click(driver, export_button)
    downloaded = wait_for_new_download(config)
    move_and_rename(downloaded, target_dir, filename)
```

### Paginated Scraping

```python
with chrome(config) as driver:
    go_to(driver, base_url)

    while has_next_page(driver):
        # Process current page
        for item in get_items(driver):
            process_item(item)
            rate_limit(config.request_delay_sec)

        # Navigate to next page
        click(driver, next_button)
        rate_limit(config.page_delay_sec)
```

### Multiple Downloads

```python
with chrome(config) as driver:
    for report_id in report_ids:
        go_to(driver, f"{base_url}/{report_id}")
        click(driver, download_button)
        downloaded = wait_for_new_download(config)
        move_and_rename(downloaded, storage_dir, f"{report_id}.pdf")
        rate_limit(config.download_delay_sec)
```

---

## Rate Limiting

**Always use rate limiting** to be respectful to target servers. The library provides configurable delays with random jitter to avoid predictable patterns.

### robots.txt Compliance

Before scraping, verify the target path is allowed:

```bash
curl https://example.com/robots.txt
```

For NTSB (the target of this project):
- `/investigations/AccidentReports/` is **ALLOWED**
- No `Crawl-delay` specified, but we enforce 2-3 second delays

### Recommended Delays

| Operation | Minimum Delay | Default |
|-----------|---------------|---------|
| Between requests | 1.0s | 2.0s |
| Between downloads | 2.0s | 3.0s |
| Between page navigations | 1.0s | 2.0s |

---

## Limitations

1. **Chrome Only**: Currently only supports ChromeDriver. Firefox/Safari not implemented.

2. **Single Session**: No connection pooling or session management. Each `chrome()` context is a fresh browser instance.

3. **No Proxy Support**: Proxy configuration not currently implemented.

4. **No JavaScript Rendering Wait**: Uses page load detection, not JavaScript completion. For SPAs, add explicit waits.

5. **Local Downloads Only**: Downloads go to local filesystem. No direct cloud storage upload.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | BrowserConfig dataclass and environment loading |
| `browser.py` | Chrome driver setup and context manager |
| `actions.py` | Page navigation, form actions, clicking |
| `download.py` | Download waiting and file operations |
| `waits.py` | Rate limiting and wait utilities |
| `setup.py` | Package installation configuration |
| `__init__.py` | Package exports |

---

## See Also

- [Main README](../README.md) - Project overview
- [riskradar/README.md](../riskradar/README.md) - Configuration
- [sqlite/README.md](../sqlite/README.md) - Database schema (reports table)
