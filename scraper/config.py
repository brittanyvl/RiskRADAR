"""
scraper/config.py
-----------------
Holds configuration dataclass for scraper runs.

- Centralizes Selenium/browser settings (headless, window size, user agent).
- Controls download directory, timeouts, and file stability checks.
- Values are read from environment variables (.env) with sensible defaults.

Use: instantiate `BrowserConfig()` in your scraping scripts.
"""
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class BrowserConfig:
    headless: bool = bool(os.getenv("SCRAPER_HEADLESS", "1") == "1")
    user_agent: str = os.getenv("HTTP_USER_AGENT", "RiskRADARBot/1.0 (+https://github.com/)")
    window_width: int = int(os.getenv("SCRAPER_WIN_W", "1400"))
    window_height: int = int(os.getenv("SCRAPER_WIN_H", "900"))
    downloads_dir: Path = Path(os.getenv("SCRAPER_DOWNLOADS_DIR", ".scraper_tmp_downloads")).resolve()
    page_load_timeout_sec: int = int(os.getenv("SCRAPER_PAGELOAD_TIMEOUT", "60"))
    implicit_wait_sec: int = int(os.getenv("SCRAPER_IMPLICIT_WAIT", "0"))
    partial_suffixes: tuple[str, ...] = (".crdownload", ".part", ".tmp")
    download_timeout_sec: int = int(os.getenv("SCRAPER_DOWNLOAD_TIMEOUT", "120"))
    stable_check_interval_sec: float = float(os.getenv("SCRAPER_STABLE_CHECK_INTERVAL", "0.5"))
