"""
scraper/browser.py
------------------
Context manager to create and tear down a Selenium ChromeDriver
with sane defaults.

- Configures Chrome with download directory, user agent, headless mode, etc.
- Yields a `driver` object for use in scraping scripts.
- Ensures proper cleanup with `driver.quit()` after use.

Use:
    with chrome(BrowserConfig()) as driver:
        driver.get("...")
"""
from __future__ import annotations
from contextlib import contextmanager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import Options
from webdriver_manager.chrome import ChromeDriverManager
from .config import BrowserConfig

def _chrome_options(cfg: BrowserConfig) -> Options:
    opts = Options()
    if cfg.headless:
        opts.add_argument("--headless=new")
    opts.add_argument(f"--window-size={cfg.window_width},{cfg.window_height}")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"--user-agent={cfg.user_agent}")
    prefs = {
        "download.default_directory": str(cfg.downloads_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    opts.add_experimental_option("prefs", prefs)
    return opts

@contextmanager
def chrome(cfg: BrowserConfig):
    cfg.downloads_dir.mkdir(parents=True, exist_ok=True)
    options = _chrome_options(cfg)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                              options=options)
    driver.set_page_load_timeout(cfg.page_load_timeout_sec)
    driver.implicitly_wait(cfg.implicit_wait_sec)
    try:
        yield driver
    finally:
        driver.quit()
