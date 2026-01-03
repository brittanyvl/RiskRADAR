# Scraper Library Documentation

This scraper library makes Selenium scraping clean and consistent. It provides building blocks for browser setup, form actions, waits, and download handling. You keep site-specific locators in your scripts, while the library handles repetitive boilerplate.

## Features

### Browser configuration and setup
Use the configuration object to control headless mode, user agent, timeouts, and download directory. The browser context manager sets up ChromeDriver and closes it automatically.

### Page navigation and form actions
Functions let you go to a page, select values from dropdown menus, or click buttons with proper waits so the page is ready.

### Download handling
Functions wait for a new file to appear, confirm the file is stable, and move or rename it atomically so downstream code always sees complete files.

## Quick Example

Short example showing how to scrape and download a file:

```python
from pathlib import Path
from datetime import datetime
from selenium.webdriver.common.by import By

from scraper.config import BrowserConfig
from scraper.browser import chrome
from scraper.actions import go_to, select_dropdown_by_value, click
from scraper.download import wait_for_new_download, move_and_rename

URL = "https://example.com/data"
SHOW_LENGTH_DROPDOWN = (By.NAME, "DataTables_Table_0_length")
EXPORT_BUTTON = (By.CLASS_NAME, "buttons-excel")

def download_data(download_directory: str) -> str:
    config = BrowserConfig()
    file_name = f"data_{datetime.today().strftime('%Y-%m-%d')}.xlsx"

    with chrome(config) as driver:
        go_to(driver, URL)
        select_dropdown_by_value(driver, SHOW_LENGTH_DROPDOWN, "-1")
        click(driver, EXPORT_BUTTON)
        downloaded_file = wait_for_new_download(config)

    final_file = move_and_rename(downloaded_file, Path(download_directory), file_name)
    return str(final_file)
```

## Usage Patterns

### Simple form and download
Go to page, set dropdown, click export, wait for download, move file.

### Stable naming
Move once with dated filename, again with "latest.xlsx".

### Multiple downloads
Trigger each export, call wait, then move/rename each.

### PDF downloads
The library handles any file type. For PDFs, trigger the download link/button and use the same wait/move pattern.

## Installation

From the repository root:

```bash
pip install -e ./scraper
```

## API Reference

### Browser Configuration

```python
from scraper.config import BrowserConfig

# Default configuration
config = BrowserConfig()

# Custom configuration
config = BrowserConfig(
    headless=False,
    downloads_dir=Path("/custom/downloads"),
    user_agent="CustomBot/1.0",
    timeout_secs=30
)
```

### Browser Management

```python
from scraper.browser import chrome

with chrome(config) as driver:
    # Your scraping code here
    pass
```

### Page Actions

```python
from scraper.actions import go_to, select_dropdown_by_value, click

# Navigate to a page
go_to(driver, "https://example.com")

# Select dropdown value
select_dropdown_by_value(driver, (By.NAME, "dropdown"), "value")

# Click button
click(driver, (By.CLASS_NAME, "button"))
```

### Download Handling

```python
from scraper.download import wait_for_new_download, move_and_rename

# Wait for download to complete
downloaded_file = wait_for_new_download(config)

# Move and rename file
final_path = move_and_rename(
    downloaded_file,
    target_dir=Path("/target"),
    new_name="final_file.pdf"
)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRAPER_HEADLESS` | `1` | Run browser in headless mode |
| `SCRAPER_DOWNLOADS_DIR` | `.scraper_tmp_downloads` | Temporary download directory |
| `SCRAPER_DOWNLOAD_TIMEOUT` | `120` | Download timeout in seconds |
| `HTTP_USER_AGENT` | `RiskRADARBot/1.0` | Browser user agent string |
| `SCRAPER_PAGELOAD_TIMEOUT` | `60` | Page load timeout in seconds |

## Summary

The scraper library keeps your code short and expressive. Compose browser setup, actions, waits, and downloads in sequence. Keep the library generic and put site-specific steps in your application code.
