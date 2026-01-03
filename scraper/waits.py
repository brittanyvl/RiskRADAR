"""
scraper/waits.py
----------------
Utility wait functions for Selenium.

- `until`: wraps WebDriverWait with expected conditions.
- `sleep`: thin wrapper over time.sleep for consistency.

Use in actions or scripts where explicit waits are required.
"""
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def until(driver, condition, timeout=30):
    return WebDriverWait(driver, timeout).until(condition)

def sleep(seconds: float):
    time.sleep(seconds)
