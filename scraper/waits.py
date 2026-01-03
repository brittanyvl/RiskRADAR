"""
scraper/waits.py
----------------
Utility wait functions for Selenium.

- `until`: wraps WebDriverWait with expected conditions.
- `sleep`: thin wrapper over time.sleep for consistency.
- `rate_limit`: delay between requests to be respectful to servers.

Use in actions or scripts where explicit waits are required.
"""
import time
import random
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def until(driver, condition, timeout=30):
    return WebDriverWait(driver, timeout).until(condition)

def sleep(seconds: float):
    time.sleep(seconds)

def rate_limit(base_delay: float, jitter: float = 0.5):
    """
    Sleep for base_delay +/- jitter to avoid predictable request patterns.

    Args:
        base_delay: Base delay in seconds
        jitter: Random variation factor (0.5 = +/- 50%)
    """
    variation = base_delay * jitter * (2 * random.random() - 1)
    actual_delay = max(0.5, base_delay + variation)  # Minimum 0.5s
    time.sleep(actual_delay)
