"""
scraper/actions.py
------------------
Reusable Selenium actions to reduce boilerplate.

- Navigate to a URL with `go_to`.
- Select dropdown options by value or text.
- Click elements with proper wait conditions.

These functions encapsulate common "form → action" patterns.
"""
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from .waits import until

def go_to(driver, url: str, timeout=60):
    driver.get(url)
    until(driver, lambda d: d.execute_script("return document.readyState") == "complete", timeout=timeout)

def select_dropdown_by_value(driver, locator: tuple[By, str], value: str, timeout=15):
    elem = until(driver, EC.presence_of_element_located(locator), timeout)
    Select(elem).select_by_value(value)

def select_dropdown_by_text(driver, locator: tuple[By, str], text: str, timeout=15):
    elem = until(driver, EC.presence_of_element_located(locator), timeout)
    Select(elem).select_by_visible_text(text)

def click(driver, locator: tuple[By, str], timeout=15):
    elem = until(driver, EC.element_to_be_clickable(locator), timeout)
    elem.click()
