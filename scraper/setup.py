from setuptools import setup, find_packages

setup(
    name="scraper",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "selenium",
        "webdriver-manager",
    ],
    python_requires=">=3.9",
)
