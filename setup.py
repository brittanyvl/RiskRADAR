"""
RiskRADAR - setup.py
--------------------
Installs all RiskRADAR packages and provides CLI entry point.

Usage:
    pip install -e .
    riskradar scrape --help
"""
from setuptools import setup, find_packages

setup(
    name="riskradar",
    version="0.1.0",
    description="NTSB Aviation Accident Report Analysis Pipeline",
    author="RiskRADAR",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "selenium",
        "webdriver-manager",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "riskradar=riskradar.cli:main",
        ],
    },
)
