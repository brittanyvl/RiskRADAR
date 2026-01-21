"""
taxonomy - CICTT-anchored causal factor taxonomy for NTSB aviation accident reports.

This module maps NTSB accident reports to CAST/ICAO Common Taxonomy Team (CICTT)
occurrence categories, enabling users to browse accidents by causal factors.

The CICTT taxonomy provides a standardized classification system used internationally
for aviation safety analysis.

Usage:
    python -m taxonomy.cli map             # Map reports to CICTT categories
    python -m taxonomy.cli review          # Export for human review
    python -m taxonomy.cli stats           # Show taxonomy statistics
"""

__version__ = "0.2.0"
