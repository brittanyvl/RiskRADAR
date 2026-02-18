"""
analytics.queries — Reusable query modules for RiskRADAR stakeholder analytics.

Modules:
    shared            Core analytics: population, category counts, co-occurrence
    fleet_safety      Aircraft type, manufacturer, component failure queries
    underwriting      Region/season/weather/time risk matrices
    operational_risk  LOC-I/CFIT deep dives, weather x time matrix
    glossary_data     Structured glossary content from taxonomy sources
"""

from .shared import (
    get_connection,
    get_analytics_population,
    category_counts,
    subcategory_counts,
    cooccurrence_matrix,
    category_by_feature,
    category_prevalence_by_decade,
    dataset_summary,
)

__all__ = [
    "get_connection",
    "get_analytics_population",
    "category_counts",
    "subcategory_counts",
    "cooccurrence_matrix",
    "category_by_feature",
    "category_prevalence_by_decade",
    "dataset_summary",
]
