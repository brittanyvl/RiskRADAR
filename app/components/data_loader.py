"""
app.components.data_loader — Cached data access for Streamlit pages.

Every analytics function is wrapped with @st.cache_data(ttl=3600) so that
repeated renders don't re-query SQLite. The Bayesian model uses
@st.cache_resource since it is a stateful object.
"""

import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# shared.py wrappers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_analytics_population() -> pd.DataFrame:
    from analytics.queries.shared import get_analytics_population as _fn
    return _fn()


@st.cache_data(ttl=3600)
def category_counts() -> pd.DataFrame:
    from analytics.queries.shared import category_counts as _fn
    return _fn()


@st.cache_data(ttl=3600)
def subcategory_counts(parent_code: str) -> pd.DataFrame:
    from analytics.queries.shared import subcategory_counts as _fn
    return _fn(parent_code)


@st.cache_data(ttl=3600)
def cooccurrence_matrix() -> pd.DataFrame:
    from analytics.queries.shared import cooccurrence_matrix as _fn
    return _fn()


@st.cache_data(ttl=3600)
def category_by_feature(feature_col: str, categories: list[str] | None = None) -> pd.DataFrame:
    from analytics.queries.shared import category_by_feature as _fn
    return _fn(feature_col, categories)


@st.cache_data(ttl=3600)
def category_prevalence_by_decade(categories: list[str] | None = None) -> pd.DataFrame:
    from analytics.queries.shared import category_prevalence_by_decade as _fn
    return _fn(categories)


@st.cache_data(ttl=3600)
def dataset_summary() -> dict:
    from analytics.queries.shared import dataset_summary as _fn
    return _fn()


# ---------------------------------------------------------------------------
# fleet_safety.py wrappers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def risk_by_aircraft_category() -> pd.DataFrame:
    from analytics.queries.fleet_safety import risk_by_aircraft_category as _fn
    return _fn()


@st.cache_data(ttl=3600)
def risk_by_manufacturer(top_n: int = 15) -> pd.DataFrame:
    from analytics.queries.fleet_safety import risk_by_manufacturer as _fn
    return _fn(top_n)


@st.cache_data(ttl=3600)
def scf_pp_breakdown() -> pd.DataFrame:
    from analytics.queries.fleet_safety import scf_pp_breakdown as _fn
    return _fn()


@st.cache_data(ttl=3600)
def scf_np_breakdown() -> pd.DataFrame:
    from analytics.queries.fleet_safety import scf_np_breakdown as _fn
    return _fn()


@st.cache_data(ttl=3600)
def failure_trends_by_decade() -> pd.DataFrame:
    from analytics.queries.fleet_safety import failure_trends_by_decade as _fn
    return _fn()


@st.cache_data(ttl=3600)
def manufacturer_category_heatmap(top_n: int = 10) -> pd.DataFrame:
    from analytics.queries.fleet_safety import manufacturer_category_heatmap as _fn
    return _fn(top_n)


# ---------------------------------------------------------------------------
# underwriting.py wrappers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def region_season_matrix() -> pd.DataFrame:
    from analytics.queries.underwriting import region_season_matrix as _fn
    return _fn()


@st.cache_data(ttl=3600)
def vmc_imc_category_distribution() -> pd.DataFrame:
    from analytics.queries.underwriting import vmc_imc_category_distribution as _fn
    return _fn()


@st.cache_data(ttl=3600)
def time_of_day_distribution() -> pd.DataFrame:
    from analytics.queries.underwriting import time_of_day_distribution as _fn
    return _fn()


@st.cache_data(ttl=3600)
def multi_label_complexity() -> pd.DataFrame:
    from analytics.queries.underwriting import multi_label_complexity as _fn
    return _fn()


@st.cache_data(ttl=3600)
def bayesian_profile_comparison(profiles: list[dict]) -> pd.DataFrame:
    from analytics.queries.underwriting import bayesian_profile_comparison as _fn
    return _fn(profiles)


@st.cache_data(ttl=3600)
def category_by_aircraft_and_weather() -> pd.DataFrame:
    from analytics.queries.underwriting import category_by_aircraft_and_weather as _fn
    return _fn()


# ---------------------------------------------------------------------------
# operational_risk.py wrappers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def loc_i_breakdown() -> pd.DataFrame:
    from analytics.queries.operational_risk import loc_i_breakdown as _fn
    return _fn()


@st.cache_data(ttl=3600)
def cfit_breakdown() -> pd.DataFrame:
    from analytics.queries.operational_risk import cfit_breakdown as _fn
    return _fn()


@st.cache_data(ttl=3600)
def weather_time_matrix() -> pd.DataFrame:
    from analytics.queries.operational_risk import weather_time_matrix as _fn
    return _fn()


@st.cache_data(ttl=3600)
def seasonal_patterns(categories: list[str] | None = None) -> pd.DataFrame:
    from analytics.queries.operational_risk import seasonal_patterns as _fn
    return _fn(categories)


@st.cache_data(ttl=3600)
def aircraft_type_risk_signatures() -> pd.DataFrame:
    from analytics.queries.operational_risk import aircraft_type_risk_signatures as _fn
    return _fn()


@st.cache_data(ttl=3600)
def critical_phase_categories() -> pd.DataFrame:
    from analytics.queries.operational_risk import critical_phase_categories as _fn
    return _fn()


# ---------------------------------------------------------------------------
# glossary_data.py wrappers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_l1_glossary() -> pd.DataFrame:
    from analytics.queries.glossary_data import get_l1_glossary as _fn
    return _fn()


@st.cache_data(ttl=3600)
def get_l2_glossary() -> pd.DataFrame:
    from analytics.queries.glossary_data import get_l2_glossary as _fn
    return _fn()


@st.cache_data(ttl=3600)
def get_feature_definitions() -> dict:
    from analytics.queries.glossary_data import get_feature_definitions as _fn
    return _fn()


@st.cache_data(ttl=3600)
def get_aviation_terms() -> list[dict]:
    from analytics.queries.glossary_data import get_aviation_terms as _fn
    return _fn()


@st.cache_data(ttl=3600)
def get_statistical_terms() -> list[dict]:
    from analytics.queries.glossary_data import get_statistical_terms as _fn
    return _fn()


# ---------------------------------------------------------------------------
# Bayesian model (stateful object, not serializable)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_bayesian_model():
    """Load the pre-trained Bayesian risk model."""
    from risk_profiler.bayesian_model import load_model
    from riskradar.config import DB_PATH
    return load_model(db_path=str(DB_PATH))
