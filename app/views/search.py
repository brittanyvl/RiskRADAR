"""
Search page — hybrid search interface for 24,766 NTSB accident report chunks.

Horizontal filter bar above search box, full-width results below.
"""

import html
import re
import streamlit as st

from app.components import data_loader as dl
from app.components.report_layout import abbr, page_header, section_divider
from app.components.theme import NAVY, STEEL, CORAL, AMBER, TEAL, SLATE

# ── Constants ──────────────────────────────────────────────────────────────

PAGE_SIZE = 10

# L2 codes grouped by L1 parent (from CLAUDE.md taxonomy)
L2_BY_PARENT = {
    "LOC-I": ["LOC-I-STALL", "LOC-I-UPSET", "LOC-I-SD", "LOC-I-ENV", "LOC-I-SYS", "LOC-I-LOAD"],
    "CFIT":  ["CFIT-NAV", "CFIT-SA", "CFIT-VIS", "CFIT-TAWS", "CFIT-PROC"],
    "SCF-PP": ["SCF-PP-ENG", "SCF-PP-FUEL", "SCF-PP-PROP", "SCF-PP-FIRE"],
    "SCF-NP": ["SCF-NP-FLT", "SCF-NP-HYD", "SCF-NP-ELEC", "SCF-NP-STRUCT", "SCF-NP-GEAR"],
    "ICE":   ["ICE-STRUCT", "ICE-INDUCT", "ICE-PITOT"],
    "FUEL":  ["FUEL-EXHAUST", "FUEL-STARVE", "FUEL-CONTAM"],
}

AIRCRAFT_DISPLAY = {
    "single-piston": "Single-Engine Piston",
    "multi-piston":  "Multi-Engine Piston",
    "turboprop":     "Turboprop",
    "jet-narrow":    "Narrow-Body Jet",
    "jet-wide":      "Wide-Body Jet",
    "jet-regional":  "Regional Jet",
    "helicopter":    "Helicopter",
    "balloon":       "Balloon",
    "not-applicable": "Not Applicable",
    "other":         "Other",
}

MODE_MAP = {
    "Hybrid (default)": "hybrid",
    "Semantic":         "semantic",
    "Keyword":          "bm25",
}

SOURCE_COLORS = {
    "both":     TEAL,
    "semantic": STEEL,
    "bm25":     AMBER,
}

SOURCE_LABELS = {
    "both":     "Hybrid",
    "semantic": "Semantic",
    "bm25":     "Keyword",
}


# ── Search-specific CSS ───────────────────────────────────────────────────

_SEARCH_CSS = """
<style>
.result-card {
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.result-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.09); }
.result-rank {
    font-size: 0.7rem;
    font-weight: 700;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.result-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1B2A4A;
    margin: 0.15rem 0;
    line-height: 1.3;
}
.result-meta {
    font-size: 0.78rem;
    color: #6c757d;
    margin-bottom: 0.5rem;
}
.result-snippet {
    font-size: 0.83rem;
    color: #343a40;
    line-height: 1.6;
    border-left: 3px solid #dee2e6;
    padding-left: 0.6rem;
    margin: 0.4rem 0 0.5rem 0;
}
.result-snippet mark {
    background: #fff3cd;
    padding: 0 2px;
    border-radius: 2px;
}
.result-tags span {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    margin-right: 0.3rem;
    margin-bottom: 0.2rem;
    color: white;
}
.filter-note {
    font-size: 0.75rem;
    color: #868e96;
    font-style: italic;
    margin-top: 0.25rem;
}
.search-stats {
    font-size: 0.85rem;
    color: #495057;
    margin-bottom: 0.75rem;
}
</style>
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _esc(val: str) -> str:
    """HTML-escape a string for safe injection into unsafe_allow_html blocks."""
    return html.escape(val or "")


def _safe_url(url: str) -> str:
    """Only allow https:// URLs; reject anything else."""
    if url and url.startswith("https://"):
        return html.escape(url)
    return ""


def _snippet(text: str, query: str, max_chars: int = 350) -> str:
    """Return a highlighted, truncated snippet from chunk text."""
    # Strip leading [SECTION NAME] tag if present
    if text.startswith("[") and "]" in text:
        text = text[text.index("]") + 1:].strip()
    text = text.replace("\n", " ").strip()

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"

    # HTML-escape BEFORE inserting <mark> tags
    text = html.escape(text)

    # Highlight query terms (>2 chars, case-insensitive)
    terms = [t.strip() for t in query.split() if len(t.strip()) > 2]
    for term in terms:
        escaped_term = html.escape(term)
        pattern = re.compile(re.escape(escaped_term), re.IGNORECASE)
        text = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", text)

    return text


def _source_badge(source: str) -> str:
    color = SOURCE_COLORS.get(source, SLATE)
    label = SOURCE_LABELS.get(source, source)
    return f'<span style="background:{color};">{label}</span>'


def _category_badge(code: str) -> str:
    return f'<span style="background:{STEEL};">{_esc(code)}</span>'


def _render_result_card(result, query: str):
    """Render a single search result as an HTML card."""
    title = _esc(result.title or result.report_id)
    section = result.section_name or ""
    snippet = _snippet(result.chunk_text, query) if result.chunk_text else "<em>Chunk text unavailable</em>"

    # Meta line (escaped)
    meta_parts = []
    if result.accident_date:
        meta_parts.append(_esc(result.accident_date[:10]))
    if result.location:
        meta_parts.append(_esc(result.location))
    if section:
        meta_parts.append(_esc(section.title()))
    if result.aircraft_category:
        meta_parts.append(_esc(AIRCRAFT_DISPLAY.get(result.aircraft_category, result.aircraft_category)))
    meta_str = " · ".join(meta_parts)

    # Tags
    tags_html = _source_badge(result.source)
    for code in (result.l1_categories or [])[:4]:
        tags_html += _category_badge(code)

    # PDF link (URL validated)
    link_html = ""
    safe_url = _safe_url(result.pdf_url)
    if safe_url:
        link_html = (
            f'<a href="{safe_url}" target="_blank" '
            f'style="font-size:0.78rem;color:{STEEL};text-decoration:none;font-weight:500;">'
            f'View NTSB Report ↗</a>'
        )

    st.markdown(f"""
    <div class="result-card">
        <div class="result-rank">#{result.rank} · score {result.score:.4f}</div>
        <div class="result-title">{title}</div>
        <div class="result-meta">{meta_str}</div>
        <div class="result-snippet">{snippet}</div>
        <div class="result-tags">{tags_html}</div>
        {"<div style='margin-top:0.4rem;'>" + link_html + "</div>" if link_html else ""}
    </div>
    """, unsafe_allow_html=True)


def _render_empty_state():
    """Show introductory content when no query has been entered."""
    section_divider()
    st.markdown("#### How It Works")
    st.markdown(
        "RiskRADAR combines two retrieval strategies for comprehensive accident report search:\n\n"
        "- **BM25 (keyword)** — Term-frequency matching for exact phrases and report numbers\n"
        "- **Semantic embeddings** — MiniLM model finds conceptually related passages "
        "even when wording differs\n"
        "- **Hybrid (RRF)** — Reciprocal Rank Fusion merges both result lists, "
        "giving the best of both strategies\n\n"
        "Results are enriched with taxonomy tags and direct links to the original NTSB PDF reports."
    )
    st.markdown("#### Example Queries")
    examples = [
        "engine failure during initial climb",
        "spatial disorientation in IMC at night",
        "fuel exhaustion single engine piston",
        "runway excursion during landing in crosswind",
        "icing encounter turboprop approach",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")


# ── Main render ────────────────────────────────────────────────────────────

def render():
    st.markdown(_SEARCH_CSS, unsafe_allow_html=True)

    # Page header
    page_header(
        "Accident Report Search",
        "Search across 24,766 chunks from 510 NTSB accident reports using "
        "hybrid retrieval (BM25 + semantic embeddings).",
    )

    # Session state init
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_page" not in st.session_state:
        st.session_state.search_page = 0
    if "search_elapsed" not in st.session_state:
        st.session_state.search_elapsed = 0.0

    # Load filter options (cached)
    filter_opts = dl.get_search_filter_options()
    l1_codes = [c[0] for c in filter_opts["l1_categories"]]
    ac_options = filter_opts.get("aircraft_categories", [])

    # ── Horizontal filter bar ─────────────────────────────────────────────
    f_mode, f_l1, f_ac, f_from, f_to = st.columns([1.2, 2, 2, 1, 1])

    with f_mode:
        mode_label = st.selectbox(
            "Mode",
            options=list(MODE_MAP.keys()),
            index=0,
        )
        mode_key = MODE_MAP[mode_label]

    with f_l1:
        l1_filter = st.multiselect(
            "Risk Category",
            options=l1_codes,
            placeholder="All categories",
        )

    with f_ac:
        ac_filter = st.multiselect(
            "Aircraft Type",
            options=ac_options,
            format_func=lambda k: AIRCRAFT_DISPLAY.get(k, k),
            placeholder="All aircraft",
        )

    with f_from:
        date_from = st.date_input("From", value=None, key="search_date_from")

    with f_to:
        date_to = st.date_input("To", value=None, key="search_date_to")

    # L2 subcategories (conditional row — only shows when L1 selected)
    available_l2 = []
    for parent in l1_filter:
        available_l2.extend(L2_BY_PARENT.get(parent, []))

    l2_filter = []
    if available_l2:
        l2_filter = st.multiselect(
            "Subcategory (L2)",
            options=available_l2,
            placeholder="All subcategories",
        )

    # BM25 filter warning
    if mode_key == "bm25" and (l1_filter or l2_filter or date_from or date_to):
        st.markdown(
            '<div class="filter-note">Category and date filters apply to '
            'Semantic/Hybrid modes only. Keyword mode searches the full corpus.</div>',
            unsafe_allow_html=True,
        )

    # ── Search bar ────────────────────────────────────────────────────────
    search_col, btn_col = st.columns([5, 1])
    with search_col:
        query = st.text_input(
            "Search accident reports",
            placeholder="e.g. engine failure during climb, spatial disorientation in IMC",
            label_visibility="collapsed",
        )
    with btn_col:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # ── Detect filter/query changes → reset page ─────────────────────────
    state_key = (
        query, mode_key,
        tuple(sorted(l1_filter)),
        tuple(sorted(l2_filter)),
        tuple(sorted(ac_filter)),
        str(date_from), str(date_to),
    )
    if state_key != st.session_state.get("_search_state_key"):
        st.session_state.search_page = 0
        st.session_state._search_state_key = state_key

    # ── Execute search ────────────────────────────────────────────────────
    if (search_clicked or query) and query.strip():
        if search_clicked or st.session_state.get("_last_search_key") != state_key:
            from search.service import SearchQuery

            sq = SearchQuery(
                query=query.strip(),
                mode=mode_key,
                l1_categories=l1_filter,
                l2_subcategories=l2_filter,
                aircraft_categories=ac_filter,
                date_from=str(date_from) if date_from else None,
                date_to=str(date_to) if date_to else None,
            )

            with st.spinner("Searching…"):
                try:
                    service = dl.get_search_service()
                    results, elapsed = service.search(sq)
                    st.session_state.search_results = results
                    st.session_state.search_elapsed = elapsed
                    st.session_state.search_page = 0
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state.search_results = None
                    return

            st.session_state._last_search_key = state_key

        # ── Render results ────────────────────────────────────────────────
        results = st.session_state.search_results
        if results is None:
            return

        total = len(results)
        elapsed = st.session_state.search_elapsed

        if total == 0:
            st.info("No results found. Try broadening your query or removing filters.")
            return

        # Stats line
        page = st.session_state.search_page
        n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        st.markdown(
            f'<div class="search-stats">'
            f'<b>{total} results</b> · {mode_label} · {elapsed:.0f}ms'
            f' · page {page + 1} of {n_pages}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Paginated results
        offset = page * PAGE_SIZE
        page_results = results[offset:offset + PAGE_SIZE]

        for result in page_results:
            _render_result_card(result, query)

        # Pagination controls
        if n_pages > 1:
            st.markdown("---")
            pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
            with pcol1:
                if page > 0:
                    if st.button("← Previous", key="search_prev"):
                        st.session_state.search_page -= 1
                        st.rerun()
            with pcol2:
                st.markdown(
                    f'<div style="text-align:center;color:#6c757d;font-size:0.85rem;">'
                    f'Page {page + 1} of {n_pages}</div>',
                    unsafe_allow_html=True,
                )
            with pcol3:
                if page < n_pages - 1:
                    if st.button("Next →", key="search_next"):
                        st.session_state.search_page += 1
                        st.rerun()

    elif not query:
        _render_empty_state()
