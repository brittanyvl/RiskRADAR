"""
app.components.charts — Professional Plotly chart builders for RiskRADAR.

Design principles:
- Max 3-5 series per chart (if you need more, rethink the visual)
- Colorblind-safe palette (tested with Coblis)
- No donut/pie charts — use horizontal bars instead
- Rich hover tooltips with context and definitions
- Clean, minimal axes — data-to-ink ratio matters
- Sequential palettes for heatmaps, not rainbow
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from app.components.theme import (
    NAVY, STEEL, CORAL, AMBER, TEAL, SLATE,
    CHART_PALETTE, HEATMAP_SCALE, SEQUENTIAL_SCALE,
)

# ── Shared layout applied to every figure ─────────────────────────────────

_BASE_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#495057"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=36, b=10),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#dee2e6",
        font_size=12,
        font_family="Inter, sans-serif",
    ),
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.15,
        xanchor="center", x=0.5,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
    coloraxis_colorbar=dict(
        thickness=12,
        len=0.6,
        title_font_size=11,
        tickfont_size=10,
    ),
)

_AXIS_STYLE = dict(
    gridcolor="#f0f0f0",
    linecolor="#dee2e6",
    linewidth=1,
    tickfont=dict(size=11),
    title_font=dict(size=12, color="#495057"),
)


def _apply_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply professional layout to a figure."""
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=title, font=dict(size=14, color=NAVY), x=0, xanchor="left") if title else None,
        height=height,
    )
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


# ── Bar Charts ────────────────────────────────────────────────────────────

def horizontal_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: str = STEEL,
    hover_data: dict | None = None,
    height: int = 400,
    show_values: bool = True,
) -> go.Figure:
    """
    Horizontal bar chart — the workhorse for ranked lists.

    Use instead of: donut/pie charts, vertical bars with long labels.
    """
    df_sorted = df.sort_values(x, ascending=True)
    text = df_sorted[x].apply(lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else str(v)) if show_values else None

    hover_template = "<b>%{y}</b><br>Count: %{x:,.0f}<extra></extra>"
    if hover_data:
        hover_parts = ["<b>%{y}</b>", "Count: %{x:,.0f}"]
        for key, values in hover_data.items():
            hover_parts.append(f"{key}: %{{customdata[{list(hover_data.keys()).index(key)}]}}")
        hover_template = "<br>".join(hover_parts) + "<extra></extra>"

    fig = go.Figure(go.Bar(
        x=df_sorted[x],
        y=df_sorted[y],
        orientation="h",
        marker_color=color,
        text=text,
        textposition="outside",
        textfont=dict(size=11, color="#495057"),
        customdata=df_sorted[list(hover_data.keys())].values if hover_data else None,
        hovertemplate=hover_template,
    ))
    fig.update_xaxes(showgrid=True, zeroline=True, zerolinecolor="#dee2e6")
    fig.update_yaxes(showgrid=False)
    return _apply_layout(fig, title, height)


def vertical_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: str = STEEL,
    height: int = 400,
    show_values: bool = True,
) -> go.Figure:
    """Vertical bar chart — use for short x-axis labels (categories, seasons)."""
    text = df[y].apply(lambda v: f"{v:,.0f}") if show_values else None
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y],
        marker_color=color,
        text=text,
        textposition="outside",
        textfont=dict(size=11, color="#495057"),
        hovertemplate="<b>%{x}</b><br>Count: %{y:,.0f}<extra></extra>",
    ))
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(showgrid=False)
    return _apply_layout(fig, title, height)


def grouped_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str = "",
    height: int = 420,
    colors: dict | None = None,
    barmode: str = "group",
) -> go.Figure:
    """
    Grouped bar chart — MAX 2-4 groups. If you need more, use a different visual.

    Args:
        colors: Optional dict mapping group values to hex colors.
    """
    groups = df[group].unique()
    if len(groups) > 5:
        # Silently limit — caller should pre-filter
        groups = groups[:5]

    fig = go.Figure()
    for i, grp in enumerate(groups):
        subset = df[df[group] == grp]
        bar_color = (colors or {}).get(grp, CHART_PALETTE[i % len(CHART_PALETTE)])
        fig.add_trace(go.Bar(
            name=str(grp),
            x=subset[x], y=subset[y],
            marker_color=bar_color,
            hovertemplate=f"<b>%{{x}}</b><br>{grp}: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(barmode=barmode)
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(showgrid=False)
    return _apply_layout(fig, title, height)


# ── Heatmaps ─────────────────────────────────────────────────────────────

def heatmap(
    matrix: pd.DataFrame,
    title: str = "",
    height: int = 450,
    mask_diagonal: bool = False,
    lower_triangle_only: bool = False,
    annotation_threshold: int | None = None,
    colorscale: list | str | None = None,
) -> go.Figure:
    """
    Heatmap from a pivot table / matrix DataFrame.

    Args:
        mask_diagonal: If True, set diagonal to NaN so it doesn't dominate visually.
        lower_triangle_only: If True, mask the upper triangle (for symmetric matrices).
        annotation_threshold: Only annotate cells with values >= this threshold.
        colorscale: Custom colorscale. Defaults to SEQUENTIAL_SCALE.
    """
    z = matrix.values.astype(float).copy()
    if lower_triangle_only:
        # Mask upper triangle (keep lower + diagonal unless mask_diagonal is also set)
        mask = np.triu_indices_from(z, k=1)
        z[mask] = np.nan
    if mask_diagonal:
        np.fill_diagonal(z, np.nan)

    scale = colorscale or SEQUENTIAL_SCALE

    # Build annotation text — only show values above threshold
    text = z.copy().astype(object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z[i, j]
            if np.isnan(val):
                text[i, j] = ""
            elif annotation_threshold is not None and val < annotation_threshold:
                text[i, j] = ""
            else:
                text[i, j] = f"{int(val)}"

    fig = go.Figure(go.Heatmap(
        z=z,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=scale,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b> & <b>%{x}</b><br>Reports: %{z:.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=12, len=0.5, title="Reports"),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10))
    fig.update_xaxes(showgrid=False, tickangle=-45, tickfont=dict(size=10))
    return _apply_layout(fig, title, height)


# ── Line Charts ──────────────────────────────────────────────────────────

def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "",
    height: int = 380,
    y_label: str = "",
) -> go.Figure:
    """
    Line chart — optionally multi-series. Max 4-5 series for readability.
    """
    if color:
        groups = df[color].unique()
        fig = go.Figure()
        for i, grp in enumerate(groups):
            subset = df[df[color] == grp].sort_values(x)
            fig.add_trace(go.Scatter(
                x=subset[x], y=subset[y],
                mode="lines+markers",
                name=str(grp),
                line=dict(color=CHART_PALETTE[i % len(CHART_PALETTE)], width=2.5),
                marker=dict(size=7),
                hovertemplate=f"<b>{grp}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>",
            ))
    else:
        df_sorted = df.sort_values(x)
        fig = go.Figure(go.Scatter(
            x=df_sorted[x], y=df_sorted[y],
            mode="lines+markers",
            line=dict(color=STEEL, width=2.5),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}<extra></extra>",
        ))

    fig.update_yaxes(showgrid=True, title_text=y_label)
    fig.update_xaxes(showgrid=False)
    return _apply_layout(fig, title, height)


# ── Stacked Bar ──────────────────────────────────────────────────────────

def stacked_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str = "",
    height: int = 420,
    orientation: str = "h",
) -> go.Figure:
    """Stacked bar chart — horizontal or vertical."""
    groups = df[group].unique()
    fig = go.Figure()
    for i, grp in enumerate(groups[:5]):  # Max 5 segments
        subset = df[df[group] == grp]
        if orientation == "h":
            fig.add_trace(go.Bar(
                name=str(grp), y=subset[x], x=subset[y],
                orientation="h",
                marker_color=CHART_PALETTE[i % len(CHART_PALETTE)],
                hovertemplate=f"<b>%{{y}}</b><br>{grp}: %{{x:,.0f}}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Bar(
                name=str(grp), x=subset[x], y=subset[y],
                marker_color=CHART_PALETTE[i % len(CHART_PALETTE)],
                hovertemplate=f"<b>%{{x}}</b><br>{grp}: %{{y:,.0f}}<extra></extra>",
            ))
    fig.update_layout(barmode="stack")
    return _apply_layout(fig, title, height)
