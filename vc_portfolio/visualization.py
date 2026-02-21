"""
visualization.py — Plotly figure factories for VC portfolio visualization.

Depends on: fund.py, jcurve.py, portfolio.py, scenarios.py
All functions return plotly.graph_objects.Figure objects.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from vc_portfolio.fund import Fund
from vc_portfolio.portfolio import Portfolio
from vc_portfolio.scenarios import ScenarioResults


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

_VC_COLORS = {
    "background": "#0D1117",
    "paper": "#161B22",
    "grid": "#21262D",
    "text": "#C9D1D9",
    "text_secondary": "#8B949E",
    "accent": "#58A6FF",
    "positive": "#3FB950",
    "negative": "#F85149",
    "neutral": "#FFA657",
}

_PLOTLY_TEMPLATE = "plotly_dark"


def _apply_vc_theme(fig: go.Figure) -> go.Figure:
    """
    Apply consistent dark-background professional VC styling to a Plotly figure.

    Modifies the figure in-place and returns it for chaining.
    """
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor=_VC_COLORS["paper"],
        plot_bgcolor=_VC_COLORS["background"],
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color=_VC_COLORS["text"],
            size=12,
        ),
        title_font=dict(size=16, color=_VC_COLORS["text"]),
        legend=dict(
            bgcolor=_VC_COLORS["paper"],
            bordercolor=_VC_COLORS["grid"],
            borderwidth=1,
            font=dict(color=_VC_COLORS["text_secondary"]),
        ),
        hoverlabel=dict(
            bgcolor=_VC_COLORS["paper"],
            bordercolor=_VC_COLORS["grid"],
            font=dict(color=_VC_COLORS["text"]),
        ),
    )
    fig.update_xaxes(
        gridcolor=_VC_COLORS["grid"],
        zerolinecolor=_VC_COLORS["grid"],
        tickfont=dict(color=_VC_COLORS["text_secondary"]),
    )
    fig.update_yaxes(
        gridcolor=_VC_COLORS["grid"],
        zerolinecolor=_VC_COLORS["grid"],
        tickfont=dict(color=_VC_COLORS["text_secondary"]),
    )
    return fig


# ---------------------------------------------------------------------------
# J-Curve visualization
# ---------------------------------------------------------------------------

def plot_jcurve(
    df: pd.DataFrame,
    show_components: bool = True,
    title: str = "Fund J-Curve — NAV Progression",
) -> go.Figure:
    """
    Plot fund J-curve with NAV line, stacked area decomposition, and cost basis.

    Parameters
    ----------
    df:
        Output of JCurve.project().
    show_components:
        If True, show stacked area of unrealized + realized value.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    periods = df["period"]

    if show_components and "unrealized_value" in df.columns:
        # Stacked area: unrealized on bottom, cost basis line overlay
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=df["unrealized_value"],
                fill="tozeroy",
                name="Unrealized Value",
                line=dict(color=_VC_COLORS["accent"], width=1),
                fillcolor="rgba(88, 166, 255, 0.15)",
                hovertemplate="Period %{x}<br>Unrealized: $%{y:,.0f}<extra></extra>",
            )
        )

        if "realized_value" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=df["cumulative_distributions"],
                    fill="tozeroy",
                    name="Cumulative Distributions",
                    line=dict(color=_VC_COLORS["positive"], width=1),
                    fillcolor="rgba(63, 185, 80, 0.10)",
                    hovertemplate="Period %{x}<br>Distributed: $%{y:,.0f}<extra></extra>",
                )
            )

    # Cost basis line
    if "cost_basis" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=df["cost_basis"],
                name="Cost Basis",
                line=dict(color=_VC_COLORS["neutral"], width=2, dash="dash"),
                hovertemplate="Period %{x}<br>Cost Basis: $%{y:,.0f}<extra></extra>",
            )
        )

    # NAV line (main series)
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=df["nav"],
            name="NAV",
            line=dict(color=_VC_COLORS["accent"], width=3),
            hovertemplate="Period %{x}<br>NAV: $%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Value ($)",
        hovermode="x unified",
    )

    return _apply_vc_theme(fig)


# ---------------------------------------------------------------------------
# Portfolio breakdown
# ---------------------------------------------------------------------------

def plot_portfolio_breakdown(
    portfolio: Portfolio,
    breakdown_by: Literal["stage", "sector"] = "stage",
) -> go.Figure:
    """
    Plot portfolio breakdown as treemap + horizontal bar chart.

    Parameters
    ----------
    portfolio:
        Portfolio instance.
    breakdown_by:
        Dimension to group by ('stage' or 'sector').

    Returns
    -------
    go.Figure (subplot with treemap + bar chart)
    """
    df = portfolio.portfolio_breakdown()
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "treemap"}, {"type": "bar"}]],
        subplot_titles=[
            f"Portfolio by {breakdown_by.title()} (Treemap)",
            f"Investment by {breakdown_by.title()} (Bar)",
        ],
    )

    # Group by dimension
    grouped = df.groupby(breakdown_by)["investment_amount"].sum().reset_index()
    grouped.columns = [breakdown_by, "investment_amount"]

    colors = [
        "#58A6FF", "#3FB950", "#FFA657", "#F85149",
        "#A371F7", "#39D353", "#FF7B72", "#79C0FF",
    ]

    # Treemap
    fig.add_trace(
        go.Treemap(
            labels=grouped[breakdown_by],
            parents=[""] * len(grouped),
            values=grouped["investment_amount"],
            textinfo="label+percent root",
            hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percentRoot:.1%}<extra></extra>",
            marker=dict(colors=colors[: len(grouped)]),
        ),
        row=1,
        col=1,
    )

    # Horizontal bar
    fig.add_trace(
        go.Bar(
            y=grouped[breakdown_by],
            x=grouped["investment_amount"],
            orientation="h",
            marker_color=colors[: len(grouped)],
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
            name="Investment",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title=f"Portfolio Breakdown by {breakdown_by.title()}")
    return _apply_vc_theme(fig)


# ---------------------------------------------------------------------------
# Scenario fan chart
# ---------------------------------------------------------------------------

def plot_scenario_fan(
    scenario_results: ScenarioResults,
    metric: Literal["moic", "irr"] = "moic",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot p10–p90 fan chart for each scenario (Cambridge Associates style).

    Parameters
    ----------
    scenario_results:
        Output of ScenarioSet.run().
    metric:
        Metric to plot ('moic' or 'irr').
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    fan_df = scenario_results.to_fan_chart_data(metric)
    metric_label = "MOIC (x)" if metric == "moic" else "IRR (%)"
    chart_title = title or f"Scenario Analysis — {metric_label} Distribution"

    fig = go.Figure()

    scenarios = fan_df["scenario"].tolist()
    x_pos = list(range(len(scenarios)))

    for _, row in fan_df.iterrows():
        color = row.get("color", "#636EFA")

        # P10–P90 range bar (widest band)
        fig.add_trace(
            go.Bar(
                x=[row["scenario"]],
                y=[row["p90"] - row["p10"]],
                base=[row["p10"]],
                marker_color=color,
                opacity=0.2,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # P25–P75 range bar (interquartile band)
        fig.add_trace(
            go.Bar(
                x=[row["scenario"]],
                y=[row["p75"] - row["p25"]],
                base=[row["p25"]],
                marker_color=color,
                opacity=0.45,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Median marker
        fig.add_trace(
            go.Scatter(
                x=[row["scenario"]],
                y=[row["p50"]],
                mode="markers+text",
                marker=dict(color=color, size=12, symbol="diamond"),
                text=[f"{row['p50']:.2f}x" if metric == "moic" else f"{row['p50']:.1%}"],
                textposition="top center",
                name=row["scenario"],
                hovertemplate=(
                    f"{row['scenario']}<br>"
                    f"P10: {row['p10']:.2f}<br>"
                    f"P25: {row['p25']:.2f}<br>"
                    f"P50: {row['p50']:.2f}<br>"
                    f"P75: {row['p75']:.2f}<br>"
                    f"P90: {row['p90']:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    # Reference line at 1x MOIC or 0% IRR
    ref_val = 1.0 if metric == "moic" else 0.0
    fig.add_hline(
        y=ref_val,
        line_dash="dot",
        line_color=_VC_COLORS["text_secondary"],
        annotation_text="Breakeven" if metric == "moic" else "0%",
        annotation_position="right",
    )

    fig.update_layout(
        title=chart_title,
        xaxis_title="Scenario",
        yaxis_title=metric_label,
        barmode="overlay",
        showlegend=True,
    )

    return _apply_vc_theme(fig)


# ---------------------------------------------------------------------------
# Return distribution
# ---------------------------------------------------------------------------

def plot_return_distribution(
    simulations: dict,
    metric: Literal["moic", "irr"] = "moic",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Histogram with Gaussian KDE overlay for return distribution.

    Parameters
    ----------
    simulations:
        Output of Portfolio.simulate_exits() (must have 'raw_moics' or 'raw_irrs').
    metric:
        'moic' or 'irr'
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    raw = simulations.get(f"raw_{metric}s", np.array([]))
    if len(raw) == 0:
        return go.Figure()

    metric_label = "MOIC (x)" if metric == "moic" else "IRR"
    chart_title = title or f"Return Distribution — {metric_label}"

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=raw,
            nbinsx=80,
            name="Simulated outcomes",
            marker_color=_VC_COLORS["accent"],
            opacity=0.6,
            histnorm="probability density",
            hovertemplate=f"{metric_label}: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>",
        )
    )

    # KDE overlay
    try:
        kde = gaussian_kde(raw, bw_method="scott")
        x_range = np.linspace(raw.min(), raw.max(), 200)
        kde_values = kde(x_range)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_values,
                name="KDE",
                line=dict(color=_VC_COLORS["neutral"], width=2),
                hoverinfo="skip",
            )
        )
    except np.linalg.LinAlgError:
        pass  # KDE fails on degenerate distributions

    # Median line
    median = float(np.median(raw))
    fig.add_vline(
        x=median,
        line_dash="dash",
        line_color=_VC_COLORS["positive"],
        annotation_text=f"Median: {median:.2f}",
        annotation_position="top right",
    )

    # Breakeven line
    ref_val = 1.0 if metric == "moic" else 0.0
    fig.add_vline(
        x=ref_val,
        line_dash="dot",
        line_color=_VC_COLORS["negative"],
        annotation_text="Breakeven",
        annotation_position="top left",
    )

    fig.update_layout(
        title=chart_title,
        xaxis_title=metric_label,
        yaxis_title="Probability Density",
        bargap=0.02,
    )

    return _apply_vc_theme(fig)


# ---------------------------------------------------------------------------
# Cash flow waterfall
# ---------------------------------------------------------------------------

def plot_cash_flows(
    fund: Fund,
    show_cumulative: bool = True,
    title: str = "Fund Cash Flow Analysis",
) -> go.Figure:
    """
    Bar chart of period cash flows with optional cumulative line on secondary y-axis.

    Parameters
    ----------
    fund:
        Fund instance.
    show_cumulative:
        If True, add cumulative net cash flow line on secondary y-axis.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    df = fund.get_cashflows()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Capital calls (negative)
    fig.add_trace(
        go.Bar(
            x=df["period"],
            y=-df["capital_called"],  # show as negative bars
            name="Capital Calls",
            marker_color=_VC_COLORS["negative"],
            opacity=0.8,
            hovertemplate="Period %{x}<br>Capital Called: $%{customdata:,.0f}<extra></extra>",
            customdata=df["capital_called"],
        ),
        secondary_y=False,
    )

    # Fees (negative)
    fig.add_trace(
        go.Bar(
            x=df["period"],
            y=-df["fees"],
            name="Mgmt Fees",
            marker_color=_VC_COLORS["neutral"],
            opacity=0.7,
            hovertemplate="Period %{x}<br>Fees: $%{customdata:,.0f}<extra></extra>",
            customdata=df["fees"],
        ),
        secondary_y=False,
    )

    # Distributions (positive)
    fig.add_trace(
        go.Bar(
            x=df["period"],
            y=df["distributions"],
            name="Distributions",
            marker_color=_VC_COLORS["positive"],
            opacity=0.8,
            hovertemplate="Period %{x}<br>Distributions: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    if show_cumulative:
        cumulative_net = (df["distributions"] - df["capital_called"] - df["fees"]).cumsum()
        fig.add_trace(
            go.Scatter(
                x=df["period"],
                y=cumulative_net,
                name="Cumulative Net CF",
                line=dict(color=_VC_COLORS["accent"], width=2),
                mode="lines+markers",
                marker=dict(size=4),
                hovertemplate="Period %{x}<br>Cumulative: $%{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=title,
        barmode="relative",
        xaxis_title="Period",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Cash Flow ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Net CF ($)", secondary_y=True)

    return _apply_vc_theme(fig)


# ---------------------------------------------------------------------------
# Sensitivity tornado
# ---------------------------------------------------------------------------

def plot_sensitivity_tornado(
    df: pd.DataFrame,
    metric: str = "moic_p50",
    title: str = "Sensitivity Analysis — Tornado Chart",
) -> go.Figure:
    """
    Horizontal bar tornado chart for sensitivity analysis.

    Parameters
    ----------
    df:
        Output of SensitivityAnalysis.tornado() with columns:
        parameter, low_value, high_value, swing (sorted descending).
    metric:
        Metric name for axis label.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # Base case (midpoint)
    base = df[["low_value", "high_value"]].mean(axis=1)

    for _, row in df.iterrows():
        # Downside bar (low_value to base)
        fig.add_trace(
            go.Bar(
                y=[row["parameter"]],
                x=[base.iloc[df.index.get_loc(_)] - row["low_value"]],
                base=[row["low_value"]],
                orientation="h",
                marker_color=_VC_COLORS["negative"],
                opacity=0.8,
                showlegend=False,
                name="Downside",
                hovertemplate=(
                    f"{row['parameter']}<br>"
                    f"Low: {row['low_value']:.2f}<br>"
                    f"Swing: {row['swing']:.2f}"
                    "<extra></extra>"
                ),
            )
        )

        # Upside bar (base to high_value)
        fig.add_trace(
            go.Bar(
                y=[row["parameter"]],
                x=[row["high_value"] - base.iloc[df.index.get_loc(_)]],
                base=[base.iloc[df.index.get_loc(_)]],
                orientation="h",
                marker_color=_VC_COLORS["positive"],
                opacity=0.8,
                showlegend=False,
                name="Upside",
                hovertemplate=(
                    f"{row['parameter']}<br>"
                    f"High: {row['high_value']:.2f}<br>"
                    f"Swing: {row['swing']:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    # Base case vertical line
    mean_base = float(base.mean())
    fig.add_vline(
        x=mean_base,
        line_dash="dash",
        line_color=_VC_COLORS["text_secondary"],
        annotation_text="Base",
    )

    fig.update_layout(
        title=title,
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title="Parameter",
        barmode="overlay",
    )

    return _apply_vc_theme(fig)


# ---------------------------------------------------------------------------
# PME comparison
# ---------------------------------------------------------------------------

def plot_pme_comparison(
    fund: Fund,
    index_returns: list[float],
    index_name: str = "S&P 500",
) -> go.Figure:
    """
    Visualize PME metrics: KS-PME, PME Alpha, and Direct Alpha.

    Parameters
    ----------
    fund:
        Fund instance.
    index_returns:
        Period-by-period index returns (same length as fund periods).
    index_name:
        Label for the benchmark index.

    Returns
    -------
    go.Figure
    """
    import numpy as np
    from vc_portfolio.metrics import pme

    lp_cfs = np.array(fund.get_lp_cashflows())
    index_arr = np.array(index_returns[: len(lp_cfs)])
    if len(index_arr) < len(lp_cfs):
        # Pad with mean return
        mean_ret = float(np.mean(index_arr)) if len(index_arr) > 0 else 0.08
        index_arr = np.concatenate(
            [index_arr, np.full(len(lp_cfs) - len(index_arr), mean_ret)]
        )

    pme_metrics = pme(lp_cfs, index_arr)

    fig = go.Figure()

    metrics_data = {
        "KS-PME": pme_metrics["ks_pme"],
        "PME Alpha (ann.)": pme_metrics["pme_alpha"],
        "Direct Alpha": pme_metrics["direct_alpha"],
    }

    colors = [
        _VC_COLORS["positive"] if v > (1.0 if k == "KS-PME" else 0.0) else _VC_COLORS["negative"]
        for k, v in metrics_data.items()
    ]

    fig.add_trace(
        go.Bar(
            x=list(metrics_data.keys()),
            y=list(metrics_data.values()),
            marker_color=colors,
            text=[f"{v:.3f}" for v in metrics_data.values()],
            textposition="outside",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )

    # Reference lines
    fig.add_hline(y=1.0, line_dash="dot", line_color=_VC_COLORS["text_secondary"],
                  annotation_text="Benchmark (KS-PME=1)")
    fig.add_hline(y=0.0, line_dash="dot", line_color=_VC_COLORS["grid"])

    fig.update_layout(
        title=f"Public Market Equivalent vs {index_name}",
        yaxis_title="Metric Value",
        xaxis_title="PME Metric",
        showlegend=False,
    )

    return _apply_vc_theme(fig)
