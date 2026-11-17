"""
VC Portfolio Toolkit: interactive app.

Run locally:
    pip install -e ".[app]"
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from vc_portfolio import (
    BASE_SCENARIO,
    BEAR_SCENARIO,
    BULL_SCENARIO,
    STAGE_LOSS_RATES,
    CompanyTimeline,
    JCurve,
    MonteCarloSimulator,
    Portfolio,
    PortfolioInputError,
    Scenario,
    ScenarioSet,
    SensitivityAnalysis,
    parse_portfolio_table,
)
from vc_portfolio import visualization as viz
from vc_portfolio.portfolio import (
    _COLUMN_ALIASES,
    TABLE_COLUMNS,
    _normalize_key,
    _to_number,
    expected_exit_multiple,
)

SAMPLE_CSV = Path(__file__).parent / "examples" / "sample_portfolio.csv"
STAGES = list(STAGE_LOSS_RATES)

st.set_page_config(page_title="VC Portfolio Toolkit", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def money(x: float) -> str:
    sign, x = ("-" if x < 0 else ""), abs(x)
    if x >= 1e9:
        return f"{sign}${x / 1e9:,.2f}B"
    if x >= 1e6:
        return f"{sign}${x / 1e6:,.1f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:,.0f}K"
    return f"{sign}${x:,.0f}"


def chart(fig: object) -> None:
    """Show a toolkit figure with its own dark theme."""
    st.plotly_chart(fig, use_container_width=True, theme=None)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


NUMERIC_COLUMNS = [
    "investment_amount", "entry_year", "exit_year", "entry_valuation",
    "ownership_pct", "reserved_capital", "expected_multiple",
]
TEXT_COLUMNS = ["name", "stage", "sector"]


def template_table() -> pd.DataFrame:
    row = {
        "name": "ExampleCo", "investment_amount": 2_000_000, "stage": "seed",
        "entry_year": 0, "exit_year": 7, "sector": "saas", "entry_valuation": 10_000_000,
        "ownership_pct": 0.15, "reserved_capital": 1_000_000, "expected_multiple": None,
    }
    return pd.DataFrame([row], columns=TABLE_COLUMNS)


def prepare_for_editor(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Put any table into the editor's shape: known columns in a fixed order,
    numbers parsed ("$2M", "15%"), stages normalized ("Series A" -> series_a).
    Returns the table and a list of cells that could not be read as numbers.
    """
    df = raw.copy()
    df.columns = [_COLUMN_ALIASES.get(_normalize_key(c), _normalize_key(c)) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    out = pd.DataFrame(index=df.index)
    problems: list[str] = []
    for col in TABLE_COLUMNS:
        if col not in df.columns:
            out[col] = None if col in TEXT_COLUMNS else np.nan
        elif col in NUMERIC_COLUMNS:
            vals = []
            for i, v in enumerate(df[col], start=1):
                try:
                    vals.append(_to_number(v))
                except (TypeError, ValueError):
                    problems.append(f"Row {i}, {col}: '{v}'")
                    vals.append(np.nan)
            out[col] = pd.Series(vals, index=df.index, dtype="float64")
        else:
            out[col] = df[col].map(lambda v: None if pd.isna(v) else str(v).strip())
    out["stage"] = out["stage"].map(lambda v: _normalize_key(v) if v else v)
    # Ownership as a fraction: 15 and 0.15 both mean 15%
    own = out["ownership_pct"]
    out["ownership_pct"] = own.where(~(own > 1), own / 100)
    out["sector"] = out["sector"].astype("object")
    return out.reset_index(drop=True), problems


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@st.cache_data(show_spinner=False)
def read_upload(data: bytes, name: str) -> pd.DataFrame:
    import io

    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Cached computations (keyed on the cleaned table and settings)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running Monte Carlo simulation...")
def run_simulation(clean: pd.DataFrame, n_sims: int, seed: int) -> dict:
    portfolio = Portfolio.from_dataframe(clean)
    sim = MonteCarloSimulator(portfolio, n_simulations=n_sims, seed=seed)
    result = sim.run()
    result["var_95"] = sim.compute_var(0.95, "moic")
    result["cvar_95"] = sim.expected_shortfall(0.95, "moic")
    return result


@st.cache_data(show_spinner="Running scenarios...")
def run_scenarios(clean: pd.DataFrame, n_sims: int, seed: int, settings: tuple) -> object:
    portfolio = Portfolio.from_dataframe(clean)
    scenarios = [
        Scenario(name, desc, ret, timing, loss, color)
        for name, desc, ret, timing, loss, color in settings
    ]
    return ScenarioSet(scenarios).run(portfolio, n_simulations=n_sims, base_seed=seed)


@st.cache_data(show_spinner="Running sensitivity analysis...")
def run_tornado(
    clean: pd.DataFrame, n_sims: int, seed: int, params: tuple, metric: str, pct: float
) -> pd.DataFrame:
    portfolio = Portfolio.from_dataframe(clean)
    sa = SensitivityAnalysis(portfolio, n_simulations=n_sims, seed=seed)
    return sa.tornado(list(params), metric=metric, pct_change=pct)


@st.cache_data(show_spinner=False)
def run_jcurve(clean: pd.DataFrame, committed: float, fee_rate: float) -> tuple[pd.DataFrame, dict]:
    timelines = []
    for r in clean.itertuples(index=False):
        mult = r.expected_multiple
        if mult is None or (isinstance(mult, float) and math.isnan(mult)):
            mult = expected_exit_multiple(r.stage)
        timelines.append(
            CompanyTimeline(
                entry_period=int(round(r.entry_year * 4)),
                exit_period=max(int(round(r.exit_year * 4)), int(round(r.entry_year * 4)) + 1),
                initial_investment=float(r.investment_amount),
                exit_multiple=float(mult),
                company_name=r.name,
            )
        )
    n_periods = max(40, max(t.exit_period for t in timelines) + 4)
    jc = JCurve(n_periods=n_periods, fee_rate=fee_rate, committed_capital=committed)
    jc.add_companies(timelines)
    return jc.project(), jc.get_jcurve_shape()


# ---------------------------------------------------------------------------
# Sidebar: data source and settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Portfolio")
    source = st.radio("Start from", ["Sample portfolio", "Upload a file", "Blank template"])
    upload = None
    if source == "Upload a file":
        upload = st.file_uploader("CSV or Excel, one row per company", type=["csv", "xlsx", "xls"])
    st.download_button(
        "Download CSV template",
        csv_bytes(template_table()),
        file_name="portfolio_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.header("Simulation")
    n_sims = st.select_slider(
        "Simulations", options=[1_000, 2_000, 5_000, 10_000, 25_000], value=10_000
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.header("Fund (for the J-curve)")
    fee_pct = st.number_input("Annual management fee (%)", 0.0, 5.0, 2.0, 0.25)

    st.caption(
        "Your data stays in this browser session. Nothing you enter or upload is stored."
    )

# ---------------------------------------------------------------------------
# Header and portfolio table
# ---------------------------------------------------------------------------

st.title("VC Portfolio Toolkit")
st.write(
    "Simulate how a venture portfolio might return. Edit the table below or upload your own "
    "portfolio, and the analysis updates. Exit outcomes are drawn from a power-law mixture "
    "calibrated by stage, so results show the range of plausible outcomes, not a forecast."
)

if source == "Sample portfolio":
    initial, editor_key = load_sample(), "editor_sample"
elif source == "Upload a file" and upload is not None:
    try:
        initial = read_upload(upload.getvalue(), upload.name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {upload.name}: {exc}")
        st.stop()
    editor_key = f"editor_{upload.file_id}"
elif source == "Upload a file":
    st.info("Upload a CSV or Excel file in the sidebar, or download the template to get started.")
    st.stop()
else:
    initial, editor_key = template_table(), "editor_blank"

initial, unreadable = prepare_for_editor(initial)
if unreadable:
    st.warning(
        "Some cells could not be read as numbers and were left blank: "
        + "; ".join(unreadable[:10])
        + (" ..." if len(unreadable) > 10 else "")
    )

with st.expander("How to fill in the table", expanded=source != "Sample portfolio"):
    st.markdown(
        """
- **name, investment_amount, stage, entry_year, exit_year** are required.
- **Years** count from the fund's first close (0 = first year) and can be fractional, e.g. 2.5.
- **stage** is one of: """
        + ", ".join(f"`{s}`" for s in STAGES)
        + """. Stage sets the chance of a total loss.
- **ownership_pct**: files can use 0.15, 15 or 15%. In the table, type it as a fraction (0.15).
- **Amounts** in files can be 2000000, 2,000,000 or $2M.
- **expected_multiple** is optional and only used for the J-curve. Leave it blank to use the
  average multiple for the company's stage.
"""
    )

edited = st.data_editor(
    initial,
    key=editor_key,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "name": st.column_config.TextColumn("Company", required=True),
        "investment_amount": st.column_config.NumberColumn("Invested ($)", min_value=0, format="localized"),
        "stage": st.column_config.SelectboxColumn("Stage", options=STAGES, required=True),
        "sector": st.column_config.TextColumn("Sector"),
        "entry_year": st.column_config.NumberColumn("Entry year", min_value=0.0, step=0.25),
        "exit_year": st.column_config.NumberColumn("Exit year", min_value=0.0, step=0.25),
        "entry_valuation": st.column_config.NumberColumn("Entry valuation ($)", min_value=0, format="localized"),
        "ownership_pct": st.column_config.NumberColumn("Ownership", min_value=0.0, max_value=1.0, step=0.01, format="percent"),
        "reserved_capital": st.column_config.NumberColumn("Reserves ($)", min_value=0, format="localized"),
        "expected_multiple": st.column_config.NumberColumn("Expected multiple (J-curve)", min_value=0.0),
    },
)

try:
    clean = parse_portfolio_table(edited)
except PortfolioInputError as exc:
    st.error("Please fix these before the analysis can run:\n\n" + "\n".join(f"- {e}" for e in exc.errors))
    st.stop()

total_invested = float(clean["investment_amount"].sum())
total_reserves = float(clean["reserved_capital"].sum())
default_committed = max(1e6, round((total_invested + total_reserves) / 0.8 / 1e6) * 1e6)

# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------

sim = run_simulation(clean, int(n_sims), int(seed))

c = st.columns(6)
c[0].metric("Companies", len(clean))
c[1].metric("Invested", money(total_invested))
c[2].metric("Median MOIC", f"{sim['moic_p50']:.2f}x")
c[3].metric("Median IRR", f"{sim['irr_p50']:.1%}")
c[4].metric("Chance of losing money", f"{sim['loss_probability']:.0%}")
c[5].metric("Chance of 3x or more", f"{sim['home_run_probability']:.0%}")
st.caption("Gross returns on invested capital, before fees and carry.")

tab_returns, tab_scen, tab_sens, tab_j, tab_port = st.tabs(
    ["Return distribution", "Scenarios", "Sensitivity", "J-curve", "Portfolio"]
)

# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

with tab_returns:
    metric = st.radio("Show", ["MOIC", "IRR"], horizontal=True, key="dist_metric")
    chart(viz.plot_return_distribution(sim, metric=metric.lower(), clip_percentile=99.0))
    st.caption("The axis shows the 1st to 99th percentile of outcomes; the rare extremes are still included in every statistic.")

    left, right = st.columns([2, 1])
    with left:
        pct = pd.DataFrame(
            {
                "Percentile": ["10th", "25th", "50th (median)", "75th", "90th"],
                "MOIC": [f"{sim[f'moic_p{p}']:.2f}x" for p in (10, 25, 50, 75, 90)],
                "IRR": [f"{sim[f'irr_p{p}']:.1%}" for p in (10, 25, 50, 75, 90)],
            }
        )
        st.dataframe(pct, hide_index=True, use_container_width=True)
    with right:
        st.metric("95% VaR (MOIC)", f"{sim['var_95']:.2f}x", help="The worst 5% of outcomes are at or below this.")
        st.metric("95% CVaR (MOIC)", f"{sim['cvar_95']:.2f}x", help="Average of the worst 5% of outcomes.")
        st.download_button(
            "Download all simulations (CSV)",
            csv_bytes(pd.DataFrame({"moic": sim["raw_moics"], "irr": sim["raw_irrs"]})),
            file_name="simulations.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

with tab_scen:
    st.write(
        "Compare market environments. Each scenario scales exit multiples, shifts the stage "
        "loss rates and moves exits earlier or later. All scenarios use the same random draws, "
        "so differences come from the assumptions alone."
    )
    settings = []
    cols = st.columns(3)
    for col, base in zip(cols, [BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO], strict=True):
        with col:
            st.subheader(base.name)
            ret = st.slider("Exit multiple factor", 0.3, 2.5, float(base.portfolio_return_adjustment), 0.05, key=f"r_{base.name}")
            loss = st.slider("Loss rate change (pts)", -30, 30, int(round(base.loss_rate_adjustment * 100)), 1, key=f"l_{base.name}")
            timing = st.slider("Exit timing (quarters)", -8, 12, int(base.exit_timing_adjustment), 1, key=f"t_{base.name}")
            settings.append((base.name, base.description, ret, timing, loss / 100, base.color))

    results = run_scenarios(clean, int(n_sims), int(seed), tuple(settings))
    fan_metric = st.radio("Show", ["MOIC", "IRR"], horizontal=True, key="fan_metric")
    chart(viz.plot_scenario_fan(results, metric=fan_metric.lower()))

    comp = results.compare()
    view = pd.DataFrame(
        {
            "Scenario": comp["scenario"],
            "Median MOIC": comp["moic_p50"].map("{:.2f}x".format),
            "MOIC range (10th to 90th)": [f"{a:.2f}x to {b:.2f}x" for a, b in zip(comp["moic_p10"], comp["moic_p90"], strict=True)],
            "Median IRR": comp["irr_p50"].map("{:.1%}".format),
            "Chance of losing money": comp["loss_probability"].map("{:.0%}".format),
            "Chance of 3x or more": comp["home_run_probability"].map("{:.0%}".format),
        }
    )
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button("Download comparison (CSV)", csv_bytes(comp), file_name="scenarios.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

with tab_sens:
    st.write("Which assumption moves the result most? Each bar shows the metric with the assumption lowered and raised.")
    labels = {
        "return_adjustment": "Exit multiples",
        "loss_rate_delta": "Loss rates",
        "holding_period": "Holding periods",
    }
    metrics = {
        "Median IRR": "irr_p50",
        "Median MOIC": "moic_p50",
        "Chance of losing money": "loss_probability",
    }
    a, b, cc = st.columns([2, 1, 1])
    chosen = a.multiselect("Assumptions", list(labels), default=list(labels), format_func=labels.get)
    metric_label = b.selectbox("Metric", list(metrics))
    pct_change = cc.slider("Change (±%)", 5, 50, 25, 5)
    if chosen:
        sens_n = min(int(n_sims), 5_000)
        tornado = run_tornado(clean, sens_n, int(seed), tuple(chosen), metrics[metric_label], pct_change / 100)
        shown = tornado.assign(parameter=tornado["parameter"].map(labels))
        tfig = viz.plot_sensitivity_tornado(shown, metric=metrics[metric_label], title=f"Sensitivity: {metric_label}")
        tfig.update_xaxes(title=metric_label, tickformat=".1%" if metrics[metric_label] != "moic_p50" else ".2f")
        tfig.update_yaxes(title=None)
        chart(tfig)
        st.caption(f"Uses {sens_n:,} simulations per run.")
    else:
        st.info("Pick at least one assumption.")

# ---------------------------------------------------------------------------
# J-curve
# ---------------------------------------------------------------------------

with tab_j:
    st.write(
        "How fund value builds over time, quarter by quarter, if each company reaches its "
        "expected multiple. Winners are held at cost early and marked up along an S-curve; "
        "write-offs are recognized early; fees count toward paid-in capital."
    )
    committed = st.number_input(
        "Fund size / committed capital ($)", min_value=1e6, value=float(default_committed), step=1e6, format="%.0f",
        help="Management fees are charged on this amount.",
    )
    st.caption(f"{money(committed)} fund, {fee_pct:.2f}% annual fee = {money(committed * fee_pct / 100)} a year.")
    jdf, shape = run_jcurve(clean, float(committed), fee_pct / 100)
    m = st.columns(4)
    m[0].metric("Lowest point", money(shape["trough_nav"]), help="Lowest value of NAV plus distributions minus paid-in.")
    m[1].metric("Reached in", f"Year {shape['trough_period'] / 4:.1f}")
    m[2].metric("Breakeven", f"Year {shape['breakeven_period'] / 4:.1f}" if shape["breakeven_period"] >= 0 else "Not reached")
    final = jdf.iloc[-1]
    m[3].metric("Final net TVPI", f"{final['tvpi']:.2f}x")
    jfig = viz.plot_jcurve(jdf, show_components=True, title="Fund J-curve")
    jfig.update_xaxes(title="Quarter")
    chart(jfig)
    if clean["expected_multiple"].isna().any():
        st.caption("Companies without an expected multiple use the average multiple for their stage.")
    st.download_button("Download J-curve data (CSV)", csv_bytes(jdf), file_name="jcurve.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

with tab_port:
    portfolio = Portfolio.from_dataframe(clean)
    by = st.radio("Group by", ["stage", "sector"], horizontal=True)
    chart(viz.plot_portfolio_breakdown(portfolio, breakdown_by=by))
    conc = portfolio.concentration_analysis()
    k = st.columns(3)
    k[0].metric("Largest 3 positions", f"{conc['top_3_concentration']:.0%} of capital")
    k[1].metric("Concentration (HHI)", f"{conc['herfindahl_index']:.3f}", help="0 = spread evenly across many companies, 1 = all in one.")
    k[2].metric("Stages / sectors", f"{conc['stage_diversity']} / {conc['sector_diversity']}")

st.divider()
st.caption(
    "Built with [vc-portfolio-toolkit](https://github.com/antoniadumitriu/VC-Portfolio-Toolkit). "
    "For illustration only, not investment advice."
)
