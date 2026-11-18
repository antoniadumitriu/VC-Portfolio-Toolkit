"""
Founder Round Modeler: compare term sheets from the founder's side of the table.

Run with the main app:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vc_portfolio import STAGE_LOSS_RATES
from vc_portfolio import visualization as viz
from vc_portfolio.founder import (
    DEFAULT_EXIT_SIGMA,
    DEFAULT_MEDIAN_EXIT,
    FOUNDERS,
    CapTable,
    Convertible,
    ExitAssumptions,
    RoundInputError,
    RoundResult,
    RoundTerms,
    add_later_rounds,
    breakpoints,
    exit_for_payout,
    founder_outcome,
    minimum_pre_money,
    negotiation_levers,
    ownership_grid,
    price_round,
    waterfall,
)

st.set_page_config(page_title="Founder Round Modeler", page_icon="📈", layout="wide")

M = 1_000_000.0
INVESTOR = "New investors"
OFFERS = ("Offer A", "Offer B")
OFFER_COLORS = {"Offer A": "#3987E5", "Offer B": "#D95926"}
BRIDGE_COLORS = {
    "Existing holders": "#199E70",
    "Option pool": "#C98500",
    "SAFEs and notes": "#9085E9",
    "Antidilution": "#D55181",
}
PARTICIPATION = {
    "Non-participating": "none",
    "Participating, capped": "capped",
    "Fully participating": "full",
}
ANTI_DILUTION = {
    "Broad-based weighted average": "broad",
    "Narrow-based weighted average": "narrow",
    "Full ratchet": "full_ratchet",
    "None": "none",
}
KINDS = {
    "Post-money SAFE": "post_money_safe",
    "Pre-money SAFE": "pre_money_safe",
    "Convertible note": "note",
}
STAGE_LABELS = {
    "pre_seed": "Pre-seed",
    "seed": "Seed",
    "series_a": "Series A",
    "series_b": "Series B",
    "series_c": "Series C",
    "growth": "Growth",
}


def money(x: float) -> str:
    sign, x = ("-" if x < 0 else ""), abs(x)
    if x >= 1e9:
        return f"{sign}${x / 1e9:,.2f}B"
    if x >= 1e6:
        return f"{sign}${x / 1e6:,.1f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:,.0f}K"
    return f"{sign}${x:,.0f}"


def signed_money(x: float) -> str:
    return f"+{money(x)}" if x > 0 else money(x)


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None)


def money_axis(axis_max: float) -> dict[str, object]:
    """Round tick positions labeled in $K, $M and $B."""
    magnitude = 10 ** np.floor(np.log10(max(axis_max, 1.0)))
    step = next(
        (m * magnitude / 10 for m in (1, 2, 2.5, 5, 10) if axis_max / (m * magnitude / 10) <= 6),
        magnitude,
    )
    ticks = np.arange(0, axis_max * 1.0001, step)
    return {"tickvals": ticks.tolist(), "ticktext": [money(float(t)) for t in ticks]}


def themed(fig: go.Figure, height: int = 420) -> go.Figure:
    viz._apply_vc_theme(fig)
    fig.update_layout(
        height=height,
        margin=dict(l=70, r=30, t=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="normal"),
    )
    return fig


def default_convertibles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Name": "Seed SAFE", "Type": "Post-money SAFE", "Amount ($)": 500_000,
                "Valuation cap ($)": 6_000_000, "Discount (%)": 0.0,
                "Interest (%)": 0.0, "Years outstanding": 0.0,
            },
            {
                "Name": "Angel note", "Type": "Convertible note", "Amount ($)": 250_000,
                "Valuation cap ($)": 8_000_000, "Discount (%)": 20.0,
                "Interest (%)": 6.0, "Years outstanding": 1.5,
            },
        ]
    )


def parse_convertibles(table: pd.DataFrame) -> tuple[tuple[Convertible, ...], list[str]]:
    items, problems = [], []
    for position, row in enumerate(table.to_dict("records"), start=1):
        amount = row.get("Amount ($)")
        if pd.isna(amount) or not amount:
            continue
        name = str(row.get("Name") or "").strip()
        if not name or name == "nan":
            name = f"Security {position}"
        cap = row.get("Valuation cap ($)")
        try:
            items.append(
                Convertible(
                    name=name,
                    amount=float(amount),
                    kind=KINDS.get(row.get("Type"), "post_money_safe"),  # type: ignore[arg-type]
                    cap=None if pd.isna(cap) or not cap else float(cap),
                    discount=float(row.get("Discount (%)") or 0) / 100,
                    interest_rate=float(row.get("Interest (%)") or 0) / 100,
                    years_outstanding=float(row.get("Years outstanding") or 0),
                )
            )
        except RoundInputError as error:
            problems.append(str(error))
    return tuple(items), problems


def offer_form(label: str, defaults: dict[str, object]) -> dict[str, object]:
    k = label[-1]
    with st.container(border=True):
        st.subheader(label)
        a, b, c = st.columns(3)
        raise_m = a.number_input("Raise ($M)", 0.05, 500.0, defaults["raise"], 0.25, key=f"raise{k}")
        pre_m = b.number_input("Pre-money ($M)", 0.1, 5000.0, defaults["pre"], 0.5, key=f"pre{k}")
        pool = c.number_input(
            "Option pool (%)", 0.0, 50.0, defaults["pool"], 1.0, key=f"pool{k}",
            help="Unissued options as a share of the company after the round.",
        )
        d, e, f = st.columns([1, 1.5, 1])
        multiple = d.number_input(
            "Preference (x)", 0.0, 5.0, defaults["multiple"], 0.25, key=f"lp{k}",
            help="Liquidation preference, as a multiple of the price paid.",
        )
        kind_label = e.selectbox(
            "Preferred type", list(PARTICIPATION),
            index=list(PARTICIPATION).index(defaults["type"]), key=f"type{k}",
        )
        cap_multiple = f.number_input(
            "Total cap (x)", 0.0, 10.0, 3.0, 0.5, key=f"cap{k}",
            help="Most a capped participating investor receives before converting.",
            disabled=PARTICIPATION[kind_label] != "capped",
        )
        with st.expander("Fine print"):
            pool_pre = st.checkbox("New options come out of the pre-money", True, key=f"poolpre{k}")
            notes_pre = st.checkbox("SAFEs and notes come out of the pre-money", True, key=f"notepre{k}")
            shadow = st.checkbox(
                "SAFEs and notes get a preference on what they paid (shadow series)",
                True, key=f"shadow{k}",
            )
            dividend = st.number_input("Cumulative dividend (%)", 0.0, 20.0, 0.0, 1.0, key=f"div{k}")
            anti = st.selectbox("Antidilution", list(ANTI_DILUTION), key=f"anti{k}")
    return {
        "investment": raise_m * M,
        "pre_money": pre_m * M,
        "option_pool_target": pool / 100,
        "pool_in_pre_money": pool_pre,
        "convertibles_in_pre_money": notes_pre,
        "shadow_series": shadow,
        "liquidation_multiple": multiple,
        "participation": PARTICIPATION[kind_label],
        "participation_cap": max(cap_multiple, multiple),
        "dividend_rate": dividend / 100,
        "anti_dilution": ANTI_DILUTION[anti],
        "name": INVESTOR,
    }


def crossovers(grid: np.ndarray, a: np.ndarray, b: np.ndarray) -> list[float]:
    """Exit values where the better offer changes."""
    gap = a - b
    tolerance = 1.0
    points = []
    last_index, last_sign = None, 0
    for i, value in enumerate(gap):
        sign = 0 if abs(value) <= tolerance else int(np.sign(value))
        if sign == 0:
            continue
        if last_sign and sign != last_sign:
            j = last_index
            weight = abs(gap[j]) / (abs(gap[j]) + abs(gap[i]))
            points.append(float(grid[j] + weight * (grid[i] - grid[j])))
        last_index, last_sign = i, sign
    return points


def payoff_summary(grid: np.ndarray, a: np.ndarray, b: np.ndarray) -> str:
    gap = a - b
    active = np.abs(gap) > 1.0
    if not active.any():
        return "Both offers pay founders the same at every exit value shown."
    crossing = crossovers(grid, a, b)
    first = OFFERS[0] if gap[active][0] > 0 else OFFERS[1]
    second = OFFERS[1] if first == OFFERS[0] else OFFERS[0]
    if not crossing:
        return f"**{first} pays founders more at every exit value shown where the offers differ.**"
    text = f"**{first} pays founders more below {money(crossing[0])}; {second} pays more above it.**"
    if len(crossing) > 1:
        rest = ", ".join(money(x) for x in crossing[1:])
        text += f" The lead changes again at {rest}."
    return text


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.page_link("streamlit_app.py", label="Portfolio analysis")
    st.page_link("pages/founder_round_modeler.py", label="Founder round modeler")

    st.header("Company today")
    founder_shares = st.number_input("Founder shares", 1_000, 10**10, 8_000_000, 100_000)
    other_common = st.number_input("Other shares and granted options", 0, 10**10, 500_000, 50_000)
    unissued = st.number_input("Unissued option pool", 0, 10**10, 1_000_000, 50_000)

    st.header("Later rounds")
    st.caption("Financing you expect between this round and an exit.")
    later_capital = st.number_input("Capital raised ($M)", 0.0, 5000.0, 10.0, 1.0)
    later_dilution = st.slider("Dilution to everyone (%)", 0, 80, 25)
    years_to_exit = st.slider("Years to exit", 1, 15, 5)
    if later_capital > 0 and later_dilution > 0:
        st.caption(f"Implies about {money(later_capital * M / (later_dilution / 100))} post-money.")

    st.header("Exit odds")
    stage = st.selectbox(
        "Company stage", list(STAGE_LABELS), index=1, format_func=STAGE_LABELS.get
    )
    no_exit = st.slider(
        "Chance of no proceeds (%)", 0, 95, int(STAGE_LOSS_RATES[stage] * 100),
        key=f"no_exit_{stage}",
    )
    median_exit = st.number_input("Median exit if one happens ($M)", 1.0, 5000.0, DEFAULT_MEDIAN_EXIT / M, 5.0)
    spread = st.slider("Spread of exit values", 0.5, 2.5, DEFAULT_EXIT_SIGMA, 0.1)
    st.caption(
        "The default median and spread echo PitchBook's 2016 exit data: about half of "
        "exits under $50M and about 35% under $25M. Replace them with your own view."
    )

st.title("Founder Round Modeler")
st.write(
    "A term sheet is more than its valuation. Enter two offers to see what each does to your "
    "ownership, and what you actually take home at each exit value."
)

with st.expander("SAFEs and notes outstanding", expanded=False):
    st.caption("Leave the cap blank if there is none. Clear the amount to remove a row.")
    edited = st.data_editor(
        default_convertibles(),
        key="convertibles",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn(options=list(KINDS), required=True),
            "Amount ($)": st.column_config.NumberColumn(min_value=0, format="localized"),
            "Valuation cap ($)": st.column_config.NumberColumn(min_value=0, format="localized"),
            "Discount (%)": st.column_config.NumberColumn(min_value=0.0, max_value=99.0),
            "Interest (%)": st.column_config.NumberColumn(min_value=0.0, max_value=50.0),
            "Years outstanding": st.column_config.NumberColumn(min_value=0.0, max_value=20.0),
        },
    )
convertibles, problems = parse_convertibles(edited)
for problem in problems:
    st.error(problem)

left, right = st.columns(2)
with left:
    form_a = offer_form(
        "Offer A",
        {"raise": 3.0, "pre": 12.0, "pool": 10.0, "multiple": 1.0, "type": "Non-participating"},
    )
with right:
    form_b = offer_form(
        "Offer B",
        {"raise": 3.0, "pre": 18.0, "pool": 15.0, "multiple": 1.5, "type": "Fully participating"},
    )

try:
    company = CapTable(float(founder_shares), float(other_common), float(unissued))
    terms = {
        "Offer A": RoundTerms(convertibles=convertibles, **form_a),  # type: ignore[arg-type]
        "Offer B": RoundTerms(convertibles=convertibles, **form_b),  # type: ignore[arg-type]
    }
    results: dict[str, RoundResult] = {k: price_round(company, t) for k, t in terms.items()}
    assumptions = ExitAssumptions(no_exit / 100, median_exit * M, spread)
except RoundInputError as error:
    st.error(str(error))
    st.stop()


def to_exit_cap(cap: CapTable) -> CapTable:
    return add_later_rounds(cap, later_capital * M, later_dilution / 100, years_to_exit)


exit_caps = {k: to_exit_cap(r.after) for k, r in results.items()}
exits = assumptions.draw(50_000)
outcomes = {k: founder_outcome(c, exits) for k, c in exit_caps.items()}

focus = st.radio(
    "Offer to examine in the pricing, negotiation, investor and down round views",
    OFFERS, horizontal=True,
)
focus_result, focus_terms, focus_exit_cap = results[focus], terms[focus], exit_caps[focus]

tab_round, tab_payout, tab_odds, tab_levers, tab_investor, tab_down = st.tabs(
    ["The round", "Who gets paid", "Exit odds", "What to negotiate", "Investor math", "Down round"]
)

# ---------------------------------------------------------------------------
# The round
# ---------------------------------------------------------------------------

with tab_round:
    columns = st.columns(2)
    for column, name in zip(columns, OFFERS):
        r = results[name]
        with column:
            st.subheader(name)
            a, b = st.columns(2)
            a.metric("Founders after the round", f"{r.founder_ownership:.1%}")
            b.metric("Effective pre-money", money(r.effective_pre_money))
            c, d = st.columns(2)
            c.metric("Price per share", f"${r.price:,.3f}")
            d.metric("Post-money", money(r.post_money))

    a_r, b_r = results["Offer A"], results["Offer B"]
    headline_gap = b_r.terms.pre_money - a_r.terms.pre_money
    effective_gap = b_r.effective_pre_money - a_r.effective_pre_money
    st.write(
        f"Going from Offer A to Offer B changes the headline pre-money by "
        f"{signed_money(headline_gap)}, and what existing holders keep by "
        f"{signed_money(effective_gap)}."
    )

    st.subheader("Where the headline valuation goes")
    fig = go.Figure()
    segments: dict[str, list[float]] = {
        "Existing holders": [results[n].effective_pre_money for n in OFFERS],
        "Option pool": [-results[n].bridge["Option pool"] for n in OFFERS],
        "SAFEs and notes": [-results[n].bridge["SAFEs and notes"] for n in OFFERS],
        "Antidilution": [-results[n].bridge["Antidilution"] for n in OFFERS],
    }
    for label, values in segments.items():
        if max(values) <= 0:
            continue
        fig.add_bar(
            name=label, y=list(OFFERS), x=values, orientation="h",
            marker=dict(color=BRIDGE_COLORS[label], line=dict(color=viz._VC_COLORS["paper"], width=2)),
            text=[money(v) for v in values], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#FFFFFF"),
            hovertemplate="%{y}<br>" + label + ": %{text}<extra></extra>",
        )
    fig.update_layout(barmode="stack", bargap=0.45)
    largest_pre = max(t.pre_money for t in terms.values())
    fig.update_xaxes(title="Pre-money valuation", **money_axis(largest_pre))
    fig.update_yaxes(autorange="reversed")
    chart(themed(fig, 260))

    st.subheader("Cap table")
    holders = list(company.holdings())
    for name in OFFERS:
        holders += [h for h in results[name].after.holdings() if h not in holders]
    table = pd.DataFrame(
        {
            "Holder": holders,
            "Today": [100 * company.ownership().get(h, 0.0) for h in holders],
            "After Offer A": [100 * a_r.after.ownership().get(h, 0.0) for h in holders],
            "After Offer B": [100 * b_r.after.ownership().get(h, 0.0) for h in holders],
        }
    )
    percent = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={c: percent for c in table.columns[1:]},
    )

    if focus_result.conversions:
        st.subheader(f"How SAFEs and notes convert under {focus}")
        conversion_rows = pd.DataFrame(
            [
                {
                    "Security": c.name,
                    "Invested": money(c.invested),
                    "Converts at": f"${c.price:,.3f}",
                    "Set by": c.basis,
                    "Ownership": 100 * c.shares / focus_result.after.fully_diluted,
                    "Liquidation preference": money(c.liquidation_preference),
                    "Above money invested": money(c.preference_overhang),
                }
                for c in focus_result.conversions
            ]
        )
        st.dataframe(
            conversion_rows, hide_index=True, use_container_width=True,
            column_config={"Ownership": percent},
        )

    st.subheader(f"Find your price under {focus}")
    target = st.slider(
        "Founders should own at least (%)", 5, 95,
        int(round(focus_result.founder_ownership * 100)), key=f"target_{focus}",
    )
    needed = minimum_pre_money(company, focus_terms, target / 100)
    if needed is None:
        st.warning(
            f"No valuation gets founders to {target}% with these terms: the pool and "
            "converting securities take too much on their own."
        )
    else:
        st.write(
            f"With a {money(focus_terms.investment)} raise and these terms, you need a pre-money "
            f"of at least **{money(needed)}**."
        )
    raises = [focus_terms.investment * x for x in (0.5, 0.75, 1.0, 1.25, 1.5)]
    pres = [focus_terms.pre_money * x for x in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6)]
    grid = ownership_grid(company, focus_terms, raises, pres)
    heatmap = go.Figure(
        go.Heatmap(
            z=grid * 100,
            x=[money(p) for p in pres],
            y=[money(v) for v in raises],
            colorscale=[[0, "#14304F"], [1, "#2F73C7"]],
            xgap=2, ygap=2,
            textfont=dict(color="#FFFFFF"),
            text=[[("" if np.isnan(v) else f"{v:.1%}") for v in row] for row in grid],
            texttemplate="%{text}",
            colorbar=dict(title="Founders (%)"),
            hovertemplate="Raise %{y} at %{x} pre-money<br>Founders own %{text}<extra></extra>",
        )
    )
    heatmap.update_xaxes(title="Pre-money valuation", type="category")
    heatmap.update_yaxes(title="Amount raised", type="category")
    heatmap.add_shape(
        type="rect", x0=1.5, x1=2.5, y0=1.5, y1=2.5,
        line=dict(color="#FFFFFF", width=2),
    )
    chart(themed(heatmap, 340))
    st.caption("The outlined cell is this offer. Blank cells are terms that cannot fit.")

# ---------------------------------------------------------------------------
# Who gets paid
# ---------------------------------------------------------------------------

with tab_payout:
    top = st.select_slider(
        "Largest exit to show", options=[25, 50, 100, 250, 500, 1000], value=100,
        format_func=lambda v: money(v * M),
    )
    values = np.linspace(0, top * M, 1001)
    curves = {k: waterfall(c, values).founders for k, c in exit_caps.items()}
    st.markdown(payoff_summary(values, curves["Offer A"], curves["Offer B"]))

    fig = go.Figure()
    for name in OFFERS:
        fig.add_scatter(
            x=values, y=curves[name], name=name, mode="lines",
            line=dict(color=OFFER_COLORS[name], width=2),
            text=[money(float(v)) for v in curves[name]],
            hovertemplate=name + ": %{text}<extra></extra>",
        )
        fig.add_annotation(
            x=values[-1], y=curves[name][-1], text=name, showarrow=False,
            xanchor="right", yanchor="bottom" if name == "Offer A" else "top",
            font=dict(color=viz._VC_COLORS["text"]), yshift=6 if name == "Offer A" else -6,
        )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(
        title="Exit value", range=[-0.015 * top * M, top * M], hoverformat="$,.0f",
        **money_axis(top * M),
    )
    highest = max(float(c.max()) for c in curves.values())
    fig.update_yaxes(title="Paid to founders", rangemode="tozero", **money_axis(highest))
    chart(themed(fig))

    marks = np.array([10, 25, 50, 100, 250, 500]) * M
    rows = pd.DataFrame(
        {
            "Exit value": [money(v) for v in marks],
            **{k: [money(x) for x in waterfall(c, marks).founders] for k, c in exit_caps.items()},
            "B minus A": [
                signed_money(b - a)
                for a, b in zip(
                    waterfall(exit_caps["Offer A"], marks).founders,
                    waterfall(exit_caps["Offer B"], marks).founders,
                )
            ],
        }
    )
    st.dataframe(rows, hide_index=True, use_container_width=True)

    with st.expander(f"Where the waterfall changes shape under {focus}"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Exit value": money(float(e["exit_value"])), "What happens": e["event"]}
                    for e in breakpoints(focus_exit_cap)
                ]
            ),
            hide_index=True, use_container_width=True,
        )
    st.caption(
        f"Includes {money(later_capital * M)} of later financing, {later_dilution}% later dilution "
        f"and {years_to_exit} years of any cumulative dividends. All options count as exercised, "
        "which is the cautious case for founders."
    )

# ---------------------------------------------------------------------------
# Exit odds
# ---------------------------------------------------------------------------

with tab_odds:
    goal = st.number_input("Founder payout you care about, in total ($M)", 0.5, 5000.0, 10.0, 1.0)
    columns = st.columns(2)
    for column, name in zip(columns, OFFERS):
        o = outcomes[name]
        with column:
            st.subheader(name)
            a, b, c = st.columns(3)
            a.metric(f"Chance of {money(goal * M)}+", f"{np.mean(o.payouts >= goal * M):.1%}")
            b.metric("Chance of nothing", f"{o.probability_nothing:.1%}")
            c.metric("Median if there is an exit", money(o.median_if_exit))

    thresholds = np.logspace(4.85, 9.5, 240)
    fig = go.Figure()
    for name in OFFERS:
        ordered = np.sort(outcomes[name].payouts)
        share = 1 - np.searchsorted(ordered, thresholds, side="left") / ordered.size
        fig.add_scatter(
            x=thresholds, y=share, name=name, mode="lines",
            line=dict(color=OFFER_COLORS[name], width=2),
            hovertemplate=name + ": %{y:.1%}<extra></extra>",
        )
    fig.add_vline(x=goal * M, line=dict(color=viz._VC_COLORS["text_secondary"], width=1, dash="dot"))
    fig.update_layout(hovermode="x unified")
    log_ticks = [1e5, 1e6, 1e7, 1e8, 1e9]
    fig.update_xaxes(
        title="Founders receive at least", type="log", tickvals=log_ticks,
        ticktext=[money(t) for t in log_ticks], hoverformat="$,.0f",
    )
    fig.update_yaxes(title="Chance", tickformat=".0%", rangemode="tozero")
    chart(themed(fig))
    st.caption(
        "Both offers face the same 50,000 simulated exits, so differences come from the terms. "
        "The model assumes the terms do not change how the company performs. These are "
        "illustrations of your assumptions, not forecasts."
    )

# ---------------------------------------------------------------------------
# Negotiation
# ---------------------------------------------------------------------------

with tab_levers:
    st.write(
        f"What founders gain under {focus} if each term alone moves in their favor. "
        "A 10% higher valuation is included for scale."
    )
    levers = sorted(
        negotiation_levers(company, focus_terms, exits, median_exit * M, to_exit_cap),
        key=lambda lever: -lever.typical_exit_change,
    )
    fig = go.Figure(
        go.Bar(
            y=[lever.name for lever in levers],
            x=[lever.typical_exit_change for lever in levers],
            orientation="h",
            marker=dict(color=OFFER_COLORS[focus]),
            width=0.5,
            text=[signed_money(lever.typical_exit_change) for lever in levers],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.update_xaxes(
        title=f"Extra founder proceeds at a {money(median_exit * M)} exit",
        **money_axis(max((lever.typical_exit_change for lever in levers), default=1.0)),
    )
    fig.update_yaxes(autorange="reversed")
    chart(themed(fig, max(240, 70 * len(levers))).update_layout(margin=dict(l=170, r=80)))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Term": lever.name,
                    "Change": lever.change,
                    "Founder stake": f"{lever.ownership_change * 100:+.1f} pts",
                    f"At a {money(median_exit * M)} exit": signed_money(lever.typical_exit_change),
                    "Average across simulated exits": signed_money(lever.expected_change),
                }
                for lever in levers
            ]
        ),
        hide_index=True, use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Investor math
# ---------------------------------------------------------------------------

with tab_investor:
    st.write(f"What has to happen for this round to work for the investor under {focus}.")
    fund_size = st.number_input("Investor's fund size ($M)", 1.0, 10_000.0, 75.0, 5.0)
    investment = focus_terms.investment
    investor_rows = []
    goals = [("Money back", investment), ("3x", 3 * investment), ("10x", 10 * investment)]
    goals.append(("Returns the whole fund", fund_size * M))
    for label, amount in goals:
        needed_exit = exit_for_payout(focus_exit_cap, INVESTOR, amount)
        investor_rows.append(
            {
                "Goal": label,
                "Investor needs": money(amount),
                "Exit value required": money(needed_exit) if needed_exit is not None else "Out of reach",
                "Chance under your assumptions": (
                    f"{assumptions.probability_at_least(needed_exit):.1%}"
                    if needed_exit is not None else "0.0%"
                ),
            }
        )
    st.dataframe(pd.DataFrame(investor_rows), hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    a.metric("Investor stake after this round", f"{focus_result.investor_ownership:.1%}")
    later_stake = focus_exit_cap.ownership().get(INVESTOR, 0.0)
    b.metric("Investor stake at exit", f"{later_stake:.1%}")
    st.caption(
        "Investors usually need each check to be able to matter to the fund. A higher price "
        "raises the exit they must believe in, which is often the real limit on valuation."
    )

# ---------------------------------------------------------------------------
# Down round
# ---------------------------------------------------------------------------

with tab_down:
    st.write(
        f"Suppose the next round comes in lower than {focus}. Antidilution terms decide "
        "who absorbs the drop."
    )
    a, b = st.columns(2)
    next_raise = a.number_input("Next round raise ($M)", 0.1, 1000.0, 3.0, 0.5)
    default_pre = max(0.5, round(focus_result.post_money / M * 0.6, 1))
    next_pre = b.number_input(
        "Next round pre-money ($M)", 0.1, 5000.0, default_pre, 0.5, key=f"next_pre_{focus}"
    )
    rows, stakes = [], {}
    for label, kind in ANTI_DILUTION.items():
        try:
            outcome = price_round(
                focus_result.after.with_anti_dilution(kind),  # type: ignore[arg-type]
                RoundTerms(next_raise * M, next_pre * M, name="Next round"),
            )
        except RoundInputError as error:
            st.error(str(error))
            st.stop()
        stakes[label] = outcome.founder_ownership
        rows.append(
            {
                "Protection": label,
                "Founders after": 100 * outcome.founder_ownership,
                "Next round price": f"${outcome.price:,.3f}",
                "Extra shares to earlier investors": f"{outcome.antidilution_shares:,.0f}",
            }
        )
    if next_pre * M >= focus_result.post_money:
        st.info(
            f"This is not a down round: the pre-money is at or above the {money(focus_result.post_money)} "
            "post-money of the current round."
        )
    rows.sort(key=lambda row: -row["Founders after"])
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={"Founders after": st.column_config.NumberColumn(format="%.1f%%")},
    )
    spread_points = (stakes["None"] - stakes["Full ratchet"]) * 100
    st.write(
        f"The choice of protection moves founder ownership by up to **{spread_points:.1f} points** "
        "in this scenario. In practice these rights are often renegotiated in a down round."
    )

st.divider()
st.caption(
    "For education and planning only, not legal, tax or investment advice. Have startup counsel "
    "review any term sheet. Taxes, debt, deal fees and option strike proceeds are ignored. "
    "Mechanics follow Dave Lishego's The Founder's Guide to VC Finance and Y Combinator's "
    "post-money SAFE documentation."
)
