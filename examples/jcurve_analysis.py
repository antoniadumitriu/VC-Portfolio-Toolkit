"""
jcurve_analysis.py — J-curve NAV projection and scenario fan chart.

Demonstrates:
- Building a fund's company timeline
- Projecting NAV with S-curve appreciation
- Identifying J-curve shape (trough, breakeven, peak)
- Visualizing the J-curve with components
- Running bear/base/bull fan chart across scenarios

Run:
    python examples/jcurve_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vc_portfolio import CompanyTimeline, JCurve
from vc_portfolio import visualization as viz


def build_jcurve() -> JCurve:
    """
    Construct a realistic 10-year fund J-curve with 8 portfolio companies.

    Entry periods and exit multiples are calibrated to typical VC dynamics:
    - Early vintage companies: high risk, potential for high returns
    - Mid-vintage: Series A follow-ons with moderate multiples
    - Late vintage: Series B positions with lower risk/return
    """
    timelines = [
        # Early vintage — seed investments, long hold, high potential
        CompanyTimeline(
            entry_period=0,
            exit_period=28,  # ~7 years (quarterly)
            initial_investment=8_000_000,
            exit_multiple=8.0,  # star outcome
            company_name="AlphaWave (Seed → Series C)",
            follow_on_investments={8: 3_000_000, 16: 4_000_000},
        ),
        CompanyTimeline(
            entry_period=2,
            exit_period=24,
            initial_investment=5_000_000,
            exit_multiple=0.0,  # complete loss
            company_name="BetaLoss (Seed → Write-off)",
        ),
        CompanyTimeline(
            entry_period=4,
            exit_period=22,
            initial_investment=6_000_000,
            exit_multiple=3.5,
            company_name="GammaTech (Seed → Series B)",
            follow_on_investments={10: 2_500_000},
        ),
        # Mid-vintage — Series A investments
        CompanyTimeline(
            entry_period=6,
            exit_period=26,
            initial_investment=10_000_000,
            exit_multiple=4.0,
            company_name="DeltaHealth (Series A → IPO)",
            follow_on_investments={14: 5_000_000, 20: 3_000_000},
        ),
        CompanyTimeline(
            entry_period=8,
            exit_period=28,
            initial_investment=7_000_000,
            exit_multiple=2.0,  # moderate return
            company_name="EpsilonSaaS (Series A)",
        ),
        CompanyTimeline(
            entry_period=10,
            exit_period=30,
            initial_investment=5_000_000,
            exit_multiple=1.2,  # zombie return
            company_name="ZetaConsumer (Series A → Zombie)",
        ),
        # Late vintage — Series B positions
        CompanyTimeline(
            entry_period=12,
            exit_period=32,
            initial_investment=12_000_000,
            exit_multiple=3.0,
            company_name="EtaFintech (Series B → M&A)",
        ),
        CompanyTimeline(
            entry_period=16,
            exit_period=36,
            initial_investment=8_000_000,
            exit_multiple=2.5,
            company_name="ThetaClimate (Series B)",
        ),
    ]

    return JCurve(
        n_periods=40,  # 10 years quarterly
        fee_rate=0.02,
        committed_capital=100_000_000,
        period_unit="quarter",
    ).add_companies(timelines)


def main() -> None:
    # -------------------------------------------------------------------
    # 1. Build and project
    # -------------------------------------------------------------------
    jcurve = build_jcurve()
    print(jcurve)
    print(f"Total companies: {len(jcurve)}")

    df = jcurve.project()

    # -------------------------------------------------------------------
    # 2. J-curve shape analysis
    # -------------------------------------------------------------------
    shape = jcurve.get_jcurve_shape()

    print("\n=== J-Curve Shape Analysis ===")
    print(f"  Trough:    Period {shape['trough_period']} "
          f"(NAV position: ${shape['trough_nav']:,.0f})")
    print(f"  Breakeven: Period {shape['breakeven_period']} "
          f"({'Year ' + str(shape['breakeven_period'] // 4) if shape['breakeven_period'] > 0 else 'Not reached'})")
    print(f"  Peak NAV:  ${shape['peak_nav']:,.0f} at Period {shape['peak_period']}")

    # -------------------------------------------------------------------
    # 3. Period summary table (every 4 quarters)
    # -------------------------------------------------------------------
    print("\n=== Quarterly Fund Progression (every 4 quarters) ===")
    annual_df = df[df["period"] % 4 == 0].copy()
    annual_df["year"] = annual_df["period"] // 4

    display = annual_df[["year", "nav", "cumulative_called", "cumulative_distributions", "tvpi", "dpi"]].copy()
    display.columns = ["Year", "NAV", "Called", "Distributions", "TVPI", "DPI"]

    # Format money columns
    for col in ["NAV", "Called", "Distributions"]:
        display[col] = display[col].map(lambda x: f"${x/1e6:.1f}M")
    display["TVPI"] = display["TVPI"].map(lambda x: f"{x:.2f}x")
    display["DPI"] = display["DPI"].map(lambda x: f"{x:.2f}x")

    print(display.to_string(index=False))

    # -------------------------------------------------------------------
    # 4. Rolling IRR at key milestones
    # -------------------------------------------------------------------
    print("\n=== Rolling IRR at Key Milestones ===")
    milestones = [8, 16, 24, 32, 36, 39]
    for period in milestones:
        if period < len(df):
            irr = df.loc[period, "rolling_irr"]
            tvpi = df.loc[period, "tvpi"]
            irr_str = f"{irr:.1%}" if not np.isnan(irr) else "n/a"
            print(f"  Q{period:2d} (Year {period//4}): IRR={irr_str:>8}, TVPI={tvpi:.2f}x")

    # -------------------------------------------------------------------
    # 5. Visualization
    # -------------------------------------------------------------------
    print("\nOpening J-curve chart with components...")
    fig_jcurve = viz.plot_jcurve(
        df,
        show_components=True,
        title="VC Fund J-Curve — NAV Progression with S-Curve Appreciation",
    )
    fig_jcurve.show()

    # -------------------------------------------------------------------
    # 6. Scenario fan chart using ScenarioSet on a matching Portfolio
    # -------------------------------------------------------------------
    print("Building scenario fan chart...")

    # Convert JCurve timelines to Portfolio companies for ScenarioSet
    from vc_portfolio import Company, Portfolio, ScenarioSet
    from vc_portfolio import BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO

    # Approximate portfolio from JCurve timelines (using quarterly holding periods → years)
    companies = []
    for tl in jcurve._timelines:
        # Skip zero-multiple companies (write-offs handled by loss rate in simulation)
        eff_multiple = max(tl.exit_multiple, 0.01)
        companies.append(
            Company(
                name=tl.company_name,
                investment_amount=tl.total_investment,
                stage="seed" if tl.entry_period < 8 else "series_a" if tl.entry_period < 14 else "series_b",
                sector="diversified",
                entry_valuation=tl.total_investment * 5,  # approximate
                entry_period=tl.entry_period // 4,  # convert quarters → years
                expected_exit_period=tl.exit_period // 4,
                ownership_pct=0.10,
            )
        )

    portfolio = Portfolio(seed=42).add_companies(companies)

    scenario_results = ScenarioSet([BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO]).run(
        portfolio=portfolio,
        n_simulations=5_000,
    )

    print("\nScenario comparison:")
    print(scenario_results.compare()[["scenario", "moic_p50", "irr_p50", "loss_probability"]].to_string(index=False))

    fig_fan = viz.plot_scenario_fan(
        scenario_results,
        metric="moic",
        title="VC Fund Scenarios — Bear/Base/Bull MOIC Distribution",
    )
    fig_fan.show()


if __name__ == "__main__":
    main()
