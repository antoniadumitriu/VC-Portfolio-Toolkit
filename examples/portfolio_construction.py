"""
portfolio_construction.py — Monte Carlo simulation and scenario analysis.

Demonstrates:
- Building a diversified VC portfolio
- Running Monte Carlo exit simulations
- Scenario comparison (Bear/Base/Bull)
- Sensitivity analysis and tornado chart
- Portfolio concentration metrics

Run:
    python examples/portfolio_construction.py
"""
from __future__ import annotations

import numpy as np

from vc_portfolio import (
    BASE_SCENARIO,
    BEAR_SCENARIO,
    BULL_SCENARIO,
    Company,
    MonteCarloSimulator,
    Portfolio,
    ScenarioSet,
    SensitivityAnalysis,
)
from vc_portfolio import visualization as viz


def build_portfolio() -> Portfolio:
    """Construct a diversified 10-company portfolio."""
    companies = [
        # Pre-seed / Seed
        Company("QuantumAI",     2_000_000, "pre_seed", "AI/ML",       8_000_000,  0, 9,  0.20),
        Company("NanoHealth",    1_500_000, "pre_seed", "biotech",      6_000_000,  1, 10, 0.22),
        Company("ClimateOS",     3_000_000, "seed",     "climate",      15_000_000, 0, 8,  0.15),
        Company("SecureVault",   2_500_000, "seed",     "cybersecurity",12_000_000, 1, 7,  0.14),
        Company("DataMesh",      2_000_000, "seed",     "saas",         10_000_000, 2, 8,  0.15),
        # Series A
        Company("HealthStack",   5_000_000, "series_a", "healthtech",   25_000_000, 1, 7,  0.12),
        Company("FinanceFlow",   4_000_000, "series_a", "fintech",      20_000_000, 0, 6,  0.13),
        Company("LogiRoute",     3_500_000, "series_a", "logistics",    18_000_000, 2, 7,  0.14),
        # Series B
        Company("EnterpriseX",   7_000_000, "series_b", "saas",         50_000_000, 0, 5,  0.09),
        Company("GridPower",     6_000_000, "series_b", "energy",       45_000_000, 1, 6,  0.10),
    ]
    return Portfolio(seed=42).add_companies(companies)


def print_percentile_table(results: dict, label: str = "Portfolio Returns") -> None:
    """Pretty-print simulation results."""
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    print(f"  {'Percentile':<15} {'MOIC':>10} {'IRR':>10}")
    print(f"  {'─' * 38}")
    for p in [10, 25, 50, 75, 90]:
        moic = results[f"moic_p{p}"]
        irr = results[f"irr_p{p}"]
        print(f"  P{p:<14} {moic:>9.2f}x {irr:>9.1%}")
    print(f"{'─' * 55}")
    print(f"  Loss Probability:    {results['loss_probability']:>10.1%}")
    print(f"  Home Run (>3x):      {results['home_run_probability']:>10.1%}")


def main() -> None:
    # -------------------------------------------------------------------
    # 1. Build portfolio
    # -------------------------------------------------------------------
    portfolio = build_portfolio()
    print(portfolio)

    # -------------------------------------------------------------------
    # 2. Portfolio analytics
    # -------------------------------------------------------------------
    print("\n=== Portfolio Breakdown ===")
    df = portfolio.portfolio_breakdown()
    print(
        df[["company", "stage", "sector", "investment_amount", "portfolio_weight"]]
        .to_string(index=False)
    )

    conc = portfolio.concentration_analysis()
    print(f"\nHerfindahl Index: {conc['herfindahl_index']:.3f} (0=diverse, 1=concentrated)")
    print(f"Top-3 Concentration: {conc['top_3_concentration']:.1%}")
    print(f"Stage diversity: {conc['stage_diversity']} distinct stages")
    print(f"Sector diversity: {conc['sector_diversity']} distinct sectors")

    # -------------------------------------------------------------------
    # 3. Monte Carlo simulation
    # -------------------------------------------------------------------
    print("\n=== Monte Carlo Simulation (10,000 runs) ===")
    sim = MonteCarloSimulator(portfolio, n_simulations=10_000, seed=42)
    results = sim.run()

    print_percentile_table(results, "Base Case Portfolio Returns")

    # Value at Risk
    var_95 = sim.compute_var(confidence=0.95, metric="moic")
    es_95 = sim.expected_shortfall(confidence=0.95, metric="moic")
    print(f"\n95% VaR (MOIC):  {var_95:.2f}x  (worst 5% of outcomes)")
    print(f"95% CVaR (MOIC): {es_95:.2f}x  (average of worst 5%)")

    # -------------------------------------------------------------------
    # 4. Scenario analysis
    # -------------------------------------------------------------------
    print("\n=== Scenario Analysis (Bear / Base / Bull) ===")
    scenario_results = ScenarioSet([BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO]).run(
        portfolio=portfolio,
        n_simulations=5_000,
    )

    comparison = scenario_results.compare()
    print(
        comparison[["scenario", "moic_p10", "moic_p50", "moic_p90", "loss_probability"]]
        .to_string(index=False)
    )

    # -------------------------------------------------------------------
    # 5. Sensitivity analysis
    # -------------------------------------------------------------------
    print("\n=== Sensitivity Analysis ===")
    sa = SensitivityAnalysis(portfolio, n_simulations=2_000, seed=42)

    # Sweep return adjustment from 0.5x to 2.0x
    sweep_df = sa.sweep(
        "return_adjustment",
        [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        metric="moic_p50",
    )
    print("\nReturn Multiplier Sweep (Median MOIC):")
    print(sweep_df.to_string(index=False))

    tornado_df = sa.tornado(
        ["return_adjustment", "loss_rate_delta", "exit_multiple_adjustment"],
        metric="moic_p50",
        pct_change=0.25,
    )
    print("\nTornado Analysis (±25% change):")
    print(tornado_df.to_string(index=False))

    # -------------------------------------------------------------------
    # 6. Visualizations
    # -------------------------------------------------------------------
    print("\nOpening return distribution chart...")
    fig_dist = viz.plot_return_distribution(results, metric="moic")
    fig_dist.show()

    print("Opening scenario fan chart...")
    fig_fan = viz.plot_scenario_fan(scenario_results, metric="moic")
    fig_fan.show()

    print("Opening portfolio breakdown (by stage)...")
    fig_breakdown = viz.plot_portfolio_breakdown(portfolio, breakdown_by="stage")
    fig_breakdown.show()

    print("Opening tornado chart...")
    fig_tornado = viz.plot_sensitivity_tornado(tornado_df, metric="moic_p50")
    fig_tornado.show()


if __name__ == "__main__":
    main()
