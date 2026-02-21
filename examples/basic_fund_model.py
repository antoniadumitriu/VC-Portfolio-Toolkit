"""
basic_fund_model.py — Demonstrates Fund modeling with cash flow ledger.

Run:
    python examples/basic_fund_model.py
"""
from __future__ import annotations

import numpy as np

from vc_portfolio import Fund, FundConfig
from vc_portfolio import visualization as viz


def main() -> None:
    # -------------------------------------------------------------------
    # 1. Configure a 10-year fund
    # -------------------------------------------------------------------
    config = FundConfig(
        name="Acme Ventures Fund I",
        committed_capital=100_000_000,
        vintage_year=2020,
        fee_rate=0.02,
        carry_rate=0.20,
        hurdle_rate=0.08,
        investment_period=5,
        fund_life=10,
        step_down_rate=0.0025,
        period_unit="year",
    )

    # -------------------------------------------------------------------
    # 2. Build fund with method chaining
    # -------------------------------------------------------------------
    fund = (
        Fund(config)
        # Deploy capital over investment period
        .deploy_capital(
            {
                0: 15_000_000,  # Year 0: seed investments
                1: 20_000_000,  # Year 1: Series A follow-ons
                2: 15_000_000,  # Year 2: new positions
                3: 10_000_000,  # Year 3: reserve deployment
                4:  5_000_000,  # Year 4: final investments
            }
        )
        # Distributions from exits
        .add_distribution(10_000_000, period=4)   # Early exit (1 company)
        .add_distribution(25_000_000, period=6)   # Mid-fund exits
        .add_distribution(35_000_000, period=8)   # Primary exit window
        .add_distribution(20_000_000, period=9)   # Late distributions
        # NAV marks (mark-to-market)
        .set_nav(3, 55_000_000)   # Early marks
        .set_nav(5, 95_000_000)   # Peak unrealized value
        .set_nav(7, 70_000_000)   # Post-first-exits
        .set_nav(9, 15_000_000)   # Wind-down
    )

    # -------------------------------------------------------------------
    # 3. Print summary
    # -------------------------------------------------------------------
    summary = fund.summary()

    print("=" * 60)
    print(f"  {summary['name']} — Fund Summary")
    print("=" * 60)
    print(f"  Vintage Year:       {summary['vintage_year']}")
    print(f"  Committed Capital:  ${summary['committed_capital']:>15,.0f}")
    print(f"  Total Invested:     ${summary['total_invested']:>15,.0f}")
    print(f"  Total Distributions:${summary['total_distributions']:>15,.0f}")
    print(f"  Current NAV:        ${summary['current_nav']:>15,.0f}")
    print(f"  Total Fees:         ${summary['total_fees']:>15,.0f}")
    print(f"  Carry (GP):         ${summary['carry']:>15,.0f}")
    print("-" * 60)
    print(f"  Gross IRR:          {summary['irr_gross']:>14.1%}")
    print(f"  Net IRR (LP):       {summary['irr_net']:>14.1%}")
    print(f"  TVPI:               {summary['tvpi']:>14.2f}x")
    print(f"  DPI:                {summary['dpi']:>14.2f}x")
    print(f"  RVPI:               {summary['rvpi']:>14.2f}x")
    print(f"  MOIC:               {summary['moic']:>14.2f}x")
    print("=" * 60)

    # -------------------------------------------------------------------
    # 4. Show cash flow table
    # -------------------------------------------------------------------
    cf_df = fund.get_cashflows()
    print("\nCash Flow Table:")
    print(
        cf_df[["period", "capital_called", "distributions", "fees", "nav", "net_cashflow"]]
        .to_string(
            index=False,
            float_format=lambda x: f"${x:>12,.0f}" if x != 0 else "$           0",
        )
    )

    # -------------------------------------------------------------------
    # 5. PME vs 8% annual benchmark
    # -------------------------------------------------------------------
    from vc_portfolio.metrics import pme

    index_returns = np.full(10, 0.08)  # 8% per year (S&P 500 approx)
    lp_cfs = np.array(fund.get_lp_cashflows())
    pme_results = pme(lp_cfs, index_returns)

    print("\nPublic Market Equivalent (vs 8% benchmark):")
    print(f"  KS-PME:      {pme_results['ks_pme']:.3f}  (>1.0 = outperforms)")
    print(f"  PME Alpha:   {pme_results['pme_alpha']:.1%} annualized")
    print(f"  Direct Alpha:{pme_results['direct_alpha']:.1%} annualized")

    # -------------------------------------------------------------------
    # 6. Visualize
    # -------------------------------------------------------------------
    print("\nOpening interactive cash flow chart...")
    fig = viz.plot_cash_flows(fund, show_cumulative=True)
    fig.update_layout(title="Acme Ventures Fund I — Cash Flows")
    fig.show()

    print("\nOpening PME comparison chart...")
    fig_pme = viz.plot_pme_comparison(fund, list(index_returns), index_name="8% Benchmark")
    fig_pme.show()


if __name__ == "__main__":
    main()
