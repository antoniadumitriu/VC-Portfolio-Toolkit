# vc-portfolio-toolkit

A polished, installable Python library for VC fund portfolio analysis. Models fund returns, J-curves, and portfolio construction scenarios using Monte Carlo simulation.

**Portfolio showcase project demonstrating:** layered architecture · type safety · numerical methods · vectorized simulation · interactive visualization

---

## Features

- **J-Curve Modeling** — NAV projection with logistic S-curve appreciation timing
- **Power Law Exit Simulation** — 4-component mixture (Pareto α=1.5, calibrated to Correlation Ventures data)
- **Fund Cash Flow Ledger** — Management fees, carry computation, LP IRR with Newton-Raphson + Brent fallback
- **Monte Carlo Orchestration** — Vectorized NumPy simulation with common random numbers across scenarios
- **Fan Charts** — p10/p25/p50/p75/p90 percentile bands (Cambridge Associates standard)
- **Sensitivity Analysis** — One-way sweeps and tornado charts over exit multiples, loss rates and holding periods
- **PME Benchmarking** — KS-PME, PME Alpha, Direct Alpha vs public market indices
- **Interactive Visualization** — Professional dark-themed Plotly charts

---

## Interactive app

Analyze any portfolio in the browser: start from a sample, edit the table directly, or upload a CSV or Excel file with one row per company.

```bash
pip install -e ".[app]"
streamlit run streamlit_app.py
```

What it shows:
- **Return distribution**: simulated MOIC and IRR, percentiles, chance of losing money or returning 3x+, 95% VaR and CVaR
- **Scenarios**: Bear / Base / Bull with adjustable exit multiples, loss rates and exit timing
- **Sensitivity**: tornado chart of which assumption moves median IRR, MOIC or loss probability most
- **J-curve**: quarterly fund value with fees, from each company's expected multiple (or the stage average)
- **Portfolio**: breakdown by stage or sector, concentration

Every table and result can be downloaded as CSV. Uploaded data lives only in the browser session and is never stored.

**Input format** (see `examples/sample_portfolio.csv`, or download the template in the app):

| Column | Required | Notes |
|---|---|---|
| `name` | yes | |
| `investment_amount` | yes | `2000000`, `2,000,000` or `$2M` |
| `stage` | yes | `pre_seed`, `seed`, `series_a`, `series_b`, `series_c`, `growth` ("Series A" also works) |
| `entry_year`, `exit_year` | yes | years from fund start, fractions allowed |
| `sector` | no | |
| `entry_valuation`, `reserved_capital` | no | |
| `ownership_pct` | no | `0.15`, `15` or `15%` |
| `expected_multiple` | no | J-curve only; blank uses the stage average |

Common header variants (`Company`, `Amount`, `Ownership`, `Exit Year`, ...) are recognized. Problems are reported row by row.

The same loader works in Python:

```python
from vc_portfolio import Portfolio
portfolio = Portfolio.from_csv("my_portfolio.csv", seed=42)
```

### Founder Round Modeler

A second page in the app, `pages/founder_round_modeler.py`, looks at one company from the founder's side. Enter your cap table, any SAFEs and notes, and two term sheets to see:

- **The round**: price per share, post-money, founder ownership, and the effective pre-money after the option pool and converting securities, plus the pre-money needed to hold a target stake
- **Who gets paid**: founder proceeds at every exit value through the full liquidation waterfall
- **Exit odds**: the chance of reaching a payout, with both offers run on the same simulated exits
- **What to negotiate**: what each term is worth to founders, next to a 10% higher valuation
- **Investor math**: the exit needed for 1x, 3x, 10x and to return the fund
- **Down round**: founder ownership under each type of antidilution protection

The same logic is available in Python:

```python
from vc_portfolio import CapTable, Convertible, RoundTerms, price_round, waterfall

company = CapTable(founders=8_000_000, available_options=1_000_000)
safe = Convertible("Seed SAFE", 500_000, cap=6_000_000)
terms = RoundTerms(3_000_000, 12_000_000, option_pool_target=0.10, convertibles=(safe,))

result = price_round(company, terms)
print(result.price, result.founder_ownership, result.effective_pre_money)
print(waterfall(result.after, [25_000_000, 100_000_000]).founders)
```

Round mechanics and waterfalls follow Dave Lishego, *The Founder's Guide to VC Finance* (2017), and post-money SAFE conversion follows Y Combinator's SAFE documentation. Taxes, debt, deal fees and option strike proceeds are ignored. This is for education and planning, not legal or investment advice.

### Deploying

The repo is ready for [Streamlit Community Cloud](https://streamlit.io/cloud) (free): create an app from this repository with `streamlit_app.py` as the main file. `requirements.txt` installs the library and the app's dependencies, and `.streamlit/config.toml` sets the theme.

---

## Installation

```bash
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, numpy ≥ 1.24, scipy ≥ 1.10, pandas ≥ 2.0, plotly ≥ 5.15

---

## Quick Start

### Fund Modeling

```python
from vc_portfolio import Fund, FundConfig

config = FundConfig(
    name="Acme Fund I",
    committed_capital=100_000_000,
    vintage_year=2020,
    fee_rate=0.02,
    carry_rate=0.20,
    hurdle_rate=0.08,
    investment_period=5,
    fund_life=10,
)

fund = (
    Fund(config)
    .deploy_capital({0: 20e6, 1: 20e6, 2: 15e6, 3: 10e6})
    .add_distribution(15e6, period=5)
    .add_distribution(60e6, period=8)
    .set_nav(6, 80e6)
    .set_nav(9, 20e6)
)

summary = fund.summary()
print(f"Net IRR: {summary['irr_net']:.1%}")
print(f"TVPI:    {summary['tvpi']:.2f}x")
print(f"DPI:     {summary['dpi']:.2f}x")
```

### Portfolio Monte Carlo

```python
from vc_portfolio import Portfolio, Company

portfolio = (
    Portfolio(seed=42)
    .add_company(Company("TechCo",  3e6, "seed",     "saas",    12e6, 0, 7, 0.15))
    .add_company(Company("HealthAI", 5e6, "series_a", "health",  25e6, 1, 8, 0.12))
    .add_company(Company("FinTech",  2e6, "pre_seed", "fintech", 6e6,  0, 9, 0.20))
)

results = portfolio.simulate_exits(n_simulations=10_000)
print(f"Median MOIC:     {results['moic_p50']:.2f}x")
print(f"Loss Prob:       {results['loss_probability']:.1%}")
print(f"Home Run Prob:   {results['home_run_probability']:.1%}")
```

### Scenario Analysis

```python
from vc_portfolio import ScenarioSet, BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO
from vc_portfolio import visualization as viz

results = ScenarioSet([BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO]).run(
    portfolio=portfolio,
    n_simulations=10_000,
)

print(results.compare())

fig = viz.plot_scenario_fan(results, metric="moic")
fig.show()
```

### J-Curve Analysis

```python
from vc_portfolio import JCurve, CompanyTimeline
from vc_portfolio import visualization as viz

jcurve = (
    JCurve(n_periods=40, committed_capital=100e6)
    .add_company(CompanyTimeline(0,  16, 10e6, 4.0, "StarCo"))
    .add_company(CompanyTimeline(4,  20, 8e6,  3.0, "MidCo"))
    .add_company(CompanyTimeline(8,  24, 6e6,  2.0, "LateCo"))
)

df = jcurve.project()
shape = jcurve.get_jcurve_shape()
print(f"Trough at period {shape['trough_period']}")
print(f"Breakeven at period {shape['breakeven_period']}")

fig = viz.plot_jcurve(df)
fig.show()
```

---

## Architecture

```
metrics.py  ←  fund.py, jcurve.py, portfolio.py  ←  scenarios.py  ←  visualization.py
```

Unidirectional dependencies — `metrics.py` is pure math with no library imports. Each layer depends only on layers below it.

### Module Overview

| Module | Role | Key Classes |
|--------|------|-------------|
| `metrics.py` | Pure math, no side effects | Functions only |
| `fund.py` | Cash flow ledger | `Fund`, `FundConfig` |
| `portfolio.py` | Power law simulation | `Portfolio`, `Company` |
| `jcurve.py` | NAV projection | `JCurve`, `CompanyTimeline` |
| `scenarios.py` | Monte Carlo orchestration | `ScenarioSet`, `MonteCarloSimulator`, `SensitivityAnalysis` |
| `visualization.py` | Plotly figure factories | 8 chart functions |
| `streamlit_app.py` | Interactive web app | |

---

## Design Decisions

### Power Law Exit Distribution
Exit multiples use a 4-component mixture calibrated to empirical VC data. Weights are shares of all outcomes:
- **p_loss → 0x**: complete losses (stage loss rate, 15% at growth to 65% at pre-seed, adjustable per scenario)
- **min(20%, 25% × (1 − p_loss)) → Uniform(0.5, 1.5)**: zombie returns
- **65% × (1 − p_loss) → LogNormal(μ = ln 2, σ = 0.85)**: typical successes, median 2x
- **remainder → Pareto(α = 1.5, xm = 2x)**: power law tail (home runs)

### IRR Computation
`calc_irr` uses Newton-Raphson with an analytical derivative, with Brent's method as fallback for edge cases (flat curves, multiple roots). Simulated portfolios use the vectorized solver described below.

### Simulation and IRR
Exit multiples for all simulations are drawn at once as NumPy arrays. Portfolio IRR is solved from each simulation's actual cash flows (investment at entry, proceeds at exit, in years) with a vectorized grid search plus bisection, picking the root closest to 0% when staggered deals produce more than one.

Scenarios and sensitivity runs reuse one seed (common random numbers), so differences between them come from the assumptions, not from sampling noise.

### Units
`Company` entry and exit periods are in years. `Scenario.exit_timing_adjustment` is in quarters and is converted to years. `JCurve` runs on a quarterly grid by default and reports an annualized rolling IRR. Net TVPI and DPI in the J-curve use paid-in capital including management fees, which is what produces the early dip.

### Type Safety
`from __future__ import annotations`, `npt.NDArray[np.float64]`, `Literal` types throughout. mypy strict mode compatible.

---

## Testing

```bash
pytest --cov=vc_portfolio --cov-report=term-missing
```

---

## Examples

```bash
python examples/basic_fund_model.py
python examples/portfolio_construction.py
python examples/jcurve_analysis.py
```
# VC-Portfolio-Toolkit
