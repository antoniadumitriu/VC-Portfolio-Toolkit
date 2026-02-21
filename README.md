# vc-portfolio-toolkit

A polished, installable Python library for VC fund portfolio analysis. Models fund returns, J-curves, and portfolio construction scenarios using Monte Carlo simulation.

**Portfolio showcase project demonstrating:** layered architecture · type safety · numerical methods · parallelism · interactive visualization

---

## Features

- **J-Curve Modeling** — NAV projection with logistic S-curve appreciation timing
- **Power Law Exit Simulation** — 4-component mixture (Pareto α=1.5, calibrated to Correlation Ventures data)
- **Fund Cash Flow Ledger** — Management fees, carry computation, LP IRR with Newton-Raphson + Brent fallback
- **Monte Carlo Orchestration** — ProcessPoolExecutor parallelism for scenario analysis
- **Fan Charts** — p10/p25/p50/p75/p90 percentile bands (Cambridge Associates standard)
- **Sensitivity Analysis** — One-way sweeps and tornado charts
- **PME Benchmarking** — KS-PME, PME Alpha, Direct Alpha vs public market indices
- **Interactive Visualization** — Professional dark-themed Plotly charts

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

---

## Design Decisions

### Power Law Exit Distribution
Exit multiples use a 4-component mixture calibrated to empirical VC data:
- **p_loss × 0x** — Complete losses (stage-dependent: 15%–65%)
- **20% × Uniform(0.5, 1.5)** — Zombie returns (marginal outcomes)
- **60% × LogNormal(μ=0.69, σ=0.85)** — Successful outcomes (median ≈ 2x)
- **Pareto(α=1.5, xm=2x)** — Power law tail (home runs)

### IRR Computation
Newton-Raphson with analytical Jacobian, with Brent's method as fallback for edge cases (flat curves, multiple roots).

### Parallelism
Monte Carlo uses `ProcessPoolExecutor` (not `ThreadPoolExecutor`) to bypass the GIL for CPU-bound numerical simulation.

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
