"""
portfolio.py: Power law return simulation for VC portfolios.

No dependencies on other library modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd



# Stage-level loss rates calibrated to Correlation Ventures / CB Insights data
STAGE_LOSS_RATES: dict[str, float] = {
    "pre_seed": 0.65,
    "seed": 0.50,
    "series_a": 0.35,
    "series_b": 0.25,
    "series_c": 0.20,
    "growth": 0.15,
}

Stage = Literal["pre_seed", "seed", "series_a", "series_b", "series_c", "growth"]


# Mixture weights for the non-loss share of outcomes
ZOMBIE_SHARE = 0.25      # of non-loss outcomes, capped at 20% overall
ZOMBIE_CAP = 0.20
LOGNORMAL_SHARE = 0.65   # of non-loss outcomes; the rest goes to the Pareto tail
MAX_LOSS_RATE = 0.95


def draw_exit_multiples(
    rng: np.random.Generator,
    stage: str,
    n: int,
    loss_rate_adjustment: float = 0.0,
    loss_rate_multiplier: float = 1.0,
) -> npt.NDArray[np.float64]:
    """
    Draw gross exit multiples from the 4-component mixture.

    Components (weights are shares of all outcomes):
        p_loss                        -> 0x (complete loss)
        min(0.20, 0.25 * (1-p_loss))  -> Uniform(0.5, 1.5) zombie returns
        0.65 * (1-p_loss)             -> LogNormal(mu=ln 2, sigma=0.85), median 2x
        remainder                     -> Pareto(alpha=1.5, xm=2x) home runs

    ``p_loss`` is the stage loss rate, scaled by ``loss_rate_multiplier`` and
    shifted by ``loss_rate_adjustment``, then clipped to [0, 0.95].
    """
    base = STAGE_LOSS_RATES.get(stage, 0.40)
    p_loss = float(np.clip(base * loss_rate_multiplier + loss_rate_adjustment, 0.0, MAX_LOSS_RATE))
    p_rest = 1.0 - p_loss
    p_zombie = min(ZOMBIE_CAP, p_rest * ZOMBIE_SHARE)
    p_lognorm = p_rest * LOGNORMAL_SHARE

    u = rng.uniform(size=n)
    zombie = rng.uniform(0.5, 1.5, size=n)
    lognorm = rng.lognormal(mean=np.log(2.0), sigma=0.85, size=n)
    pareto = 2.0 / (1.0 - rng.uniform(size=n)) ** (1.0 / 1.5)

    return np.select(
        [u < p_loss, u < p_loss + p_zombie, u < p_loss + p_zombie + p_lognorm],
        [0.0, zombie, lognorm],
        default=pareto,
    ).astype(np.float64)


def portfolio_irrs(
    investments: npt.NDArray[np.float64],
    entry_years: npt.NDArray[np.float64],
    exit_years: npt.NDArray[np.float64],
    exit_values: npt.NDArray[np.float64],
    lo: float = -0.999,
    hi: float = 1000.0,
    n_grid: int = 96,
    n_iter: int = 60,
) -> npt.NDArray[np.float64]:
    """
    Vectorized IRR for many simulated portfolios.

    Each company invests ``investments[j]`` at ``entry_years[j]`` and returns
    ``exit_values[i, j]`` at ``exit_years[j]`` (years, may be fractional).
    For each row, NPV is evaluated on a grid in log(1 + r); the sign change
    closest to r = 0 is then refined by bisection. With staggered deals the
    NPV curve can have several roots, and this picks the economically
    meaningful one, matching ``metrics.calc_irr``.

    Returns -1.0 for rows with no proceeds, and the bound (lo or hi) when
    NPV never changes sign inside it.
    """
    values = np.atleast_2d(np.asarray(exit_values, dtype=np.float64))
    origin = float(np.min(entry_years))
    t0 = np.asarray(entry_years, dtype=np.float64) - origin
    t1 = np.asarray(exit_years, dtype=np.float64) - origin
    inv = np.asarray(investments, dtype=np.float64)

    def npv(g: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """NPV for each row at log-growth g (same shape as rows, or grid)."""
        g = g[..., None]
        return (values[:, None, :] * np.exp(-g * t1)).sum(-1) - (inv * np.exp(-g * t0)).sum(-1)

    n = values.shape[0]
    grid = np.linspace(np.log1p(lo), np.log1p(hi), n_grid)
    f = npv(np.broadcast_to(grid, (n, n_grid)))  # (n, n_grid)

    crosses = np.sign(f[:, :-1]) * np.sign(f[:, 1:]) <= 0
    # Distance of each bracket from r = 0, inf where there is no sign change
    dist = np.where(crosses, np.abs(0.5 * (grid[:-1] + grid[1:])), np.inf)
    k = np.argmin(dist, axis=1)
    has_root = np.isfinite(dist[np.arange(n), k])

    a = grid[k].copy()
    b = grid[k + 1].copy()
    f_a = f[np.arange(n), k]
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        f_m = npv(m[:, None])[:, 0]
        left = np.sign(f_m) == np.sign(f_a)
        a = np.where(left, m, a)
        f_a = np.where(left, f_m, f_a)
        b = np.where(left, b, m)

    irr = np.expm1(0.5 * (a + b))
    # No root in range: NPV positive everywhere means returns beyond hi
    irr = np.where(has_root, irr, np.where(f[:, -1] > 0, hi, lo))
    irr[values.sum(axis=1) <= 0] = -1.0
    return irr


def simulate_portfolio(
    companies: list[Company],
    n_simulations: int,
    rng: np.random.Generator,
    return_multiplier: float = 1.0,
    loss_rate_adjustment: float = 0.0,
    loss_rate_multiplier: float = 1.0,
    holding_period_multiplier: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Simulate portfolio MOIC and IRR across ``n_simulations`` draws.

    Periods on ``Company`` are in years. Returns (moics, irrs).
    """
    if not companies:
        raise ValueError("Portfolio is empty. Add companies before simulating.")

    investments = np.array([c.investment_amount for c in companies], dtype=np.float64)
    total = float(investments.sum())
    entry = np.array([c.entry_period for c in companies], dtype=np.float64)
    hold = np.array([c.holding_period for c in companies], dtype=np.float64)
    exit_ = entry + np.maximum(hold * holding_period_multiplier, 1e-6)

    multiples = np.column_stack(
        [
            draw_exit_multiples(
                rng, c.stage, n_simulations, loss_rate_adjustment, loss_rate_multiplier
            )
            for c in companies
        ]
    ) * return_multiplier
    values = multiples * investments

    moics = values.sum(axis=1) / total if total > 0 else np.zeros(n_simulations)
    irrs = portfolio_irrs(investments, entry, exit_, values)
    return moics, irrs


def summarize_simulation(
    moics: npt.NDArray[np.float64],
    irrs: npt.NDArray[np.float64],
) -> dict[str, object]:
    """Percentiles and probabilities for simulated MOICs and IRRs."""
    out: dict[str, object] = {}
    for p in (10, 25, 50, 75, 90):
        out[f"moic_p{p}"] = float(np.percentile(moics, p))
    for p in (10, 25, 50, 75, 90):
        out[f"irr_p{p}"] = float(np.percentile(irrs, p))
    out["loss_probability"] = float(np.mean(moics < 1.0))
    out["home_run_probability"] = float(np.mean(moics > 3.0))
    out["raw_moics"] = moics
    out["raw_irrs"] = irrs
    return out


@dataclass
class Company:
    """Represents a single portfolio company investment."""

    name: str
    investment_amount: float
    stage: Stage
    sector: str
    entry_valuation: float
    entry_period: float  # years from fund start
    expected_exit_period: float  # years from fund start
    ownership_pct: float  # post-money ownership percentage (e.g. 0.15 = 15%)
    reserved_capital: float = 0.0  # reserved for follow-on

    @property
    def total_committed(self) -> float:
        return self.investment_amount + self.reserved_capital

    @property
    def holding_period(self) -> float:
        return self.expected_exit_period - self.entry_period


class Portfolio:
    """
    VC portfolio modeler with power law exit simulation.

    Uses the modern NumPy Generator API (default_rng) for reproducibility.
    Exit multiples follow a 4-component mixture:
        - p_loss (stage loss rate) at 0x
        - zombie returns: Uniform(0.5, 1.5)
        - 65% of non-loss outcomes: LogNormal with median 2x
        - remainder: Pareto(alpha=1.5, xm=2x) home runs
    See ``draw_exit_multiples`` for exact weights.

    Supports method chaining:

        portfolio = (
            Portfolio(seed=42)
            .add_company(company_a)
            .add_companies([company_b, company_c])
        )
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._companies: list[Company] = []

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add_company(self, company: Company) -> "Portfolio":
        """Add a single company to the portfolio."""
        self._companies.append(company)
        return self

    def add_companies(self, companies: list[Company]) -> "Portfolio":
        """Add multiple companies to the portfolio."""
        self._companies.extend(companies)
        return self

    # ------------------------------------------------------------------
    # Exit multiple simulation
    # ------------------------------------------------------------------

    def _draw_exit_multiple(
        self,
        stage: Stage,
        n: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Draw exit multiples for one stage. See ``draw_exit_multiples``."""
        return draw_exit_multiples(self._rng, stage, n)

    # ------------------------------------------------------------------
    # Portfolio simulation
    # ------------------------------------------------------------------

    def simulate_exits(
        self,
        n_simulations: int = 10_000,
    ) -> dict[str, object]:
        """
        Run Monte Carlo exit simulations across all portfolio companies.

        Each simulation draws an exit multiple for every company, then
        computes portfolio MOIC and an IRR from the actual cash flow timing
        (investment at entry, proceeds at exit, in years).

        Returns
        -------
        dict with keys:
            moic_p10 ... moic_p90, irr_p10 ... irr_p90
            loss_probability (fraction of simulations with MOIC < 1)
            home_run_probability (fraction with MOIC > 3)
            raw_moics, raw_irrs (ndarrays of all simulated values)
        """
        moics, irrs = simulate_portfolio(self._companies, n_simulations, self._rng)
        return summarize_simulation(moics, irrs)

    # ------------------------------------------------------------------
    # Portfolio analytics
    # ------------------------------------------------------------------

    def reserve_ratio(self) -> dict[str, float]:
        """Return reserve ratio stats across portfolio."""
        if not self._companies:
            return {}

        total_initial = sum(c.investment_amount for c in self._companies)
        total_reserved = sum(c.reserved_capital for c in self._companies)
        total_committed = sum(c.total_committed for c in self._companies)

        return {
            "total_initial_investment": total_initial,
            "total_reserved": total_reserved,
            "total_committed": total_committed,
            "reserve_ratio": total_reserved / total_initial if total_initial > 0 else 0.0,
        }

    def follow_on_strategy(
        self,
        strategy: Literal["pro_rata", "top_up", "none"] = "pro_rata",
    ) -> pd.DataFrame:
        """
        Generate follow-on investment schedule by strategy.

        Parameters
        ----------
        strategy:
            'pro_rata': maintain ownership through each round
            'top_up': invest reserved capital in top performers only
            'none': no follow-on

        Returns
        -------
        DataFrame with columns: company, initial, follow_on, total, period
        """
        rows = []
        for company in self._companies:
            follow_on = 0.0
            if strategy == "pro_rata":
                follow_on = company.reserved_capital
            elif strategy == "top_up":
                # Invest in companies at Series B+ stages
                if company.stage in ("series_b", "series_c", "growth"):
                    follow_on = company.reserved_capital
            # else "none": no follow-on

            rows.append(
                {
                    "company": company.name,
                    "stage": company.stage,
                    "initial_investment": company.investment_amount,
                    "follow_on": follow_on,
                    "total_committed": company.investment_amount + follow_on,
                    "entry_period": company.entry_period,
                }
            )

        return pd.DataFrame(rows)

    def portfolio_breakdown(self) -> pd.DataFrame:
        """Return company-level breakdown DataFrame."""
        if not self._companies:
            return pd.DataFrame()

        rows = []
        total_invested = sum(c.investment_amount for c in self._companies)

        for company in self._companies:
            rows.append(
                {
                    "company": company.name,
                    "stage": company.stage,
                    "sector": company.sector,
                    "investment_amount": company.investment_amount,
                    "ownership_pct": company.ownership_pct,
                    "entry_valuation": company.entry_valuation,
                    "entry_period": company.entry_period,
                    "expected_exit_period": company.expected_exit_period,
                    "holding_period": company.holding_period,
                    "portfolio_weight": (
                        company.investment_amount / total_invested
                        if total_invested > 0
                        else 0.0
                    ),
                }
            )

        return pd.DataFrame(rows)

    def concentration_analysis(self) -> dict[str, object]:
        """
        Compute portfolio concentration metrics.

        Returns
        -------
        dict with:
            herfindahl_index: HHI of portfolio weights (0=diversified, 1=concentrated)
            top_3_concentration: % of portfolio in top 3 positions
            stage_diversity: number of distinct stages represented
            sector_diversity: number of distinct sectors represented
            stage_counts: dict of stage → count
            sector_counts: dict of sector → count
        """
        if not self._companies:
            return {}

        investments = np.array([c.investment_amount for c in self._companies])
        total = investments.sum()
        weights = investments / total if total > 0 else np.zeros_like(investments)

        # Herfindahl-Hirschman Index
        hhi = float(np.sum(weights**2))

        # Top 3 concentration
        sorted_weights = np.sort(weights)[::-1]
        top3 = float(sorted_weights[:3].sum())

        # Stage and sector diversity
        stages = [c.stage for c in self._companies]
        sectors = [c.sector for c in self._companies]

        from collections import Counter
        stage_counts = dict(Counter(stages))
        sector_counts = dict(Counter(sectors))

        return {
            "herfindahl_index": hhi,
            "top_3_concentration": top3,
            "stage_diversity": len(set(stages)),
            "sector_diversity": len(set(sectors)),
            "stage_counts": stage_counts,
            "sector_counts": sector_counts,
        }

    def __len__(self) -> int:
        return len(self._companies)

    def __repr__(self) -> str:
        total = sum(c.investment_amount for c in self._companies)
        return (
            f"Portfolio(n_companies={len(self._companies)}, "
            f"total_invested=${total:,.0f})"
        )
