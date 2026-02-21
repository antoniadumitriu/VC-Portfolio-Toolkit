"""
portfolio.py — Power law return simulation for VC portfolios.

Depends only on: metrics.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from vc_portfolio.metrics import calc_irr, calc_moic


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


@dataclass
class Company:
    """Represents a single portfolio company investment."""

    name: str
    investment_amount: float
    stage: Stage
    sector: str
    entry_valuation: float
    entry_period: int  # period index (years or quarters from fund start)
    expected_exit_period: int
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
        - p_loss × 0 (complete loss)
        - 0.20 × Uniform(0.5, 1.5) (zombie / slow return)
        - 0.60 × LogNormal (calibrated to median 2x)
        - 0.20 × Pareto(α=1.5) (power law tail / home runs)

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
        """
        Draw exit multiples from a 4-component mixture distribution.

        Component weights:
            p_loss          → 0x (complete loss)
            0.20            → Uniform(0.5, 1.5) zombie returns
            0.60 × (1 - p_loss - 0.20) → LogNormal(μ=0.69, σ=0.85) ≈ median 2x
            remainder       → Pareto(α=1.5) home runs

        The LogNormal and Pareto weights rescale based on the stage loss rate.
        """
        p_loss = STAGE_LOSS_RATES.get(stage, 0.40)
        p_remaining = 1.0 - p_loss
        p_zombie = min(0.20, p_remaining * 0.25)
        p_lognormal = p_remaining * 0.65
        p_pareto = p_remaining - p_zombie - p_lognormal

        # Ensure non-negative weights
        p_pareto = max(0.0, p_pareto)

        # Draw component assignments
        u = self._rng.uniform(size=n)
        multiples = np.zeros(n, dtype=np.float64)

        loss_mask = u < p_loss
        zombie_mask = (u >= p_loss) & (u < p_loss + p_zombie)
        lognorm_mask = (u >= p_loss + p_zombie) & (u < p_loss + p_zombie + p_lognormal)
        pareto_mask = ~(loss_mask | zombie_mask | lognorm_mask)

        # Complete losses → 0x
        multiples[loss_mask] = 0.0

        # Zombie returns: uniform 0.5x–1.5x
        n_zombie = int(zombie_mask.sum())
        if n_zombie > 0:
            multiples[zombie_mask] = self._rng.uniform(0.5, 1.5, size=n_zombie)

        # LogNormal: calibrated so median ≈ 2x, mean ≈ 3–4x
        n_lognorm = int(lognorm_mask.sum())
        if n_lognorm > 0:
            multiples[lognorm_mask] = self._rng.lognormal(
                mean=0.693, sigma=0.85, size=n_lognorm
            )

        # Pareto: power law tail (α=1.5), min=2x
        n_pareto = int(pareto_mask.sum())
        if n_pareto > 0:
            # Pareto with xm=2, alpha=1.5: F(x) = 1 - (2/x)^1.5
            # Sample via inverse CDF: x = xm / (1-u)^(1/alpha)
            pareto_u = self._rng.uniform(size=n_pareto)
            multiples[pareto_mask] = 2.0 / (1 - pareto_u) ** (1 / 1.5)

        return multiples

    # ------------------------------------------------------------------
    # Portfolio simulation
    # ------------------------------------------------------------------

    def simulate_exits(
        self,
        n_simulations: int = 10_000,
    ) -> dict[str, object]:
        """
        Run Monte Carlo exit simulations across all portfolio companies.

        For each simulation:
        1. Draw exit multiples for each company.
        2. Compute portfolio-level MOIC and IRR.

        Parameters
        ----------
        n_simulations:
            Number of simulation runs.

        Returns
        -------
        dict with keys:
            moic_p10, moic_p25, moic_p50, moic_p75, moic_p90
            irr_p10, irr_p25, irr_p50, irr_p75, irr_p90
            loss_probability (fraction of simulations with MOIC < 1)
            home_run_probability (fraction with MOIC > 3)
            raw_moics (ndarray of all simulated MOICs)
            raw_irrs (ndarray of all simulated IRRs)
        """
        if not self._companies:
            raise ValueError("Portfolio is empty. Add companies before simulating.")

        n_companies = len(self._companies)
        investments = np.array([c.investment_amount for c in self._companies])
        total_invested = float(investments.sum())

        # Pre-compute holding periods (in years, assuming period=year)
        holding_periods = np.array(
            [c.holding_period for c in self._companies], dtype=np.float64
        )

        moics = np.zeros(n_simulations, dtype=np.float64)
        irrs = np.zeros(n_simulations, dtype=np.float64)

        for i in range(n_simulations):
            total_value = 0.0
            weighted_period = 0.0

            for j, company in enumerate(self._companies):
                multiples = self._draw_exit_multiple(company.stage, n=1)
                exit_value = investments[j] * float(multiples[0])
                total_value += exit_value
                weighted_period += investments[j] * holding_periods[j]

            moics[i] = total_value / total_invested if total_invested > 0 else 0.0

            # Approximate IRR from MOIC and weighted-average holding period
            avg_hold = weighted_period / total_invested if total_invested > 0 else 5.0
            if avg_hold > 0 and moics[i] > 0:
                irrs[i] = moics[i] ** (1.0 / avg_hold) - 1
            else:
                irrs[i] = -1.0

        percentiles = [10, 25, 50, 75, 90]

        return {
            "moic_p10": float(np.percentile(moics, 10)),
            "moic_p25": float(np.percentile(moics, 25)),
            "moic_p50": float(np.percentile(moics, 50)),
            "moic_p75": float(np.percentile(moics, 75)),
            "moic_p90": float(np.percentile(moics, 90)),
            "irr_p10": float(np.percentile(irrs, 10)),
            "irr_p25": float(np.percentile(irrs, 25)),
            "irr_p50": float(np.percentile(irrs, 50)),
            "irr_p75": float(np.percentile(irrs, 75)),
            "irr_p90": float(np.percentile(irrs, 90)),
            "loss_probability": float(np.mean(moics < 1.0)),
            "home_run_probability": float(np.mean(moics > 3.0)),
            "raw_moics": moics,
            "raw_irrs": irrs,
        }

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
