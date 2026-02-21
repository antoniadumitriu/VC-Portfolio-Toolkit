"""
jcurve.py — NAV projection with S-curve appreciation modeling.

Depends only on: metrics.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from vc_portfolio.metrics import calc_irr, calc_dpi, calc_tvpi


@dataclass
class CompanyTimeline:
    """Timeline for a single portfolio company investment."""

    entry_period: int
    exit_period: int
    initial_investment: float
    exit_multiple: float  # expected gross exit multiple (e.g. 3.0 = 3x)
    company_name: str = "Unnamed"
    follow_on_investments: dict[int, float] = field(default_factory=dict)
    # Mapping of {period: additional_investment_amount}

    @property
    def total_investment(self) -> float:
        return self.initial_investment + sum(self.follow_on_investments.values())

    @property
    def holding_period(self) -> int:
        return self.exit_period - self.entry_period

    @property
    def expected_exit_value(self) -> float:
        return self.total_investment * self.exit_multiple


class JCurve:
    """
    NAV projection engine using S-curve appreciation timing.

    The J-Curve captures the characteristic shape of VC fund NAV over time:
    1. Initial dip: management fees + slow early value creation
    2. Trough: approximately 2–3 years in
    3. Recovery: exits and mark-ups drive NAV above cost basis
    4. Peak: around years 5–8
    5. Wind-down: distributions reduce NAV to zero

    Value appreciation follows a logistic (S-curve) function to model realistic
    startup growth: slow start, rapid mid-period growth, plateau at exit.

    Supports method chaining:

        jcurve = (
            JCurve(fund_config)
            .add_company(timeline_a)
            .add_companies([timeline_b, timeline_c])
        )
    """

    def __init__(
        self,
        n_periods: int = 40,
        fee_rate: float = 0.02,
        committed_capital: float = 100_000_000,
        period_unit: str = "quarter",
    ) -> None:
        self.n_periods = n_periods
        self.fee_rate = fee_rate
        self.committed_capital = committed_capital
        self.period_unit = period_unit
        self._timelines: list[CompanyTimeline] = []

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add_company(self, timeline: CompanyTimeline) -> "JCurve":
        """Add a single company timeline."""
        self._timelines.append(timeline)
        return self

    def add_companies(self, timelines: list[CompanyTimeline]) -> "JCurve":
        """Add multiple company timelines."""
        self._timelines.extend(timelines)
        return self

    # ------------------------------------------------------------------
    # S-curve appreciation
    # ------------------------------------------------------------------

    def _compute_scurve_weight(
        self,
        t: int,
        entry: int,
        exit_period: int,
        steepness: float = 6.0,
    ) -> float:
        """
        Logistic S-curve weight for value appreciation at period t.

        Maps the progress of a company from entry to exit onto a logistic
        function. At entry: weight ≈ 0. At exit: weight ≈ 1.

        Parameters
        ----------
        t:
            Current period.
        entry:
            Company entry period.
        exit_period:
            Company exit period.
        steepness:
            Controls how sharp the S-curve is. Higher = more concentrated growth.

        Returns
        -------
        float
            Weight in [0, 1]. Returns 0 before entry, approaches 1 near exit.
        """
        if t < entry:
            return 0.0
        if t >= exit_period:
            return 1.0

        duration = exit_period - entry
        if duration <= 0:
            return 1.0

        # Normalize to [-steepness/2, steepness/2]
        progress = (t - entry) / duration  # 0 to 1
        x = steepness * (progress - 0.5)
        return float(1.0 / (1.0 + np.exp(-x)))

    def _compute_company_nav(
        self,
        timeline: CompanyTimeline,
        period: int,
    ) -> float:
        """
        Compute mark-to-market NAV for a single company at a given period.

        Before entry: 0
        During holding: S-curve appreciation from cost basis to expected exit value
        After exit: 0 (realized)

        Parameters
        ----------
        timeline:
            Company investment timeline.
        period:
            Period to compute NAV for.

        Returns
        -------
        float
            Estimated NAV contribution at this period.
        """
        if period < timeline.entry_period or period >= timeline.exit_period:
            return 0.0

        # Cost basis at this period (initial + follow-ons invested so far)
        cost_basis = timeline.initial_investment
        for fo_period, fo_amount in timeline.follow_on_investments.items():
            if fo_period <= period:
                cost_basis += fo_amount

        # Appreciation weight from S-curve
        weight = self._compute_scurve_weight(
            period, timeline.entry_period, timeline.exit_period
        )

        # Mark-to-market: interpolate between cost and expected exit value
        expected_value = timeline.total_investment * timeline.exit_multiple
        nav = cost_basis + weight * (expected_value - cost_basis)

        return max(0.0, nav)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self) -> pd.DataFrame:
        """
        Project NAV, cash flows, and return metrics across all periods.

        Returns
        -------
        DataFrame with columns:
            period, nav, unrealized_value, realized_value, cost_basis,
            mgmt_fees, capital_calls, distributions, net_cashflow,
            cumulative_called, cumulative_distributions,
            rolling_irr, tvpi, dpi
        """
        n = self.n_periods
        periods_per_year = 4 if self.period_unit == "quarter" else 1
        annual_fee = self.committed_capital * self.fee_rate

        nav_arr = np.zeros(n, dtype=np.float64)
        unrealized = np.zeros(n, dtype=np.float64)
        realized = np.zeros(n, dtype=np.float64)
        cost_basis_arr = np.zeros(n, dtype=np.float64)
        capital_calls = np.zeros(n, dtype=np.float64)
        distributions = np.zeros(n, dtype=np.float64)
        fees = np.full(n, annual_fee / periods_per_year, dtype=np.float64)

        # Process each company timeline
        for timeline in self._timelines:
            # Initial investment
            if 0 <= timeline.entry_period < n:
                capital_calls[timeline.entry_period] += timeline.initial_investment

            # Follow-on investments
            for fo_period, fo_amount in timeline.follow_on_investments.items():
                if 0 <= fo_period < n:
                    capital_calls[fo_period] += fo_amount

            # Exit distribution
            if 0 <= timeline.exit_period < n:
                distributions[timeline.exit_period] += timeline.expected_exit_value
            elif timeline.exit_period >= n:
                # Company not exited within fund life → add to terminal NAV
                pass

        # Per-period NAV computation
        for t in range(n):
            period_unrealized = 0.0
            period_realized = 0.0
            period_cost = 0.0

            for timeline in self._timelines:
                if t < timeline.entry_period:
                    continue
                elif t >= timeline.exit_period:
                    period_realized += timeline.expected_exit_value
                else:
                    company_nav = self._compute_company_nav(timeline, t)
                    period_unrealized += company_nav

                    # Cost basis (investments made so far)
                    cb = timeline.initial_investment
                    for fo_p, fo_a in timeline.follow_on_investments.items():
                        if fo_p <= t:
                            cb += fo_a
                    period_cost += cb

            unrealized[t] = period_unrealized
            realized[t] = period_realized
            cost_basis_arr[t] = period_cost
            nav_arr[t] = period_unrealized  # NAV = unrealized portion only

        # Cumulative metrics
        cum_called = np.cumsum(capital_calls)
        cum_dist = np.cumsum(distributions)
        net_cf = distributions - capital_calls - fees

        # Rolling IRR and return multiples
        rolling_irr = np.full(n, float("nan"), dtype=np.float64)
        tvpi_arr = np.zeros(n, dtype=np.float64)
        dpi_arr = np.zeros(n, dtype=np.float64)

        for t in range(1, n):
            if cum_called[t] > 0:
                tvpi_arr[t] = (nav_arr[t] + cum_dist[t]) / cum_called[t]
                dpi_arr[t] = cum_dist[t] / cum_called[t]

                # Rolling IRR: use cashflows up to period t + terminal NAV
                cf_slice = net_cf[: t + 1].copy()
                cf_slice[t] += nav_arr[t]  # add terminal NAV to last period
                irr = calc_irr(cf_slice)
                if not np.isnan(irr) and -1 < irr < 100:
                    rolling_irr[t] = irr

        return pd.DataFrame(
            {
                "period": np.arange(n),
                "nav": nav_arr,
                "unrealized_value": unrealized,
                "realized_value": realized,
                "cost_basis": cost_basis_arr,
                "capital_calls": capital_calls,
                "distributions": distributions,
                "mgmt_fees": fees,
                "net_cashflow": net_cf,
                "cumulative_called": cum_called,
                "cumulative_distributions": cum_dist,
                "rolling_irr": rolling_irr,
                "tvpi": tvpi_arr,
                "dpi": dpi_arr,
            }
        )

    # ------------------------------------------------------------------
    # Shape analysis
    # ------------------------------------------------------------------

    def get_jcurve_shape(self) -> dict[str, float | int]:
        """
        Identify key J-curve inflection points.

        Returns
        -------
        dict with:
            trough_period: period of minimum NAV (net of fees)
            trough_nav: NAV value at trough
            breakeven_period: first period where cumulative NAV ≥ cost basis
            peak_nav: maximum NAV achieved
            peak_period: period of maximum NAV
        """
        df = self.project()

        # Trough: minimum net cashflow position (before distributions dominate)
        cumulative_value = df["nav"] + df["cumulative_distributions"] - df["cumulative_called"]
        trough_idx = int(cumulative_value.idxmin())
        trough_nav = float(cumulative_value.iloc[trough_idx])

        # Peak NAV
        peak_idx = int(df["nav"].idxmax())
        peak_nav = float(df["nav"].iloc[peak_idx])

        # Breakeven: first period where TVPI ≥ 1
        breakeven_period = -1
        for t, tvpi in enumerate(df["tvpi"]):
            if tvpi >= 1.0 and df["cumulative_called"].iloc[t] > 0:
                breakeven_period = t
                break

        return {
            "trough_period": trough_idx,
            "trough_nav": trough_nav,
            "breakeven_period": breakeven_period,
            "peak_nav": peak_nav,
            "peak_period": peak_idx,
        }

    def __len__(self) -> int:
        return len(self._timelines)

    def __repr__(self) -> str:
        total_inv = sum(t.total_investment for t in self._timelines)
        return (
            f"JCurve(n_companies={len(self._timelines)}, "
            f"n_periods={self.n_periods}, "
            f"total_invested=${total_inv:,.0f})"
        )
