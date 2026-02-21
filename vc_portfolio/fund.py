"""
fund.py — Stateful cash flow ledger for a VC fund.

Depends only on: metrics.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from vc_portfolio.metrics import (
    calc_carry,
    calc_dpi,
    calc_irr,
    calc_management_fees,
    calc_moic,
    calc_rvpi,
    calc_tvpi,
)


@dataclass
class FundConfig:
    """Configuration for a VC fund."""

    name: str
    committed_capital: float
    vintage_year: int
    fee_rate: float = 0.02
    carry_rate: float = 0.20
    hurdle_rate: float = 0.08
    investment_period: int = 5  # years
    fund_life: int = 10  # years
    step_down_rate: float = 0.0025  # annual step-down in fee rate post-investment period
    period_unit: Literal["quarter", "year"] = "quarter"

    @property
    def n_periods(self) -> int:
        ppyr = 4 if self.period_unit == "quarter" else 1
        return self.fund_life * ppyr

    @property
    def periods_per_year(self) -> int:
        return 4 if self.period_unit == "quarter" else 1


class Fund:
    """
    Stateful cash flow ledger for a VC fund.

    Records capital calls, distributions, NAV marks, and computes
    LP-level return metrics net of fees and carry.

    Supports method chaining for a builder pattern:

        fund = (
            Fund(config)
            .deploy_capital({0: 5e6, 4: 3e6})
            .add_distribution(2e6, period=12)
            .set_nav(20, 8e6)
        )
    """

    def __init__(self, config: FundConfig) -> None:
        self.config = config
        n = config.n_periods
        self._capital_called: np.ndarray = np.zeros(n, dtype=np.float64)
        self._distributions: np.ndarray = np.zeros(n, dtype=np.float64)
        self._nav: np.ndarray = np.zeros(n, dtype=np.float64)
        self._fees: np.ndarray = calc_management_fees(
            committed=config.committed_capital,
            fee_rate=config.fee_rate,
            investment_period=config.investment_period,
            fund_life=config.fund_life,
            step_down_rate=config.step_down_rate,
            period_unit=config.period_unit,
        )

    # ------------------------------------------------------------------
    # Builder methods (method chaining)
    # ------------------------------------------------------------------

    def deploy_capital(self, schedule: dict[int, float]) -> "Fund":
        """
        Record capital calls according to a deployment schedule.

        Parameters
        ----------
        schedule:
            Mapping of {period_index: amount_called}. Amounts are positive.

        Returns
        -------
        self (for chaining)
        """
        for period, amount in schedule.items():
            if period < 0 or period >= self.config.n_periods:
                raise ValueError(
                    f"Period {period} out of range [0, {self.config.n_periods - 1}]"
                )
            if amount < 0:
                raise ValueError("Capital call amount must be non-negative")
            self._capital_called[period] += amount
        return self

    def add_distribution(
        self,
        amount: float,
        period: int,
        dist_type: Literal["return_of_capital", "profit", "dividend"] = "profit",
    ) -> "Fund":
        """
        Record a distribution to LPs.

        Parameters
        ----------
        amount:
            Distribution amount (positive).
        period:
            Period index of the distribution.
        dist_type:
            Category of distribution (informational only).

        Returns
        -------
        self (for chaining)
        """
        if period < 0 or period >= self.config.n_periods:
            raise ValueError(f"Period {period} out of range")
        if amount < 0:
            raise ValueError("Distribution amount must be non-negative")
        self._distributions[period] += amount
        return self

    def set_nav(self, period: int, nav: float) -> "Fund":
        """
        Set the NAV mark at a given period.

        Parameters
        ----------
        period:
            Period index.
        nav:
            Net Asset Value at that period.

        Returns
        -------
        self (for chaining)
        """
        if period < 0 or period >= self.config.n_periods:
            raise ValueError(f"Period {period} out of range")
        if nav < 0:
            raise ValueError("NAV cannot be negative")
        self._nav[period] = nav
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def total_invested(self) -> float:
        return float(np.sum(self._capital_called))

    @property
    def total_distributions(self) -> float:
        return float(np.sum(self._distributions))

    @property
    def current_nav(self) -> float:
        """Most recent non-zero NAV, or 0 if none set."""
        nonzero = np.nonzero(self._nav)[0]
        if len(nonzero) == 0:
            return 0.0
        return float(self._nav[nonzero[-1]])

    def irr(self, net_of_carry: bool = True) -> float:
        """
        Compute LP IRR.

        Parameters
        ----------
        net_of_carry:
            If True, deducts carry from distributions before computing IRR.
        """
        lp_cfs = self.get_lp_cashflows(net_of_carry=net_of_carry)
        return calc_irr(np.array(lp_cfs))

    def tvpi(self) -> float:
        return calc_tvpi(
            invested=self.total_invested,
            nav=self.current_nav,
            distributions=self.total_distributions,
        )

    def dpi(self) -> float:
        return calc_dpi(
            invested=self.total_invested,
            distributions=self.total_distributions,
        )

    def rvpi(self) -> float:
        return calc_rvpi(
            invested=self.total_invested,
            nav=self.current_nav,
        )

    def moic(self) -> float:
        return calc_moic(
            invested=self.total_invested,
            total_value=self.total_distributions + self.current_nav,
        )

    def _compute_carry(self) -> float:
        return calc_carry(
            distributions=self._distributions,
            invested=self.total_invested,
            carry_rate=self.config.carry_rate,
            hurdle_rate=self.config.hurdle_rate,
        )

    # ------------------------------------------------------------------
    # Cash flow accessors
    # ------------------------------------------------------------------

    def get_cashflows(self) -> pd.DataFrame:
        """
        Return a period-by-period cash flow DataFrame.

        Columns:
            period, capital_called, distributions, fees, nav,
            cumulative_called, cumulative_distributions, net_cashflow
        """
        n = self.config.n_periods
        periods = np.arange(n)

        cum_called = np.cumsum(self._capital_called)
        cum_dist = np.cumsum(self._distributions)
        net_cf = self._distributions - self._capital_called - self._fees

        return pd.DataFrame(
            {
                "period": periods,
                "capital_called": self._capital_called,
                "distributions": self._distributions,
                "fees": self._fees,
                "nav": self._nav,
                "cumulative_called": cum_called,
                "cumulative_distributions": cum_dist,
                "net_cashflow": net_cf,
            }
        )

    def get_lp_cashflows(self, net_of_carry: bool = True) -> list[float]:
        """
        Return LP-perspective cash flows for IRR computation.

        Outflows (capital calls + fees) are negative.
        Inflows (distributions) are positive.
        Final period includes remaining NAV.
        """
        # LP sees: -(calls + fees) + distributions
        lp_cfs = self._distributions - self._capital_called - self._fees

        if net_of_carry:
            carry = self._compute_carry()
            # Deduct carry from last distribution period
            dist_periods = np.nonzero(self._distributions)[0]
            if len(dist_periods) > 0:
                last_dist = dist_periods[-1]
                lp_cfs = lp_cfs.copy()
                lp_cfs[last_dist] -= carry

        # Add terminal NAV to last period
        nav_periods = np.nonzero(self._nav)[0]
        if len(nav_periods) > 0:
            last_nav_period = nav_periods[-1]
            lp_cfs = lp_cfs.copy()
            lp_cfs[last_nav_period] += self._nav[last_nav_period]

        return lp_cfs.tolist()

    def summary(self) -> dict[str, float | str | int]:
        """Return a summary dict of key fund metrics."""
        carry = self._compute_carry()
        return {
            "name": self.config.name,
            "vintage_year": self.config.vintage_year,
            "committed_capital": self.config.committed_capital,
            "total_invested": self.total_invested,
            "total_distributions": self.total_distributions,
            "current_nav": self.current_nav,
            "total_fees": float(np.sum(self._fees)),
            "carry": carry,
            "irr_gross": self.irr(net_of_carry=False),
            "irr_net": self.irr(net_of_carry=True),
            "tvpi": self.tvpi(),
            "dpi": self.dpi(),
            "rvpi": self.rvpi(),
            "moic": self.moic(),
        }

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"Fund(name={cfg.name!r}, vintage={cfg.vintage_year}, "
            f"committed=${cfg.committed_capital:,.0f}, "
            f"tvpi={self.tvpi():.2f}x, irr={self.irr():.1%})"
        )
