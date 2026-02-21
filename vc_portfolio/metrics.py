"""
metrics.py — Pure mathematical functions for VC fund analysis.

No imports from within this library. All functions are stateless and
have no side effects. Safe to import from any module.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import optimize


# ---------------------------------------------------------------------------
# IRR
# ---------------------------------------------------------------------------

def calc_irr(
    cashflows: npt.NDArray[np.float64],
    periods: Optional[npt.NDArray[np.float64]] = None,
    guess: float = 0.10,
    tol: float = 1e-8,
) -> float:
    """
    Compute Internal Rate of Return using Newton-Raphson with Brent fallback.

    Parameters
    ----------
    cashflows:
        Array of cash flows. Negative = outflows, positive = inflows.
    periods:
        Time periods (years by default). If None, assumes [0, 1, 2, ...].
    guess:
        Initial guess for Newton-Raphson.
    tol:
        Convergence tolerance.

    Returns
    -------
    float
        IRR as a decimal (e.g. 0.25 = 25%). Returns nan if no solution found.
    """
    cashflows = np.asarray(cashflows, dtype=np.float64)
    if periods is None:
        periods = np.arange(len(cashflows), dtype=np.float64)
    else:
        periods = np.asarray(periods, dtype=np.float64)

    if len(cashflows) != len(periods):
        raise ValueError("cashflows and periods must have the same length")

    # Need at least one sign change
    if not (np.any(cashflows > 0) and np.any(cashflows < 0)):
        return float("nan")

    def npv_func(r: float) -> float:
        return float(np.sum(cashflows / (1 + r) ** periods))

    def dnpv_func(r: float) -> float:
        return float(np.sum(-periods * cashflows / (1 + r) ** (periods + 1)))

    # Attempt Newton-Raphson first
    try:
        result = optimize.newton(
            npv_func, x0=guess, fprime=dnpv_func, tol=tol, maxiter=500
        )
        if -1 < result < 100:
            return float(result)
    except (RuntimeError, ValueError):
        pass

    # Brent fallback — bracket search
    try:
        lo, hi = -0.999, 100.0
        if npv_func(lo) * npv_func(hi) < 0:
            result = optimize.brentq(npv_func, lo, hi, xtol=tol, maxiter=1000)
            return float(result)
    except ValueError:
        pass

    return float("nan")


# ---------------------------------------------------------------------------
# Basic return metrics
# ---------------------------------------------------------------------------

def calc_npv(
    cashflows: npt.NDArray[np.float64],
    rate: float,
    periods: Optional[npt.NDArray[np.float64]] = None,
) -> float:
    """Net Present Value at given discount rate."""
    cashflows = np.asarray(cashflows, dtype=np.float64)
    if periods is None:
        periods = np.arange(len(cashflows), dtype=np.float64)
    else:
        periods = np.asarray(periods, dtype=np.float64)
    return float(np.sum(cashflows / (1 + rate) ** periods))


def calc_tvpi(
    invested: float,
    nav: float,
    distributions: float,
) -> float:
    """
    Total Value to Paid-In capital (TVPI).

    TVPI = (NAV + cumulative distributions) / total invested
    """
    if invested <= 0:
        return float("nan")
    return (nav + distributions) / invested


def calc_dpi(invested: float, distributions: float) -> float:
    """Distributions to Paid-In capital (DPI)."""
    if invested <= 0:
        return float("nan")
    return distributions / invested


def calc_rvpi(invested: float, nav: float) -> float:
    """Residual Value to Paid-In capital (RVPI)."""
    if invested <= 0:
        return float("nan")
    return nav / invested


def calc_moic(invested: float, total_value: float) -> float:
    """Multiple on Invested Capital (MOIC = TVPI for fully invested funds)."""
    if invested <= 0:
        return float("nan")
    return total_value / invested


# ---------------------------------------------------------------------------
# Fee modeling
# ---------------------------------------------------------------------------

def calc_management_fees(
    committed: float,
    fee_rate: float,
    investment_period: int,
    fund_life: int,
    step_down_rate: float = 0.0,
    period_unit: str = "quarter",
) -> npt.NDArray[np.float64]:
    """
    Calculate management fees per period over fund life.

    During the investment period fees are charged on committed capital.
    After the investment period, fees step down (if step_down_rate > 0)
    and are charged on the stepped-down rate or deployed capital.

    Parameters
    ----------
    committed:
        Total committed capital (LP commitments).
    fee_rate:
        Annual management fee rate (e.g. 0.02 = 2%).
    investment_period:
        Length of investment period in years.
    fund_life:
        Total fund life in years.
    step_down_rate:
        Annual reduction in fee rate post-investment period (e.g. 0.0025).
    period_unit:
        'quarter' or 'year'.

    Returns
    -------
    ndarray
        Fee per period array of length (fund_life * periods_per_year).
    """
    periods_per_year = 4 if period_unit == "quarter" else 1
    n_periods = fund_life * periods_per_year
    inv_period_end = investment_period * periods_per_year

    fees = np.zeros(n_periods, dtype=np.float64)
    annual_fee = committed * fee_rate

    for t in range(n_periods):
        if t < inv_period_end:
            fees[t] = annual_fee / periods_per_year
        else:
            # Step-down begins immediately at the first post-investment period.
            # years_post starts at 1 (not 0) so the step applies from day one.
            years_post = (t - inv_period_end + 1) / periods_per_year
            stepped_rate = max(fee_rate - step_down_rate * years_post, 0.0)
            fees[t] = committed * stepped_rate / periods_per_year

    return fees


# ---------------------------------------------------------------------------
# Carried interest (American waterfall)
# ---------------------------------------------------------------------------

def calc_carry(
    distributions: npt.NDArray[np.float64],
    invested: float,
    carry_rate: float = 0.20,
    hurdle_rate: float = 0.08,
    catch_up_rate: float = 1.0,
) -> float:
    """
    Compute carried interest under the American (deal-by-deal) waterfall.

    Returns carry attributable to GP assuming all distributions have occurred.

    Parameters
    ----------
    distributions:
        Array of LP distributions over time (positive = money to LPs).
    invested:
        Total capital invested (paid-in capital).
    carry_rate:
        GP carried interest percentage (e.g. 0.20 = 20%).
    hurdle_rate:
        Annual preferred return rate (e.g. 0.08 = 8%).
    catch_up_rate:
        GP catch-up rate (1.0 = full catch-up; 0.0 = no catch-up).

    Returns
    -------
    float
        Total carry amount due to GP.
    """
    total_distributions = float(np.sum(distributions))
    # Hurdle amount: simple interest approximation for American waterfall
    hurdle_amount = invested * (1 + hurdle_rate)

    if total_distributions <= invested:
        # Return of capital only, no carry
        return 0.0

    if total_distributions <= hurdle_amount:
        # Preferred return not fully met
        return 0.0

    profit_above_hurdle = total_distributions - hurdle_amount

    if catch_up_rate >= 1.0:
        # Full catch-up: GP gets carry_rate of all profits
        total_profit = total_distributions - invested
        carry = carry_rate * total_profit
    else:
        # Partial catch-up
        catch_up_amount = profit_above_hurdle * catch_up_rate * carry_rate
        remaining = profit_above_hurdle - catch_up_amount
        carry = catch_up_amount + carry_rate * remaining

    return max(0.0, carry)


# ---------------------------------------------------------------------------
# Public Market Equivalent (PME)
# ---------------------------------------------------------------------------

def pme(
    fund_cashflows: npt.NDArray[np.float64],
    index_returns: npt.NDArray[np.float64],
) -> dict[str, float]:
    """
    Compute Public Market Equivalent metrics.

    Parameters
    ----------
    fund_cashflows:
        Array of fund net cash flows per period. Negative = capital calls,
        positive = distributions. Last element treated as NAV if fund is open.
    index_returns:
        Period returns of the public market index (e.g. 0.03 = 3% per period).
        Must have same length as fund_cashflows.

    Returns
    -------
    dict with keys:
        ks_pme     : Kaplan-Schoar PME ratio (>1 = outperformance)
        pme_alpha  : Annualized alpha vs index
        direct_alpha: Direct Alpha (IRR of combined fund + index cashflows)
    """
    fund_cashflows = np.asarray(fund_cashflows, dtype=np.float64)
    index_returns = np.asarray(index_returns, dtype=np.float64)

    n = len(fund_cashflows)
    # Compound index return from each period to end
    index_factor = np.ones(n, dtype=np.float64)
    for t in range(n - 2, -1, -1):
        index_factor[t] = index_factor[t + 1] * (1 + index_returns[t + 1])

    calls = np.where(fund_cashflows < 0, -fund_cashflows, 0.0)
    distributions = np.where(fund_cashflows > 0, fund_cashflows, 0.0)

    fv_calls = np.sum(calls * index_factor)
    fv_distributions = np.sum(distributions * index_factor)

    ks_pme = fv_distributions / fv_calls if fv_calls > 0 else float("nan")

    # PME alpha: annualized excess return
    n_periods = n - 1  # assume annual periods
    pme_alpha = (ks_pme ** (1.0 / n_periods) - 1) if n_periods > 0 and ks_pme > 0 else float("nan")

    # Direct alpha: IRR of fund cashflows plus index-adjusted cashflows
    index_cumprod = np.cumprod(1 + index_returns)
    scaled = fund_cashflows / np.concatenate([[1.0], index_cumprod[:-1]])
    direct_alpha = calc_irr(scaled)

    return {
        "ks_pme": float(ks_pme),
        "pme_alpha": float(pme_alpha),
        "direct_alpha": float(direct_alpha),
    }
