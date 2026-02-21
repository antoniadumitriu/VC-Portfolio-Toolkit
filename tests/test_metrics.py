"""Tests for vc_portfolio.metrics — pure math functions."""
from __future__ import annotations

import math

import numpy as np
import pytest

from vc_portfolio.metrics import (
    calc_carry,
    calc_dpi,
    calc_irr,
    calc_management_fees,
    calc_moic,
    calc_npv,
    calc_rvpi,
    calc_tvpi,
    pme,
)


# ---------------------------------------------------------------------------
# IRR tests
# ---------------------------------------------------------------------------

class TestCalcIrr:
    def test_simple_doubling(self):
        """$1 invested, $2 returned after 1 year → 100% IRR."""
        cashflows = np.array([-1.0, 2.0])
        assert calc_irr(cashflows) == pytest.approx(1.0, rel=1e-4)

    def test_standard_vc_fund(self):
        """Typical VC fund: calls then distributions."""
        cashflows = np.array([-100.0, -50.0, 0.0, 50.0, 150.0, 200.0])
        irr = calc_irr(cashflows)
        assert 0.15 < irr < 0.50  # reasonable VC return

    def test_negative_irr(self):
        """Fund that loses money."""
        cashflows = np.array([-100.0, 30.0, 30.0])
        irr = calc_irr(cashflows)
        assert irr < 0

    def test_no_sign_change_returns_nan(self):
        """All outflows: no solution."""
        cashflows = np.array([-100.0, -50.0])
        assert math.isnan(calc_irr(cashflows))

    def test_custom_periods(self):
        """Non-uniform periods (semi-annual)."""
        cashflows = np.array([-1.0, 1.5])
        periods = np.array([0.0, 0.5])
        irr = calc_irr(cashflows, periods=periods)
        # Verify: -1 + 1.5/(1+r)^0.5 = 0 → r = (1.5)^2 - 1 = 1.25
        assert irr == pytest.approx(1.25, rel=1e-4)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calc_irr(np.array([-1.0, 2.0]), periods=np.array([0.0, 1.0, 2.0]))


# ---------------------------------------------------------------------------
# NPV tests
# ---------------------------------------------------------------------------

class TestCalcNpv:
    def test_zero_rate(self):
        cashflows = np.array([-100.0, 50.0, 50.0, 50.0])
        assert calc_npv(cashflows, rate=0.0) == pytest.approx(50.0)

    def test_positive_npv(self):
        cashflows = np.array([-100.0, 60.0, 60.0])
        npv = calc_npv(cashflows, rate=0.10)
        assert npv > 0

    def test_at_irr_npv_is_zero(self):
        cashflows = np.array([-1.0, 2.0])
        irr = calc_irr(cashflows)
        assert calc_npv(cashflows, rate=irr) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TVPI / DPI / RVPI / MOIC tests
# ---------------------------------------------------------------------------

class TestMultiples:
    def test_tvpi_basic(self):
        assert calc_tvpi(invested=100.0, nav=50.0, distributions=100.0) == pytest.approx(1.5)

    def test_dpi_basic(self):
        assert calc_dpi(invested=100.0, distributions=80.0) == pytest.approx(0.8)

    def test_rvpi_basic(self):
        assert calc_rvpi(invested=100.0, nav=60.0) == pytest.approx(0.6)

    def test_moic_basic(self):
        assert calc_moic(invested=100.0, total_value=300.0) == pytest.approx(3.0)

    def test_tvpi_equals_dpi_plus_rvpi(self):
        invested = 100.0
        nav = 40.0
        distributions = 120.0
        tvpi = calc_tvpi(invested, nav, distributions)
        assert tvpi == pytest.approx(calc_dpi(invested, distributions) + calc_rvpi(invested, nav))

    def test_zero_invested_returns_nan(self):
        assert math.isnan(calc_tvpi(0.0, 10.0, 10.0))
        assert math.isnan(calc_dpi(0.0, 10.0))
        assert math.isnan(calc_rvpi(0.0, 10.0))
        assert math.isnan(calc_moic(0.0, 10.0))


# ---------------------------------------------------------------------------
# Management fee tests
# ---------------------------------------------------------------------------

class TestCalcManagementFees:
    def test_quarterly_constant_fees(self):
        fees = calc_management_fees(
            committed=100_000_000,
            fee_rate=0.02,
            investment_period=5,
            fund_life=10,
            step_down_rate=0.0,
            period_unit="quarter",
        )
        assert len(fees) == 40  # 10 years * 4 quarters
        # During investment period (first 20 quarters): 2M/4 = 500K per quarter
        assert fees[0] == pytest.approx(500_000.0)
        assert fees[19] == pytest.approx(500_000.0)

    def test_step_down_reduces_fees_post_investment(self):
        fees = calc_management_fees(
            committed=100_000_000,
            fee_rate=0.02,
            investment_period=5,
            fund_life=10,
            step_down_rate=0.0025,
            period_unit="year",
        )
        assert len(fees) == 10
        # Post-investment period fees should be lower
        assert fees[5] < fees[4]
        assert fees[9] <= fees[5]

    def test_annual_periods(self):
        fees = calc_management_fees(
            committed=100_000_000,
            fee_rate=0.02,
            investment_period=5,
            fund_life=10,
            period_unit="year",
        )
        assert len(fees) == 10
        assert fees[0] == pytest.approx(2_000_000.0)

    def test_nonnegative_fees(self):
        fees = calc_management_fees(
            committed=100_000_000,
            fee_rate=0.02,
            investment_period=3,
            fund_life=10,
            step_down_rate=0.01,
            period_unit="year",
        )
        assert np.all(fees >= 0)


# ---------------------------------------------------------------------------
# Carry tests
# ---------------------------------------------------------------------------

class TestCalcCarry:
    def test_no_carry_below_return_of_capital(self):
        distributions = np.array([50.0, 30.0])  # total = 80 < 100 invested
        carry = calc_carry(distributions, invested=100.0, carry_rate=0.20, hurdle_rate=0.08)
        assert carry == pytest.approx(0.0)

    def test_no_carry_below_hurdle(self):
        distributions = np.array([105.0])  # 5% return, below 8% hurdle
        carry = calc_carry(distributions, invested=100.0, carry_rate=0.20, hurdle_rate=0.08)
        assert carry == pytest.approx(0.0)

    def test_carry_above_hurdle(self):
        # 150 distributed on 100 invested with 8% hurdle → 50 profit above cost, 42 above hurdle
        distributions = np.array([150.0])
        carry = calc_carry(distributions, invested=100.0, carry_rate=0.20, hurdle_rate=0.08)
        # Full catch-up: 20% of (150-100) = 10
        assert carry == pytest.approx(10.0)

    def test_carry_is_nonnegative(self):
        distributions = np.array([1.0])
        carry = calc_carry(distributions, invested=100.0)
        assert carry >= 0.0


# ---------------------------------------------------------------------------
# PME tests
# ---------------------------------------------------------------------------

class TestPme:
    def test_returns_required_keys(self):
        fund_cfs = np.array([-1.0, -0.5, 0.5, 1.5])
        index_rets = np.array([0.08, 0.08, 0.08, 0.08])
        result = pme(fund_cfs, index_rets)
        assert "ks_pme" in result
        assert "pme_alpha" in result
        assert "direct_alpha" in result

    def test_outperforming_fund(self):
        """Fund returning 30% vs 8% index should have KS-PME > 1."""
        fund_cfs = np.array([-100.0, 0.0, 0.0, 200.0])
        index_rets = np.array([0.08, 0.08, 0.08, 0.08])
        result = pme(fund_cfs, index_rets)
        assert result["ks_pme"] > 1.0

    def test_underperforming_fund(self):
        """Fund returning 5% vs 8% index should have KS-PME < 1."""
        fund_cfs = np.array([-100.0, 0.0, 0.0, 115.0])
        index_rets = np.array([0.08, 0.08, 0.08, 0.08])
        result = pme(fund_cfs, index_rets)
        assert result["ks_pme"] < 1.0
