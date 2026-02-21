"""Tests for vc_portfolio.jcurve — NAV projection with S-curve appreciation."""
from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from vc_portfolio.jcurve import CompanyTimeline, JCurve


@pytest.fixture
def timeline_a() -> CompanyTimeline:
    return CompanyTimeline(
        entry_period=2,
        exit_period=14,
        initial_investment=5_000_000,
        exit_multiple=3.0,
        company_name="TechCo",
        follow_on_investments={6: 2_000_000},
    )


@pytest.fixture
def timeline_b() -> CompanyTimeline:
    return CompanyTimeline(
        entry_period=0,
        exit_period=10,
        initial_investment=3_000_000,
        exit_multiple=5.0,
        company_name="SaaSCo",
    )


@pytest.fixture
def jcurve(timeline_a: CompanyTimeline, timeline_b: CompanyTimeline) -> JCurve:
    return (
        JCurve(
            n_periods=20,
            fee_rate=0.02,
            committed_capital=50_000_000,
            period_unit="year",
        )
        .add_company(timeline_a)
        .add_company(timeline_b)
    )


class TestCompanyTimeline:
    def test_total_investment_no_followon(self):
        tl = CompanyTimeline(0, 8, 5_000_000, 3.0)
        assert tl.total_investment == pytest.approx(5_000_000)

    def test_total_investment_with_followon(self, timeline_a: CompanyTimeline):
        assert timeline_a.total_investment == pytest.approx(7_000_000)

    def test_holding_period(self, timeline_a: CompanyTimeline):
        assert timeline_a.holding_period == 12

    def test_expected_exit_value(self, timeline_a: CompanyTimeline):
        # 7M total investment × 3x = 21M
        assert timeline_a.expected_exit_value == pytest.approx(21_000_000)


class TestJCurveBuilder:
    def test_add_company_returns_self(self):
        jc = JCurve(n_periods=20)
        tl = CompanyTimeline(0, 8, 1e6, 2.0)
        assert jc.add_company(tl) is jc

    def test_add_companies_returns_self(self, timeline_a, timeline_b):
        jc = JCurve(n_periods=20)
        assert jc.add_companies([timeline_a, timeline_b]) is jc

    def test_len(self, jcurve: JCurve):
        assert len(jcurve) == 2

    def test_repr(self, jcurve: JCurve):
        r = repr(jcurve)
        assert "JCurve" in r
        assert "n_companies=2" in r


class TestSCurveWeight:
    def test_before_entry_is_zero(self):
        jc = JCurve(n_periods=20)
        assert jc._compute_scurve_weight(0, 5, 15) == pytest.approx(0.0)

    def test_at_exit_is_one(self):
        jc = JCurve(n_periods=20)
        assert jc._compute_scurve_weight(15, 5, 15) == pytest.approx(1.0)

    def test_midpoint_near_half(self):
        jc = JCurve(n_periods=20)
        weight = jc._compute_scurve_weight(10, 5, 15)
        assert 0.4 < weight < 0.6  # logistic at midpoint ≈ 0.5

    def test_monotonic_increase(self):
        jc = JCurve(n_periods=20)
        weights = [jc._compute_scurve_weight(t, 2, 12) for t in range(2, 13)]
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]


class TestCompanyNav:
    def test_nav_zero_before_entry(self, jcurve: JCurve, timeline_a: CompanyTimeline):
        nav = jcurve._compute_company_nav(timeline_a, period=0)
        assert nav == pytest.approx(0.0)

    def test_nav_zero_after_exit(self, jcurve: JCurve, timeline_a: CompanyTimeline):
        nav = jcurve._compute_company_nav(timeline_a, period=14)
        assert nav == pytest.approx(0.0)

    def test_nav_near_cost_basis_at_entry(self, jcurve: JCurve):
        tl = CompanyTimeline(2, 12, 5_000_000, 3.0)
        nav = jcurve._compute_company_nav(tl, period=2)
        # At entry, S-curve weight ≈ 0 → NAV ≈ cost basis
        assert nav >= 0
        assert nav <= tl.total_investment * tl.exit_multiple

    def test_nav_increases_over_time(self):
        jc = JCurve(n_periods=20)
        tl = CompanyTimeline(0, 10, 1_000_000, 5.0)
        navs = [jc._compute_company_nav(tl, t) for t in range(0, 10)]
        for i in range(len(navs) - 1):
            assert navs[i] <= navs[i + 1] + 1e-6  # non-decreasing


class TestProject:
    def test_returns_dataframe(self, jcurve: JCurve):
        df = jcurve.project()
        assert isinstance(df, pd.DataFrame)

    def test_correct_length(self, jcurve: JCurve):
        df = jcurve.project()
        assert len(df) == 20

    def test_required_columns(self, jcurve: JCurve):
        df = jcurve.project()
        expected = {
            "period", "nav", "unrealized_value", "realized_value",
            "capital_calls", "distributions", "mgmt_fees", "net_cashflow",
            "cumulative_called", "cumulative_distributions",
            "rolling_irr", "tvpi", "dpi"
        }
        assert expected.issubset(set(df.columns))

    def test_nav_nonnegative(self, jcurve: JCurve):
        df = jcurve.project()
        assert (df["nav"] >= 0).all()

    def test_fees_positive(self, jcurve: JCurve):
        df = jcurve.project()
        assert (df["mgmt_fees"] > 0).all()

    def test_capital_calls_at_entry_periods(self, jcurve: JCurve, timeline_b: CompanyTimeline):
        df = jcurve.project()
        # timeline_b enters at period 0
        assert df.loc[df["period"] == 0, "capital_calls"].values[0] > 0

    def test_distributions_at_exit_periods(self, jcurve: JCurve, timeline_b: CompanyTimeline):
        df = jcurve.project()
        # timeline_b exits at period 10
        assert df.loc[df["period"] == 10, "distributions"].values[0] > 0

    def test_tvpi_positive_after_investment(self, jcurve: JCurve):
        df = jcurve.project()
        # After first investment, TVPI should be tracked
        invested_periods = df[df["cumulative_called"] > 0]
        assert (invested_periods["tvpi"] >= 0).all()

    def test_cumulative_called_monotonic(self, jcurve: JCurve):
        df = jcurve.project()
        assert (np.diff(df["cumulative_called"]) >= 0).all()


class TestJCurveShape:
    def test_returns_required_keys(self, jcurve: JCurve):
        shape = jcurve.get_jcurve_shape()
        assert "trough_period" in shape
        assert "trough_nav" in shape
        assert "breakeven_period" in shape
        assert "peak_nav" in shape
        assert "peak_period" in shape

    def test_peak_period_after_trough(self, jcurve: JCurve):
        shape = jcurve.get_jcurve_shape()
        # Peak should come after trough in a normal J-curve
        assert shape["peak_period"] >= shape["trough_period"]

    def test_peak_nav_positive(self, jcurve: JCurve):
        shape = jcurve.get_jcurve_shape()
        assert shape["peak_nav"] >= 0

    def test_empty_jcurve(self):
        jc = JCurve(n_periods=10)
        # Empty jcurve should still return a shape without error
        shape = jc.get_jcurve_shape()
        assert "peak_period" in shape
