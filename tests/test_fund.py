"""Tests for vc_portfolio.fund — Fund cash flow ledger."""
from __future__ import annotations

import math

import numpy as np
import pytest

from vc_portfolio.fund import Fund, FundConfig


@pytest.fixture
def basic_config() -> FundConfig:
    return FundConfig(
        name="Test Fund I",
        committed_capital=100_000_000,
        vintage_year=2020,
        fee_rate=0.02,
        carry_rate=0.20,
        hurdle_rate=0.08,
        investment_period=5,
        fund_life=10,
        period_unit="year",
    )


@pytest.fixture
def basic_fund(basic_config: FundConfig) -> Fund:
    return (
        Fund(basic_config)
        .deploy_capital({0: 20_000_000, 1: 20_000_000, 2: 15_000_000, 3: 10_000_000})
        .add_distribution(10_000_000, period=5)
        .add_distribution(30_000_000, period=7)
        .add_distribution(50_000_000, period=9)
        .set_nav(6, 60_000_000)
        .set_nav(8, 40_000_000)
    )


class TestFundConfig:
    def test_n_periods_annual(self):
        cfg = FundConfig("F", 1e8, 2020, fund_life=10, period_unit="year")
        assert cfg.n_periods == 10

    def test_n_periods_quarterly(self):
        cfg = FundConfig("F", 1e8, 2020, fund_life=10, period_unit="quarter")
        assert cfg.n_periods == 40

    def test_periods_per_year(self):
        cfg_q = FundConfig("F", 1e8, 2020, period_unit="quarter")
        cfg_a = FundConfig("F", 1e8, 2020, period_unit="year")
        assert cfg_q.periods_per_year == 4
        assert cfg_a.periods_per_year == 1


class TestFundBuilderMethods:
    def test_method_chaining_returns_self(self, basic_config: FundConfig):
        fund = Fund(basic_config)
        result = fund.deploy_capital({0: 1_000_000})
        assert result is fund

    def test_deploy_capital_accumulates(self, basic_config: FundConfig):
        fund = Fund(basic_config)
        fund.deploy_capital({0: 10_000_000, 0: 5_000_000})  # last wins for same key
        assert fund.total_invested == pytest.approx(5_000_000)

    def test_add_distribution_accumulates(self, basic_config: FundConfig):
        fund = Fund(basic_config)
        fund.add_distribution(5_000_000, period=5)
        fund.add_distribution(5_000_000, period=5)
        assert fund.total_distributions == pytest.approx(10_000_000)

    def test_set_nav(self, basic_config: FundConfig):
        fund = Fund(basic_config)
        fund.set_nav(5, 50_000_000)
        assert fund.current_nav == pytest.approx(50_000_000)

    def test_invalid_period_raises(self, basic_config: FundConfig):
        fund = Fund(basic_config)
        with pytest.raises(ValueError):
            fund.deploy_capital({100: 1_000_000})

    def test_negative_amount_raises(self, basic_config: FundConfig):
        fund = Fund(basic_config)
        with pytest.raises(ValueError):
            fund.deploy_capital({0: -1_000_000})

    def test_negative_nav_raises(self, basic_config: FundConfig):
        fund = Fund(basic_config)
        with pytest.raises(ValueError):
            fund.set_nav(0, -1_000_000)


class TestFundMetrics:
    def test_total_invested(self, basic_fund: Fund):
        assert basic_fund.total_invested == pytest.approx(65_000_000)

    def test_total_distributions(self, basic_fund: Fund):
        assert basic_fund.total_distributions == pytest.approx(90_000_000)

    def test_current_nav(self, basic_fund: Fund):
        # Last NAV set is at period 8
        assert basic_fund.current_nav == pytest.approx(40_000_000)

    def test_tvpi_gt_one(self, basic_fund: Fund):
        # Fund should have positive TVPI (distributions + NAV > invested)
        assert basic_fund.tvpi() > 1.0

    def test_dpi_and_rvpi_sum_to_tvpi(self, basic_fund: Fund):
        assert basic_fund.tvpi() == pytest.approx(
            basic_fund.dpi() + basic_fund.rvpi(), rel=1e-6
        )

    def test_irr_is_finite(self, basic_fund: Fund):
        irr = basic_fund.irr()
        assert not math.isnan(irr)
        assert -1 < irr < 10

    def test_irr_net_lte_gross(self, basic_fund: Fund):
        irr_net = basic_fund.irr(net_of_carry=True)
        irr_gross = basic_fund.irr(net_of_carry=False)
        assert irr_net <= irr_gross + 1e-9  # net ≤ gross

    def test_moic_equals_tvpi(self, basic_fund: Fund):
        # For a fund where total_value = distributions + current_nav
        assert basic_fund.moic() == pytest.approx(basic_fund.tvpi(), rel=1e-6)


class TestFundCashflows:
    def test_get_cashflows_returns_dataframe(self, basic_fund: Fund):
        import pandas as pd
        df = basic_fund.get_cashflows()
        assert isinstance(df, pd.DataFrame)

    def test_get_cashflows_columns(self, basic_fund: Fund):
        df = basic_fund.get_cashflows()
        expected_cols = {
            "period", "capital_called", "distributions", "fees",
            "nav", "cumulative_called", "cumulative_distributions", "net_cashflow"
        }
        assert expected_cols.issubset(set(df.columns))

    def test_get_cashflows_length(self, basic_config: FundConfig, basic_fund: Fund):
        df = basic_fund.get_cashflows()
        assert len(df) == basic_config.n_periods

    def test_lp_cashflows_length(self, basic_config: FundConfig, basic_fund: Fund):
        lp_cfs = basic_fund.get_lp_cashflows()
        assert len(lp_cfs) == basic_config.n_periods

    def test_summary_keys(self, basic_fund: Fund):
        s = basic_fund.summary()
        for key in ["irr_net", "tvpi", "dpi", "rvpi", "moic", "carry"]:
            assert key in s

    def test_repr(self, basic_fund: Fund):
        r = repr(basic_fund)
        assert "Test Fund I" in r
        assert "tvpi=" in r


class TestFundEdgeCases:
    def test_empty_fund(self, basic_config: FundConfig):
        """Fund with no calls/distributions should return nan IRR."""
        fund = Fund(basic_config)
        assert math.isnan(fund.irr())

    def test_quarterly_fund(self):
        config = FundConfig(
            name="Quarterly Fund",
            committed_capital=50_000_000,
            vintage_year=2021,
            period_unit="quarter",
            fund_life=10,
        )
        fund = (
            Fund(config)
            .deploy_capital({0: 10_000_000, 4: 10_000_000})
            .add_distribution(30_000_000, period=20)
            .set_nav(30, 10_000_000)
        )
        assert fund.tvpi() > 1.0
        assert len(fund.get_cashflows()) == 40
