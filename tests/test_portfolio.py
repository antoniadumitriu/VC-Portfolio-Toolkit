"""Tests for vc_portfolio.portfolio — power law simulation."""
from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from vc_portfolio.portfolio import Company, Portfolio, STAGE_LOSS_RATES


@pytest.fixture
def sample_companies() -> list[Company]:
    return [
        Company(
            name="TechCo",
            investment_amount=2_000_000,
            stage="seed",
            sector="saas",
            entry_valuation=10_000_000,
            entry_period=0,
            expected_exit_period=7,
            ownership_pct=0.15,
            reserved_capital=1_000_000,
        ),
        Company(
            name="HealthAI",
            investment_amount=5_000_000,
            stage="series_a",
            sector="healthtech",
            entry_valuation=25_000_000,
            entry_period=2,
            expected_exit_period=8,
            ownership_pct=0.12,
            reserved_capital=2_500_000,
        ),
        Company(
            name="FinTechX",
            investment_amount=3_000_000,
            stage="seed",
            sector="fintech",
            entry_valuation=15_000_000,
            entry_period=1,
            expected_exit_period=6,
            ownership_pct=0.10,
            reserved_capital=1_500_000,
        ),
        Company(
            name="ClimateB",
            investment_amount=4_000_000,
            stage="series_b",
            sector="climate",
            entry_valuation=40_000_000,
            entry_period=0,
            expected_exit_period=5,
            ownership_pct=0.08,
            reserved_capital=0,
        ),
    ]


@pytest.fixture
def portfolio(sample_companies: list[Company]) -> Portfolio:
    return Portfolio(seed=42).add_companies(sample_companies)


class TestCompany:
    def test_total_committed(self):
        c = Company("A", 1_000_000, "seed", "tech", 5_000_000, 0, 5, 0.10, 500_000)
        assert c.total_committed == pytest.approx(1_500_000)

    def test_holding_period(self):
        c = Company("A", 1_000_000, "seed", "tech", 5_000_000, 2, 8, 0.10)
        assert c.holding_period == 6


class TestPortfolioBuilder:
    def test_add_company_returns_self(self):
        p = Portfolio(seed=0)
        c = Company("A", 1e6, "seed", "tech", 5e6, 0, 5, 0.1)
        assert p.add_company(c) is p

    def test_add_companies_returns_self(self, sample_companies):
        p = Portfolio(seed=0)
        assert p.add_companies(sample_companies) is p

    def test_len(self, portfolio: Portfolio):
        assert len(portfolio) == 4

    def test_repr(self, portfolio: Portfolio):
        r = repr(portfolio)
        assert "Portfolio" in r
        assert "n_companies=4" in r


class TestStageLossRates:
    def test_all_stages_defined(self):
        for stage in ["pre_seed", "seed", "series_a", "series_b", "series_c", "growth"]:
            assert stage in STAGE_LOSS_RATES

    def test_loss_rates_decrease_with_stage(self):
        assert STAGE_LOSS_RATES["pre_seed"] > STAGE_LOSS_RATES["seed"]
        assert STAGE_LOSS_RATES["seed"] > STAGE_LOSS_RATES["series_a"]
        assert STAGE_LOSS_RATES["series_a"] > STAGE_LOSS_RATES["series_b"]

    def test_loss_rates_in_valid_range(self):
        for rate in STAGE_LOSS_RATES.values():
            assert 0 < rate < 1


class TestExitMultiples:
    def test_draw_returns_correct_shape(self):
        p = Portfolio(seed=42)
        multiples = p._draw_exit_multiple("seed", n=1000)
        assert multiples.shape == (1000,)

    def test_multiples_nonnegative(self):
        p = Portfolio(seed=42)
        for stage in STAGE_LOSS_RATES:
            multiples = p._draw_exit_multiple(stage, n=1000)
            assert np.all(multiples >= 0)

    def test_loss_rate_approximately_correct(self):
        """Seed loss rate should be ~50% with large sample."""
        p = Portfolio(seed=42)
        multiples = p._draw_exit_multiple("seed", n=50_000)
        observed_loss_rate = float(np.mean(multiples == 0))
        # Allow 5% tolerance
        assert abs(observed_loss_rate - 0.50) < 0.05

    def test_growth_stage_lower_loss_rate(self):
        p = Portfolio(seed=42)
        seed_multiples = p._draw_exit_multiple("seed", n=10_000)
        growth_multiples = p._draw_exit_multiple("growth", n=10_000)
        seed_loss = float(np.mean(seed_multiples == 0))
        growth_loss = float(np.mean(growth_multiples == 0))
        # Growth stage should have fewer losses
        assert growth_loss < seed_loss


class TestSimulateExits:
    def test_returns_required_keys(self, portfolio: Portfolio):
        result = portfolio.simulate_exits(n_simulations=1000)
        required_keys = {
            "moic_p10", "moic_p25", "moic_p50", "moic_p75", "moic_p90",
            "irr_p10", "irr_p25", "irr_p50", "irr_p75", "irr_p90",
            "loss_probability", "home_run_probability",
            "raw_moics", "raw_irrs",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_percentile_ordering(self, portfolio: Portfolio):
        result = portfolio.simulate_exits(n_simulations=2000)
        assert result["moic_p10"] <= result["moic_p25"]
        assert result["moic_p25"] <= result["moic_p50"]
        assert result["moic_p50"] <= result["moic_p75"]
        assert result["moic_p75"] <= result["moic_p90"]

    def test_probabilities_in_range(self, portfolio: Portfolio):
        result = portfolio.simulate_exits(n_simulations=1000)
        assert 0 <= result["loss_probability"] <= 1
        assert 0 <= result["home_run_probability"] <= 1

    def test_raw_moics_shape(self, portfolio: Portfolio):
        result = portfolio.simulate_exits(n_simulations=500)
        assert result["raw_moics"].shape == (500,)

    def test_empty_portfolio_raises(self):
        p = Portfolio(seed=0)
        with pytest.raises(ValueError):
            p.simulate_exits()

    def test_reproducibility_with_seed(self, sample_companies: list[Company]):
        p1 = Portfolio(seed=123).add_companies(sample_companies)
        p2 = Portfolio(seed=123).add_companies(sample_companies)
        r1 = p1.simulate_exits(n_simulations=100)
        r2 = p2.simulate_exits(n_simulations=100)
        np.testing.assert_array_equal(r1["raw_moics"], r2["raw_moics"])

    def test_median_moic_reasonable(self, portfolio: Portfolio):
        """Mixed early/growth stage portfolio should have median MOIC 1–4x."""
        result = portfolio.simulate_exits(n_simulations=5000)
        assert 0.5 <= result["moic_p50"] <= 5.0


class TestPortfolioAnalytics:
    def test_reserve_ratio(self, portfolio: Portfolio):
        r = portfolio.reserve_ratio()
        assert "reserve_ratio" in r
        assert r["reserve_ratio"] >= 0

    def test_portfolio_breakdown_returns_dataframe(self, portfolio: Portfolio):
        df = portfolio.portfolio_breakdown()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_portfolio_breakdown_columns(self, portfolio: Portfolio):
        df = portfolio.portfolio_breakdown()
        assert "company" in df.columns
        assert "portfolio_weight" in df.columns

    def test_portfolio_weights_sum_to_one(self, portfolio: Portfolio):
        df = portfolio.portfolio_breakdown()
        assert df["portfolio_weight"].sum() == pytest.approx(1.0)

    def test_concentration_analysis(self, portfolio: Portfolio):
        conc = portfolio.concentration_analysis()
        assert "herfindahl_index" in conc
        assert "top_3_concentration" in conc
        assert 0 <= conc["herfindahl_index"] <= 1
        assert conc["stage_diversity"] >= 1
        assert conc["sector_diversity"] >= 1

    def test_follow_on_pro_rata(self, portfolio: Portfolio):
        df = portfolio.follow_on_strategy("pro_rata")
        assert isinstance(df, pd.DataFrame)
        # All reserved capital deployed
        total_follow_on = df["follow_on"].sum()
        expected = sum(c.reserved_capital for c in portfolio._companies)
        assert total_follow_on == pytest.approx(expected)

    def test_follow_on_none(self, portfolio: Portfolio):
        df = portfolio.follow_on_strategy("none")
        assert df["follow_on"].sum() == pytest.approx(0.0)

    def test_follow_on_top_up_only_late_stage(self, portfolio: Portfolio):
        df = portfolio.follow_on_strategy("top_up")
        # Only series_b and above get follow-on in top_up strategy
        for _, row in df.iterrows():
            if row["stage"] not in ("series_b", "series_c", "growth"):
                assert row["follow_on"] == pytest.approx(0.0)
