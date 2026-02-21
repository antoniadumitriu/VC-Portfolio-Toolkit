"""
conftest.py — Shared pytest fixtures for vc_portfolio test suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from vc_portfolio.fund import Fund, FundConfig
from vc_portfolio.jcurve import CompanyTimeline, JCurve
from vc_portfolio.portfolio import Company, Portfolio


@pytest.fixture(scope="session")
def default_fund_config() -> FundConfig:
    """Standard 10-year fund with annual periods."""
    return FundConfig(
        name="Benchmark Fund I",
        committed_capital=100_000_000,
        vintage_year=2020,
        fee_rate=0.02,
        carry_rate=0.20,
        hurdle_rate=0.08,
        investment_period=5,
        fund_life=10,
        step_down_rate=0.0025,
        period_unit="year",
    )


@pytest.fixture(scope="session")
def default_fund(default_fund_config: FundConfig) -> Fund:
    """A fully-configured fund with calls, distributions, and NAV marks."""
    return (
        Fund(default_fund_config)
        .deploy_capital(
            {0: 15_000_000, 1: 20_000_000, 2: 15_000_000, 3: 10_000_000, 4: 5_000_000}
        )
        .add_distribution(15_000_000, period=5)
        .add_distribution(25_000_000, period=7)
        .add_distribution(40_000_000, period=9)
        .set_nav(6, 80_000_000)
        .set_nav(8, 55_000_000)
        .set_nav(9, 10_000_000)
    )


@pytest.fixture(scope="session")
def sample_companies() -> list[Company]:
    """A diversified set of portfolio companies."""
    return [
        Company(
            name="AlphaAI",
            investment_amount=3_000_000,
            stage="seed",
            sector="artificial_intelligence",
            entry_valuation=12_000_000,
            entry_period=0,
            expected_exit_period=7,
            ownership_pct=0.15,
            reserved_capital=1_500_000,
        ),
        Company(
            name="BetaHealth",
            investment_amount=5_000_000,
            stage="series_a",
            sector="healthtech",
            entry_valuation=25_000_000,
            entry_period=1,
            expected_exit_period=8,
            ownership_pct=0.12,
            reserved_capital=2_500_000,
        ),
        Company(
            name="GammaFintech",
            investment_amount=2_000_000,
            stage="pre_seed",
            sector="fintech",
            entry_valuation=6_000_000,
            entry_period=0,
            expected_exit_period=9,
            ownership_pct=0.20,
            reserved_capital=1_000_000,
        ),
        Company(
            name="DeltaClimate",
            investment_amount=4_000_000,
            stage="series_b",
            sector="climate",
            entry_valuation=40_000_000,
            entry_period=2,
            expected_exit_period=6,
            ownership_pct=0.08,
            reserved_capital=0,
        ),
        Company(
            name="EpsilonSaaS",
            investment_amount=3_500_000,
            stage="series_a",
            sector="saas",
            entry_valuation=18_000_000,
            entry_period=1,
            expected_exit_period=7,
            ownership_pct=0.14,
            reserved_capital=2_000_000,
        ),
    ]


@pytest.fixture(scope="session")
def default_portfolio(sample_companies: list[Company]) -> Portfolio:
    """Portfolio with fixed seed for reproducibility."""
    return Portfolio(seed=42).add_companies(sample_companies)


@pytest.fixture(scope="session")
def sample_timelines() -> list[CompanyTimeline]:
    """Sample J-curve timelines for testing."""
    return [
        CompanyTimeline(
            entry_period=0,
            exit_period=8,
            initial_investment=10_000_000,
            exit_multiple=4.0,
            company_name="JCo Alpha",
        ),
        CompanyTimeline(
            entry_period=2,
            exit_period=12,
            initial_investment=8_000_000,
            exit_multiple=3.0,
            company_name="JCo Beta",
            follow_on_investments={5: 3_000_000},
        ),
        CompanyTimeline(
            entry_period=1,
            exit_period=10,
            initial_investment=5_000_000,
            exit_multiple=2.0,
            company_name="JCo Gamma (zombie)",
        ),
    ]


@pytest.fixture(scope="session")
def default_jcurve(sample_timelines: list[CompanyTimeline]) -> JCurve:
    """JCurve with quarterly periods for comprehensive testing."""
    return (
        JCurve(
            n_periods=40,
            fee_rate=0.02,
            committed_capital=100_000_000,
            period_unit="quarter",
        )
        .add_companies(sample_timelines)
    )
