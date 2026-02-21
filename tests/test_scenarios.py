"""Tests for vc_portfolio.scenarios — orchestration and Monte Carlo."""
from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from vc_portfolio.portfolio import Company, Portfolio
from vc_portfolio.scenarios import (
    BASE_SCENARIO,
    BEAR_SCENARIO,
    BULL_SCENARIO,
    MonteCarloSimulator,
    Scenario,
    ScenarioResult,
    ScenarioResults,
    ScenarioSet,
    SensitivityAnalysis,
)


@pytest.fixture
def sample_companies() -> list[Company]:
    return [
        Company(
            name="StartupA",
            investment_amount=3_000_000,
            stage="seed",
            sector="saas",
            entry_valuation=15_000_000,
            entry_period=0,
            expected_exit_period=7,
            ownership_pct=0.12,
        ),
        Company(
            name="StartupB",
            investment_amount=5_000_000,
            stage="series_a",
            sector="fintech",
            entry_valuation=25_000_000,
            entry_period=1,
            expected_exit_period=8,
            ownership_pct=0.10,
        ),
        Company(
            name="StartupC",
            investment_amount=2_000_000,
            stage="pre_seed",
            sector="biotech",
            entry_valuation=8_000_000,
            entry_period=0,
            expected_exit_period=9,
            ownership_pct=0.18,
        ),
    ]


@pytest.fixture
def portfolio(sample_companies: list[Company]) -> Portfolio:
    return Portfolio(seed=42).add_companies(sample_companies)


class TestScenarioDefinitions:
    def test_bear_scenario_has_negative_adjustments(self):
        assert BEAR_SCENARIO.portfolio_return_adjustment < 1.0
        assert BEAR_SCENARIO.exit_timing_adjustment > 0
        assert BEAR_SCENARIO.loss_rate_adjustment > 0

    def test_bull_scenario_has_positive_adjustments(self):
        assert BULL_SCENARIO.portfolio_return_adjustment > 1.0
        assert BULL_SCENARIO.exit_timing_adjustment < 0
        assert BULL_SCENARIO.loss_rate_adjustment < 0

    def test_base_scenario_is_neutral(self):
        assert BASE_SCENARIO.portfolio_return_adjustment == pytest.approx(1.0)
        assert BASE_SCENARIO.exit_timing_adjustment == 0
        assert BASE_SCENARIO.loss_rate_adjustment == pytest.approx(0.0)

    def test_scenario_has_color(self):
        assert BEAR_SCENARIO.color
        assert BASE_SCENARIO.color
        assert BULL_SCENARIO.color


class TestScenarioSet:
    def test_default_scenarios(self):
        ss = ScenarioSet()
        assert len(ss.scenarios) == 3

    def test_custom_scenarios(self):
        ss = ScenarioSet([BEAR_SCENARIO, BASE_SCENARIO])
        assert len(ss.scenarios) == 2

    def test_run_returns_scenario_results(self, portfolio: Portfolio):
        ss = ScenarioSet([BASE_SCENARIO])
        results = ss.run(portfolio, n_simulations=100)
        assert isinstance(results, ScenarioResults)

    def test_run_all_three_scenarios(self, portfolio: Portfolio):
        ss = ScenarioSet([BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO])
        results = ss.run(portfolio, n_simulations=200)
        scenario_names = [r.scenario.name for r in results]
        assert "Bear" in scenario_names
        assert "Base" in scenario_names
        assert "Bull" in scenario_names

    def test_apply_scenario_adjusts_exit_period(self, portfolio: Portfolio):
        ss = ScenarioSet()
        adjusted = ss._apply_scenario(portfolio._companies, BEAR_SCENARIO)
        original_exits = [c.expected_exit_period for c in portfolio._companies]
        adjusted_exits = [c.expected_exit_period for c in adjusted]
        for orig, adj in zip(original_exits, adjusted_exits):
            # Bear scenario delays exits by 6
            assert adj >= orig  # always delayed, not before original

    def test_apply_scenario_never_exit_before_entry(self, portfolio: Portfolio):
        ss = ScenarioSet()
        # Even a large negative timing_adjustment shouldn't make exit before entry
        early_exit_scenario = Scenario(
            "EarlyExit", "test", exit_timing_adjustment=-100
        )
        adjusted = ss._apply_scenario(portfolio._companies, early_exit_scenario)
        for company in adjusted:
            assert company.expected_exit_period > company.entry_period


class TestScenarioResults:
    @pytest.fixture
    def scenario_results(self, portfolio: Portfolio) -> ScenarioResults:
        ss = ScenarioSet([BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO])
        return ss.run(portfolio, n_simulations=500)

    def test_compare_returns_dataframe(self, scenario_results: ScenarioResults):
        df = scenario_results.compare()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_compare_columns(self, scenario_results: ScenarioResults):
        df = scenario_results.compare()
        assert "scenario" in df.columns
        assert "moic_p50" in df.columns
        assert "irr_p50" in df.columns
        assert "loss_probability" in df.columns

    def test_bull_better_than_bear(self, scenario_results: ScenarioResults):
        df = scenario_results.compare()
        bull_moic = float(df[df["scenario"] == "Bull"]["moic_p50"].iloc[0])
        bear_moic = float(df[df["scenario"] == "Bear"]["moic_p50"].iloc[0])
        assert bull_moic > bear_moic

    def test_to_fan_chart_data(self, scenario_results: ScenarioResults):
        df = scenario_results.to_fan_chart_data("moic")
        assert isinstance(df, pd.DataFrame)
        assert "p10" in df.columns
        assert "p50" in df.columns
        assert "p90" in df.columns
        # p10 ≤ p50 ≤ p90 for each scenario
        for _, row in df.iterrows():
            assert row["p10"] <= row["p50"] <= row["p90"]

    def test_fan_chart_for_irr(self, scenario_results: ScenarioResults):
        df = scenario_results.to_fan_chart_data("irr")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3


class TestScenarioResult:
    def test_percentile_ordering(self, portfolio: Portfolio):
        ss = ScenarioSet([BASE_SCENARIO])
        results = ss.run(portfolio, n_simulations=500)
        for result in results:
            percs = result.moic_percentiles
            assert percs["p10"] <= percs["p25"] <= percs["p50"] <= percs["p75"] <= percs["p90"]

    def test_probabilities_in_range(self, portfolio: Portfolio):
        ss = ScenarioSet([BASE_SCENARIO])
        results = ss.run(portfolio, n_simulations=500)
        for result in results:
            assert 0 <= result.loss_probability <= 1
            assert 0 <= result.home_run_probability <= 1


class TestMonteCarloSimulator:
    def test_run_returns_dict(self, portfolio: Portfolio):
        sim = MonteCarloSimulator(portfolio, n_simulations=500, seed=42)
        results = sim.run()
        assert "raw_moics" in results
        assert "raw_irrs" in results

    def test_compute_var_95(self, portfolio: Portfolio):
        sim = MonteCarloSimulator(portfolio, n_simulations=1000, seed=42)
        sim.run()
        var = sim.compute_var(confidence=0.95, metric="moic")
        # VaR is the worst 5% outcome
        raw_moics = sim._results["raw_moics"]
        assert var == pytest.approx(float(np.percentile(raw_moics, 5)), rel=1e-6)

    def test_compute_var_runs_simulation_if_needed(self, portfolio: Portfolio):
        sim = MonteCarloSimulator(portfolio, n_simulations=500, seed=42)
        # Should run automatically
        var = sim.compute_var(confidence=0.90)
        assert isinstance(var, float)

    def test_expected_shortfall_lte_var(self, portfolio: Portfolio):
        sim = MonteCarloSimulator(portfolio, n_simulations=1000, seed=42)
        sim.run()
        var = sim.compute_var(confidence=0.95, metric="moic")
        es = sim.expected_shortfall(confidence=0.95, metric="moic")
        assert es <= var + 1e-9  # ES ≤ VaR by definition


class TestSensitivityAnalysis:
    def test_sweep_returns_dataframe(self, portfolio: Portfolio):
        sa = SensitivityAnalysis(portfolio, n_simulations=200, seed=42)
        df = sa.sweep("return_adjustment", [0.5, 0.75, 1.0, 1.25, 1.5])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_sweep_columns(self, portfolio: Portfolio):
        sa = SensitivityAnalysis(portfolio, n_simulations=200, seed=42)
        df = sa.sweep("return_adjustment", [0.8, 1.0, 1.2], metric="moic_p50")
        assert "parameter" in df.columns
        assert "value" in df.columns
        assert "moic_p50" in df.columns

    def test_tornado_returns_dataframe(self, portfolio: Portfolio):
        sa = SensitivityAnalysis(portfolio, n_simulations=200, seed=42)
        df = sa.tornado(
            ["return_adjustment", "loss_rate_delta"],
            metric="moic_p50",
            pct_change=0.20,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_tornado_sorted_by_swing(self, portfolio: Portfolio):
        sa = SensitivityAnalysis(portfolio, n_simulations=300, seed=42)
        df = sa.tornado(
            ["return_adjustment", "loss_rate_delta", "exit_multiple_adjustment"],
            metric="moic_p50",
        )
        swings = df["swing"].tolist()
        assert swings == sorted(swings, reverse=True)

    def test_tornado_columns(self, portfolio: Portfolio):
        sa = SensitivityAnalysis(portfolio, n_simulations=200, seed=42)
        df = sa.tornado(["return_adjustment"], metric="moic_p50")
        assert "parameter" in df.columns
        assert "low_value" in df.columns
        assert "high_value" in df.columns
        assert "swing" in df.columns
