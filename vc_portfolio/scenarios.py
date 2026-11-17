"""
scenarios.py — Orchestration layer for scenario analysis and Monte Carlo simulation.

Depends on: metrics.py, fund.py, portfolio.py, jcurve.py
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from vc_portfolio.fund import Fund, FundConfig
from vc_portfolio.portfolio import (
    Company,
    Portfolio,
    simulate_portfolio,
    summarize_simulation,
)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """
    A named market scenario with adjustment factors.

    These multipliers are applied to portfolio simulation parameters
    to model different market environments.
    """

    name: str
    description: str
    portfolio_return_adjustment: float = 1.0  # multiplier on exit multiples
    exit_timing_adjustment: int = 0  # quarters added to exit dates (converted to years)
    loss_rate_adjustment: float = 0.0  # additive change to stage loss rates
    color: str = "#636EFA"


# Pre-built canonical scenarios
BEAR_SCENARIO = Scenario(
    name="Bear",
    description="Prolonged downturn: compressed multiples, delayed exits, higher losses",
    portfolio_return_adjustment=0.65,
    exit_timing_adjustment=6,
    loss_rate_adjustment=0.20,
    color="#EF553B",
)

BASE_SCENARIO = Scenario(
    name="Base",
    description="Historical median VC market conditions",
    portfolio_return_adjustment=1.0,
    exit_timing_adjustment=0,
    loss_rate_adjustment=0.0,
    color="#636EFA",
)

BULL_SCENARIO = Scenario(
    name="Bull",
    description="Favorable exit environment: expanded multiples, faster exits",
    portfolio_return_adjustment=1.5,
    exit_timing_adjustment=-4,
    loss_rate_adjustment=-0.15,
    color="#00CC96",
)


# ---------------------------------------------------------------------------
# Scenario Results
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Results for a single scenario simulation run."""

    scenario: Scenario
    moics: npt.NDArray[np.float64]
    irrs: npt.NDArray[np.float64]

    @property
    def moic_percentiles(self) -> dict[str, float]:
        return {
            f"p{p}": float(np.percentile(self.moics, p))
            for p in [10, 25, 50, 75, 90]
        }

    @property
    def irr_percentiles(self) -> dict[str, float]:
        return {
            f"p{p}": float(np.percentile(self.irrs, p))
            for p in [10, 25, 50, 75, 90]
        }

    @property
    def loss_probability(self) -> float:
        return float(np.mean(self.moics < 1.0))

    @property
    def home_run_probability(self) -> float:
        return float(np.mean(self.moics > 3.0))


class ScenarioResults:
    """Aggregated results across multiple scenarios."""

    def __init__(self, results: list[ScenarioResult]) -> None:
        self._results = results

    def __iter__(self):
        return iter(self._results)

    def compare(self) -> pd.DataFrame:
        """
        Return a comparison DataFrame across scenarios.

        Rows are scenarios; columns are key metrics at each percentile.
        """
        rows = []
        for result in self._results:
            row = {
                "scenario": result.scenario.name,
                "description": result.scenario.description,
                "moic_p10": result.moic_percentiles["p10"],
                "moic_p25": result.moic_percentiles["p25"],
                "moic_p50": result.moic_percentiles["p50"],
                "moic_p75": result.moic_percentiles["p75"],
                "moic_p90": result.moic_percentiles["p90"],
                "irr_p10": result.irr_percentiles["p10"],
                "irr_p25": result.irr_percentiles["p25"],
                "irr_p50": result.irr_percentiles["p50"],
                "irr_p75": result.irr_percentiles["p75"],
                "irr_p90": result.irr_percentiles["p90"],
                "loss_probability": result.loss_probability,
                "home_run_probability": result.home_run_probability,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def to_fan_chart_data(
        self,
        metric: Literal["moic", "irr"] = "moic",
    ) -> pd.DataFrame:
        """
        Return fan chart data for Cambridge Associates-style p10–p90 bands.

        Parameters
        ----------
        metric:
            'moic' or 'irr'

        Returns
        -------
        DataFrame with columns: scenario, p10, p25, p50, p75, p90
        """
        rows = []
        for result in self._results:
            percs = result.moic_percentiles if metric == "moic" else result.irr_percentiles
            rows.append(
                {
                    "scenario": result.scenario.name,
                    "color": result.scenario.color,
                    "p10": percs["p10"],
                    "p25": percs["p25"],
                    "p50": percs["p50"],
                    "p75": percs["p75"],
                    "p90": percs["p90"],
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ScenarioSet
# ---------------------------------------------------------------------------

class ScenarioSet:
    """
    Container for multiple scenarios that runs them against a portfolio.

    Usage:
        results = ScenarioSet([BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO]).run(
            portfolio=my_portfolio,
            n_simulations=10_000,
        )
    """

    def __init__(
        self,
        scenarios: Optional[list[Scenario]] = None,
    ) -> None:
        self.scenarios = scenarios or [BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO]

    def run(
        self,
        portfolio: Portfolio,
        n_simulations: int = 10_000,
        base_seed: int = 42,
    ) -> ScenarioResults:
        """
        Run all scenarios against the portfolio.

        Parameters
        ----------
        portfolio:
            Portfolio to simulate.
        n_simulations:
            Number of Monte Carlo draws per scenario.
        base_seed:
            Random seed, reused for every scenario.

        Returns
        -------
        ScenarioResults
        """
        all_results = []

        for scenario in self.scenarios:
            adjusted_companies = self._apply_scenario(portfolio._companies, scenario)
            # Same seed for every scenario (common random numbers), so differences
            # between scenarios come from the scenario, not from sampling noise.
            moics, irrs = simulate_portfolio(
                adjusted_companies,
                n_simulations,
                np.random.default_rng(base_seed),
                return_multiplier=scenario.portfolio_return_adjustment,
                loss_rate_adjustment=scenario.loss_rate_adjustment,
            )
            all_results.append(
                ScenarioResult(scenario=scenario, moics=moics, irrs=irrs)
            )

        return ScenarioResults(all_results)

    def _apply_scenario(
        self,
        companies: list[Company],
        scenario: Scenario,
    ) -> list[Company]:
        """
        Return a copy of companies with the scenario's exit timing applied.

        ``exit_timing_adjustment`` is in quarters; company periods are in years.
        Exits never move earlier than one quarter after entry. Return and
        loss-rate adjustments are applied during simulation.
        """
        shift_years = scenario.exit_timing_adjustment / 4.0
        return [
            replace(
                company,
                expected_exit_period=max(
                    company.entry_period + 0.25,
                    company.expected_exit_period + shift_years,
                ),
            )
            for company in companies
        ]


# ---------------------------------------------------------------------------
# MonteCarloSimulator
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """
    Standalone Monte Carlo simulator with VaR and expected shortfall.

    Usage:
        simulator = MonteCarloSimulator(portfolio, n_simulations=50_000)
        results = simulator.run()
        var_95 = simulator.compute_var(confidence=0.95, metric="moic")
    """

    def __init__(
        self,
        portfolio: Portfolio,
        n_simulations: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.portfolio = portfolio
        self.n_simulations = n_simulations
        self.seed = seed
        self._results: Optional[dict[str, npt.NDArray[np.float64]]] = None

    def run(self) -> dict[str, npt.NDArray[np.float64]]:
        """Run Monte Carlo simulation and cache results."""
        moics, irrs = simulate_portfolio(
            self.portfolio._companies,
            self.n_simulations,
            np.random.default_rng(self.seed),
        )
        raw = summarize_simulation(moics, irrs)
        self._results = raw  # type: ignore[assignment]
        return raw  # type: ignore[return-value]

    def compute_var(
        self,
        confidence: float = 0.95,
        metric: Literal["moic", "irr"] = "moic",
    ) -> float:
        """
        Compute Value at Risk at the given confidence level.

        VaR = worst outcome at (1 - confidence) percentile.
        E.g., 95% VaR = 5th percentile outcome.

        Parameters
        ----------
        confidence:
            Confidence level (e.g. 0.95 = 95%).
        metric:
            'moic' or 'irr'

        Returns
        -------
        float
            VaR threshold value.
        """
        if self._results is None:
            self.run()

        arr = self._results[f"raw_{metric}s"]
        percentile = (1 - confidence) * 100
        return float(np.percentile(arr, percentile))

    def expected_shortfall(
        self,
        confidence: float = 0.95,
        metric: Literal["moic", "irr"] = "moic",
    ) -> float:
        """
        Compute Expected Shortfall (CVaR) at given confidence level.

        Average outcome in the worst (1 - confidence) fraction.
        """
        if self._results is None:
            self.run()

        arr = self._results[f"raw_{metric}s"]
        var_threshold = self.compute_var(confidence, metric)
        tail = arr[arr <= var_threshold]
        return float(tail.mean()) if len(tail) > 0 else float(var_threshold)


# ---------------------------------------------------------------------------
# SensitivityAnalysis
# ---------------------------------------------------------------------------

SENSITIVITY_PARAMETERS: dict[str, str] = {
    "return_adjustment": "return_multiplier",
    "loss_rate_delta": "loss_rate_multiplier",
    "holding_period": "holding_period_multiplier",
}


def _extract_metric(result: dict[str, object], metric: str) -> float:
    """Read a metric such as 'moic_p50', 'irr_p25' or 'loss_probability'."""
    if metric not in result or metric.startswith("raw_"):
        raise ValueError(f"Unknown metric '{metric}'")
    return float(result[metric])  # type: ignore[arg-type]


class SensitivityAnalysis:
    """
    One-way and tornado sensitivity analysis for portfolio return drivers.

    Parameters are multipliers around a base value of 1.0:
        'return_adjustment': scales every exit multiple
        'loss_rate_delta':   scales every stage loss rate
        'holding_period':    scales every holding period (affects IRR, not MOIC)

    All runs share one seed, so differences reflect the parameter, not noise.

    Usage:
        sa = SensitivityAnalysis(portfolio, n_simulations=2_000)
        df = sa.sweep("loss_rate_delta", [0.8, 1.0, 1.2], "loss_probability")
        tornado_df = sa.tornado(["return_adjustment", "holding_period"], "irr_p50")
    """

    def __init__(
        self,
        portfolio: Portfolio,
        n_simulations: int = 2_000,
        seed: int = 42,
    ) -> None:
        self.portfolio = portfolio
        self.n_simulations = n_simulations
        self.seed = seed

    def sweep(
        self,
        parameter: str,
        values: list[float],
        metric: str = "moic_p50",
    ) -> pd.DataFrame:
        """
        One-way sensitivity: vary a single parameter across a range of values.

        Returns a DataFrame with columns: parameter, value, <metric>.
        """
        rows = []
        for val in values:
            result = self._simulate_with_adjustment(parameter, val)
            rows.append(
                {"parameter": parameter, "value": val, metric: _extract_metric(result, metric)}
            )
        return pd.DataFrame(rows)

    def tornado(
        self,
        parameters: list[str],
        metric: str = "moic_p50",
        pct_change: float = 0.20,
    ) -> pd.DataFrame:
        """
        Tornado chart data: metric at (1 - pct_change) and (1 + pct_change)
        for each parameter.

        Returns a DataFrame with columns: parameter, low_value, high_value,
        swing, sorted by swing (largest first).
        """
        rows = []
        for param in parameters:
            low_val = _extract_metric(self._simulate_with_adjustment(param, 1.0 - pct_change), metric)
            high_val = _extract_metric(self._simulate_with_adjustment(param, 1.0 + pct_change), metric)
            rows.append(
                {
                    "parameter": param,
                    "low_value": low_val,
                    "high_value": high_val,
                    "swing": abs(high_val - low_val),
                }
            )

        df = pd.DataFrame(rows)
        return df.sort_values("swing", ascending=False).reset_index(drop=True)

    def _simulate_with_adjustment(
        self,
        parameter: str,
        value: float,
    ) -> dict[str, object]:
        """Run the portfolio simulation with one parameter multiplier applied."""
        if parameter not in SENSITIVITY_PARAMETERS:
            raise ValueError(
                f"Unknown parameter '{parameter}'. "
                f"Choose from: {', '.join(SENSITIVITY_PARAMETERS)}"
            )
        moics, irrs = simulate_portfolio(
            self.portfolio._companies,
            self.n_simulations,
            np.random.default_rng(self.seed),
            **{SENSITIVITY_PARAMETERS[parameter]: value},
        )
        return summarize_simulation(moics, irrs)
