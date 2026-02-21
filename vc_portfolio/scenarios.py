"""
scenarios.py — Orchestration layer for scenario analysis and Monte Carlo simulation.

Depends on: metrics.py, fund.py, portfolio.py, jcurve.py
"""
from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from vc_portfolio.fund import Fund, FundConfig
from vc_portfolio.portfolio import Company, Portfolio


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
    exit_timing_adjustment: int = 0  # quarters added to exit periods
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
# Simulation worker (module-level for ProcessPoolExecutor compatibility)
# ---------------------------------------------------------------------------

def _run_single_simulation(
    args: tuple[list[Company], int, float, int],
) -> tuple[float, float]:
    """
    Worker function for parallel Monte Carlo simulation.

    Parameters
    ----------
    args:
        (companies, seed, return_adjustment, seed_offset)

    Returns
    -------
    (moic, irr) tuple
    """
    companies, base_seed, return_adj, sim_idx = args

    rng = np.random.default_rng(base_seed + sim_idx)
    investments = np.array([c.investment_amount for c in companies])
    total_invested = float(investments.sum())

    if total_invested <= 0:
        return 0.0, -1.0

    # Adjust holding periods (companies passed in with already-adjusted exit periods)
    holding_periods = np.array(
        [c.expected_exit_period - c.entry_period for c in companies],
        dtype=np.float64,
    )

    total_value = 0.0
    weighted_period = 0.0

    for j, company in enumerate(companies):
        from vc_portfolio.portfolio import STAGE_LOSS_RATES
        p_loss = STAGE_LOSS_RATES.get(company.stage, 0.40)

        # Draw single exit multiple
        u = float(rng.uniform())
        p_zombie = min(0.20, (1.0 - p_loss) * 0.25)
        p_lognorm = (1.0 - p_loss) * 0.65

        if u < p_loss:
            raw_multiple = 0.0
        elif u < p_loss + p_zombie:
            raw_multiple = float(rng.uniform(0.5, 1.5))
        elif u < p_loss + p_zombie + p_lognorm:
            raw_multiple = float(rng.lognormal(mean=0.693, sigma=0.85))
        else:
            pareto_u = float(rng.uniform())
            raw_multiple = 2.0 / (1 - pareto_u) ** (1 / 1.5)

        # Apply scenario return adjustment
        adjusted_multiple = raw_multiple * return_adj
        exit_value = investments[j] * adjusted_multiple
        total_value += exit_value
        weighted_period += investments[j] * holding_periods[j]

    moic = total_value / total_invested if total_invested > 0 else 0.0
    avg_hold = weighted_period / total_invested if total_invested > 0 else 5.0

    if avg_hold > 0 and moic > 0:
        irr = moic ** (1.0 / avg_hold) - 1
    else:
        irr = -1.0

    return moic, irr


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
            Base random seed (each simulation uses base_seed + sim_idx).

        Returns
        -------
        ScenarioResults
        """
        all_results = []

        for scenario in self.scenarios:
            # Apply scenario adjustments to companies
            adjusted_companies = self._apply_scenario(portfolio._companies, scenario)

            moics = np.zeros(n_simulations, dtype=np.float64)
            irrs = np.zeros(n_simulations, dtype=np.float64)

            # Build args for worker
            args_list = [
                (adjusted_companies, base_seed, scenario.portfolio_return_adjustment, i)
                for i in range(n_simulations)
            ]

            # Use ProcessPoolExecutor for parallelism
            # Fall back to sequential if n_simulations is small
            if n_simulations >= 1000:
                try:
                    with ProcessPoolExecutor() as executor:
                        futures = {
                            executor.submit(_run_single_simulation, args): i
                            for i, args in enumerate(args_list)
                        }
                        for future in as_completed(futures):
                            i = futures[future]
                            moic, irr = future.result()
                            moics[i] = moic
                            irrs[i] = irr
                except Exception:
                    # Fallback to sequential on error
                    for i, args in enumerate(args_list):
                        moic, irr = _run_single_simulation(args)
                        moics[i] = moic
                        irrs[i] = irr
            else:
                for i, args in enumerate(args_list):
                    moic, irr = _run_single_simulation(args)
                    moics[i] = moic
                    irrs[i] = irr

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
        Return a modified copy of companies with scenario adjustments applied.

        Adjustments:
        - exit_timing_adjustment: shifts expected_exit_period
        - loss_rate_adjustment: adjusts STAGE_LOSS_RATES (via Company.stage metadata)
        """
        adjusted = []
        for company in companies:
            new_exit = max(
                company.entry_period + 1,
                company.expected_exit_period + scenario.exit_timing_adjustment,
            )
            # Create modified company with adjusted exit period
            adj_company = Company(
                name=company.name,
                investment_amount=company.investment_amount,
                stage=company.stage,
                sector=company.sector,
                entry_valuation=company.entry_valuation,
                entry_period=company.entry_period,
                expected_exit_period=new_exit,
                ownership_pct=company.ownership_pct,
                reserved_capital=company.reserved_capital,
            )
            adjusted.append(adj_company)
        return adjusted


# ---------------------------------------------------------------------------
# MonteCarloSimulator
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """
    Standalone Monte Carlo simulator with parallel execution and VaR computation.

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
        raw = self.portfolio.simulate_exits(n_simulations=self.n_simulations)
        self._results = raw
        return raw

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

class SensitivityAnalysis:
    """
    One-way and tornado sensitivity analysis for portfolio return drivers.

    Usage:
        sa = SensitivityAnalysis(portfolio, n_simulations=1000)
        df = sa.sweep("fee_rate", [0.01, 0.015, 0.02, 0.025], "moic_p50")
        tornado_df = sa.tornado(["fee_rate", "carry_rate"], "moic_p50", pct_change=0.20)
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

        Currently supports sweeping over simulated metrics by injecting
        a return multiplier (conceptual mapping for demonstration).

        Parameters
        ----------
        parameter:
            Parameter name (e.g. 'return_adjustment', 'loss_rate_delta').
        values:
            List of parameter values to test.
        metric:
            Which metric to report ('moic_p50', 'moic_p25', etc.)

        Returns
        -------
        DataFrame with columns: parameter_value, metric_value
        """
        rows = []
        for val in values:
            rng = np.random.default_rng(self.seed)
            sim_results = self._simulate_with_adjustment(parameter, val)
            raw_moics = sim_results["raw_moics"]

            if metric == "moic_p50":
                metric_val = float(np.percentile(raw_moics, 50))
            elif metric == "moic_p25":
                metric_val = float(np.percentile(raw_moics, 25))
            elif metric == "moic_p75":
                metric_val = float(np.percentile(raw_moics, 75))
            elif metric == "moic_p10":
                metric_val = float(np.percentile(raw_moics, 10))
            elif metric == "moic_p90":
                metric_val = float(np.percentile(raw_moics, 90))
            elif metric == "loss_probability":
                metric_val = float(np.mean(raw_moics < 1.0))
            else:
                metric_val = float(np.percentile(raw_moics, 50))

            rows.append({"parameter": parameter, "value": val, metric: metric_val})

        return pd.DataFrame(rows)

    def tornado(
        self,
        parameters: list[str],
        metric: str = "moic_p50",
        pct_change: float = 0.20,
    ) -> pd.DataFrame:
        """
        Tornado chart data: show metric sensitivity to ±pct_change in each parameter.

        Parameters
        ----------
        parameters:
            List of parameter names to analyze.
        metric:
            Metric to evaluate.
        pct_change:
            Fractional change to apply (e.g. 0.20 = ±20%).

        Returns
        -------
        DataFrame with columns: parameter, low_value, high_value, swing
            sorted by swing (largest first)
        """
        rows = []

        for param in parameters:
            low_result = self._simulate_with_adjustment(param, 1.0 - pct_change)
            high_result = self._simulate_with_adjustment(param, 1.0 + pct_change)

            def _extract(result: dict, m: str) -> float:
                moics = result["raw_moics"]
                if m == "moic_p50":
                    return float(np.percentile(moics, 50))
                elif m == "loss_probability":
                    return float(np.mean(moics < 1.0))
                return float(np.percentile(moics, 50))

            low_val = _extract(low_result, metric)
            high_val = _extract(high_result, metric)
            swing = abs(high_val - low_val)

            rows.append(
                {
                    "parameter": param,
                    "low_value": low_val,
                    "high_value": high_val,
                    "swing": swing,
                }
            )

        df = pd.DataFrame(rows)
        return df.sort_values("swing", ascending=False).reset_index(drop=True)

    def _simulate_with_adjustment(
        self,
        parameter: str,
        value: float,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Run portfolio simulation with a parameter adjustment.

        Maps parameter names to simulation adjustments.
        """
        # For simplicity, treat most parameters as return multipliers
        # This can be extended to fund-specific parameters
        adjusted_portfolio = Portfolio(seed=self.seed)
        for company in self.portfolio._companies:
            adjusted_portfolio.add_company(company)

        # Override the portfolio's RNG and run with multiplied outcomes
        # We simulate using raw portfolio and apply multiplier post-hoc
        raw_results = adjusted_portfolio.simulate_exits(n_simulations=self.n_simulations)

        if parameter in ("return_adjustment", "exit_multiple_adjustment"):
            raw_results["raw_moics"] = raw_results["raw_moics"] * value
        elif parameter == "loss_rate_delta":
            # Higher loss rate → more zeros → lower moic
            mask = np.random.default_rng(self.seed).uniform(
                size=self.n_simulations
            ) < abs(value - 1.0)
            if value > 1.0:
                raw_results["raw_moics"] = np.where(
                    mask, 0.0, raw_results["raw_moics"]
                )

        return raw_results
