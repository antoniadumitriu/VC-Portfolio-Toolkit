"""
test_regressions.py: Tests pinning down previously fixed modeling bugs.

Each test here failed before the corresponding fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from vc_portfolio.jcurve import CompanyTimeline, JCurve
from vc_portfolio.metrics import calc_irr
from vc_portfolio.portfolio import (
    STAGE_LOSS_RATES,
    Company,
    Portfolio,
    draw_exit_multiples,
    portfolio_irrs,
)
from vc_portfolio.scenarios import (
    BASE_SCENARIO,
    BULL_SCENARIO,
    MonteCarloSimulator,
    Scenario,
    ScenarioSet,
    SensitivityAnalysis,
)


def _company(name: str, entry: float, exit_: float, stage: str = "seed") -> Company:
    return Company(name, 1_000_000, stage, "saas", 5_000_000, entry, exit_, 0.15)  # type: ignore[arg-type]


@pytest.fixture
def small_portfolio() -> Portfolio:
    return Portfolio(seed=7).add_companies(
        [
            _company("A", 0, 6, "seed"),
            _company("B", 1, 7, "series_a"),
            _company("C", 2, 9, "pre_seed"),
            _company("D", 3, 8, "series_b"),
        ]
    )


# ---------------------------------------------------------------------------
# 1. Scenario loss-rate adjustment must actually change the loss rate
# ---------------------------------------------------------------------------

class TestScenarioLossRate:
    def test_draw_applies_loss_rate_adjustment(self) -> None:
        rng = np.random.default_rng(0)
        m = draw_exit_multiples(rng, "seed", 50_000, loss_rate_adjustment=0.20)
        assert np.mean(m == 0) == pytest.approx(STAGE_LOSS_RATES["seed"] + 0.20, abs=0.01)

    def test_loss_rate_is_clipped(self) -> None:
        rng = np.random.default_rng(0)
        m = draw_exit_multiples(rng, "pre_seed", 10_000, loss_rate_adjustment=5.0)
        assert 0.9 < np.mean(m == 0) < 1.0

    def test_scenario_loss_adjustment_changes_results(self, small_portfolio: Portfolio) -> None:
        harsh = Scenario("Harsh", "more losses only", loss_rate_adjustment=0.30)
        results = list(ScenarioSet([BASE_SCENARIO, harsh]).run(small_portfolio, n_simulations=4_000))
        base, harsh_res = results
        assert harsh_res.loss_probability > base.loss_probability + 0.05


# ---------------------------------------------------------------------------
# 2. Scenario exit timing is in quarters; company periods are in years
# ---------------------------------------------------------------------------

class TestScenarioTimingUnits:
    def test_four_quarters_is_one_year(self, small_portfolio: Portfolio) -> None:
        shift = Scenario("Late", "one year later", exit_timing_adjustment=4)
        adjusted = ScenarioSet()._apply_scenario(small_portfolio._companies, shift)
        for orig, adj in zip(small_portfolio._companies, adjusted, strict=True):
            assert adj.expected_exit_period == pytest.approx(orig.expected_exit_period + 1)

    def test_bull_irr_is_plausible(self, small_portfolio: Portfolio) -> None:
        results = list(ScenarioSet([BULL_SCENARIO]).run(small_portfolio, n_simulations=4_000))
        # Bull pulls exits one year earlier, not four. Median IRR stays well below 40%.
        assert results[0].irr_percentiles["p50"] < 0.40


# ---------------------------------------------------------------------------
# 3. Portfolio IRR comes from actual cash flow timing
# ---------------------------------------------------------------------------

class TestPortfolioIrr:
    def test_matches_scalar_irr(self) -> None:
        rng = np.random.default_rng(1)
        invest = np.array([2e6, 3e6, 1e6])
        entry = np.array([0.0, 2.0, 4.0])
        exit_ = np.array([3.0, 6.0, 9.5])
        values = rng.uniform(0, 5, size=(50, 3)) * invest
        values[0] = 0.0  # total loss
        irrs = portfolio_irrs(invest, entry, exit_, values)

        assert irrs[0] == -1.0
        for i in range(1, 50):
            times = np.concatenate([entry, exit_])
            flows = np.concatenate([-invest, values[i]])
            expected = calc_irr(flows, periods=times)
            assert irrs[i] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. J-curve: annualized rolling IRR and a real dip
# ---------------------------------------------------------------------------

class TestJCurve:
    def test_rolling_irr_is_annualized(self) -> None:
        jc = JCurve(n_periods=40, fee_rate=0.0).add_company(
            CompanyTimeline(0, 20, 10e6, 2.0, "Doubler")
        )
        df = jc.project()
        # 2x over 20 quarters (5 years) is about 14.9% a year
        assert df.loc[20, "rolling_irr"] == pytest.approx(2 ** (1 / 5) - 1, abs=1e-4)

    def test_scurve_starts_at_cost(self) -> None:
        jc = JCurve()
        assert jc._compute_scurve_weight(5, 5, 15) == pytest.approx(0.0)

    def test_curve_dips_below_paid_in(self) -> None:
        jc = JCurve(n_periods=40, committed_capital=100e6).add_companies(
            [
                CompanyTimeline(0, 24, 10e6, 3.0, "A"),
                CompanyTimeline(4, 28, 10e6, 2.0, "B"),
            ]
        )
        df = jc.project()
        shape = jc.get_jcurve_shape()
        assert df.loc[0, "tvpi"] < 1.0
        assert shape["trough_nav"] < 0
        assert shape["trough_period"] > 0
        assert shape["breakeven_period"] > shape["trough_period"]


    def test_follow_ons_not_marked_before_they_happen(self) -> None:
        tl = CompanyTimeline(0, 40, 1e6, 10.0, "Star", follow_on_investments={20: 9e6})
        jc = JCurve(n_periods=40, fee_rate=0.0).add_company(tl)
        nav_before = jc._compute_company_nav(tl, 19)
        # At most 10x the $1M actually invested, not 10x the future $10M total
        assert nav_before <= 10e6

    def test_losers_written_down_early(self) -> None:
        tl = CompanyTimeline(0, 20, 5e6, 0.0, "Loss")
        jc = JCurve()
        assert jc._compute_company_nav(tl, 5) == pytest.approx(2.5e6)
        assert jc._compute_company_nav(tl, 10) == pytest.approx(0.0)

    def test_example_style_portfolio_has_a_dip(self) -> None:
        jc = JCurve(n_periods=40, committed_capital=100e6).add_companies(
            [
                CompanyTimeline(0, 28, 8e6, 8.0, "Star", follow_on_investments={8: 3e6, 16: 4e6}),
                CompanyTimeline(2, 24, 5e6, 0.0, "Loss"),
                CompanyTimeline(4, 22, 6e6, 3.5, "Good"),
                CompanyTimeline(10, 30, 5e6, 1.2, "Zombie"),
            ]
        )
        shape = jc.get_jcurve_shape()
        assert shape["trough_period"] >= 4
        assert shape["breakeven_period"] > shape["trough_period"]


# ---------------------------------------------------------------------------
# 5. Sensitivity analysis parameters are distinct and real
# ---------------------------------------------------------------------------

class TestSensitivity:
    def test_unknown_parameter_raises(self, small_portfolio: Portfolio) -> None:
        sa = SensitivityAnalysis(small_portfolio, n_simulations=200)
        with pytest.raises(ValueError):
            sa.sweep("exit_multiple_adjustment", [1.0])

    def test_loss_rate_multiplier_raises_losses(self, small_portfolio: Portfolio) -> None:
        sa = SensitivityAnalysis(small_portfolio, n_simulations=3_000)
        df = sa.sweep("loss_rate_delta", [0.8, 1.0, 1.2], metric="loss_probability")
        vals = df["loss_probability"].tolist()
        assert vals[0] < vals[1] < vals[2]

    def test_holding_period_moves_irr_not_moic(self, small_portfolio: Portfolio) -> None:
        sa = SensitivityAnalysis(small_portfolio, n_simulations=2_000)
        moic = sa.tornado(["holding_period"], metric="moic_p50")
        irr = sa.tornado(["holding_period"], metric="irr_p50")
        assert moic["swing"].iloc[0] == pytest.approx(0.0)
        assert irr["swing"].iloc[0] > 0.005
        # Longer holds lower the IRR, and the base sits between the two
        row = irr.iloc[0]
        assert row["high_value"] < row["base_value"] < row["low_value"]


# ---------------------------------------------------------------------------
# 6. MonteCarloSimulator honors its own seed
# ---------------------------------------------------------------------------

def test_simulator_seed_is_reproducible(small_portfolio: Portfolio) -> None:
    a = MonteCarloSimulator(small_portfolio, n_simulations=500, seed=3).run()
    b = MonteCarloSimulator(small_portfolio, n_simulations=500, seed=3).run()
    np.testing.assert_array_equal(a["raw_moics"], b["raw_moics"])
