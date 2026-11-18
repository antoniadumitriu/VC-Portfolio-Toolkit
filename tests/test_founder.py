"""
Tests for founder.py.

Round and waterfall cases use the figures of the worked examples in Dave
Lishego, The Founder's Guide to VC Finance (2017), cited by example number.
Where the book's running text and its cap table disagree, the assertion follows
the arithmetic.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from vc_portfolio.founder import (
    FOUNDERS,
    CapTable,
    Convertible,
    ExitAssumptions,
    RoundInputError,
    RoundTerms,
    ShareClass,
    _Engine,
    add_later_rounds,
    adjusted_conversion_price,
    breakpoints,
    exit_for_payout,
    founder_outcome,
    minimum_pre_money,
    negotiation_levers,
    ownership_grid,
    price_round,
    waterfall,
)

M = 1_000_000.0


def close(value: float, expected: float, tolerance: float = 1.0) -> bool:
    return abs(value - expected) <= tolerance


@pytest.fixture
def series_a_company() -> CapTable:
    return CapTable(founders=1_000_000, other_common=12_000, available_options=99_111)


def simple_preferred(participation: str = "none", **kwargs: object) -> CapTable:
    """$5M for 25%, as in Examples 7.1 to 7.3."""
    series_a = ShareClass(
        "Series A", 500_000, 10.0, 5 * M, participation=participation, **kwargs
    )
    return CapTable(founders=1_500_000, preferred=(series_a,))


class TestPricing:
    def test_plain_round_example_5_1(self, series_a_company: CapTable) -> None:
        r = price_round(series_a_company, RoundTerms(5 * M, 15 * M))
        assert r.price == pytest.approx(13.5, abs=0.001)
        assert close(r.investor_shares, 370_370)
        assert r.investor_ownership == pytest.approx(0.25)
        assert r.post_money == pytest.approx(20 * M)

    def test_post_money_pool_example_6_1(self, series_a_company: CapTable) -> None:
        t = RoundTerms(5 * M, 15 * M, option_pool_target=0.15, pool_in_pre_money=False)
        r = price_round(series_a_company, t)
        assert close(r.after.fully_diluted, 1_626_318)
        assert close(r.after.available_options, 243_948)

    def test_pre_money_pool_example_6_1(self, series_a_company: CapTable) -> None:
        r = price_round(
            series_a_company, RoundTerms(5 * M, 15 * M, option_pool_target=0.15)
        )
        assert close(r.after.fully_diluted, 1_686_667)
        assert close(r.after.available_options, 253_000)
        assert close(r.investor_shares, 421_667)

    @pytest.mark.parametrize(
        ("in_pre_money", "total"), [(False, 14_156_250), (True, 14_981_273)]
    )
    def test_discount_note_example_6_2(self, in_pre_money: bool, total: int) -> None:
        note = Convertible(
            "Note", 500_000, "note", discount=0.2, interest_rate=0.06,
            years_outstanding=1,
        )
        t = RoundTerms(
            M, 4 * M, convertibles=(note,), convertibles_in_pre_money=in_pre_money
        )
        assert close(price_round(CapTable(10 * M), t).after.fully_diluted, total)

    @pytest.mark.parametrize(
        ("in_pre_money", "total"), [(False, 8_550_000), (True, 8_812_500)]
    )
    def test_capped_safes_example_6_3(self, in_pre_money: bool, total: int) -> None:
        safes = (
            Convertible("Pre-seed", 100_000, "pre_money_safe", cap=2 * M, discount=0.2),
            Convertible("Seed", 500_000, "pre_money_safe", cap=4 * M, discount=0.2),
        )
        t = RoundTerms(
            3 * M, 12 * M, convertibles=safes, convertibles_in_pre_money=in_pre_money
        )
        r = price_round(CapTable(5 * M, available_options=M), t)
        assert close(r.after.fully_diluted, total)
        assert close(r.conversions[0].shares, 300_000)
        assert close(r.conversions[1].shares, 750_000)
        assert {c.basis for c in r.conversions} == {"cap"}

    def test_pool_and_note_example_6_4(self) -> None:
        note = Convertible(
            "Note", 300_000, "note", discount=0.3, interest_rate=0.08,
            years_outstanding=547 / 365,
        )
        t = RoundTerms(3 * M, 7 * M, option_pool_target=0.12, convertibles=(note,))
        r = price_round(CapTable(4 * M), t)
        assert close(note.conversion_amount, 335_967)
        assert close(r.after.fully_diluted, 7_518_731)
        assert close(r.after.available_options, 902_248)
        assert close(r.conversions[0].shares, 360_864)
        assert r.founder_ownership == pytest.approx(0.532, abs=0.0005)

    def test_step_by_step_example_1(self) -> None:
        note = Convertible(
            "Note", 200_000, "note", cap=5 * M, discount=0.2, interest_rate=0.08,
            years_outstanding=189 / 365,
        )
        t = RoundTerms(
            2.25 * M, 3.75 * M, option_pool_target=0.15, convertibles=(note,),
            shadow_series=False,
        )
        r = price_round(CapTable(2 * M), t)
        assert close(r.after.fully_diluted, 4_633_842)
        assert close(r.after.available_options, 695_076)
        assert close(r.investor_shares, 1_737_691)
        assert r.conversions[0].basis == "discount"
        assert r.conversions[0].liquidation_preference == pytest.approx(260_356, abs=5)

    def test_step_by_step_example_2(self) -> None:
        note = Convertible(
            "Note", 350_000, "note", cap=12 * M, discount=0.3, interest_rate=0.08,
            years_outstanding=804 / 365,
        )
        t = RoundTerms(5 * M, 10 * M, option_pool_target=0.10, convertibles=(note,))
        r = price_round(CapTable(4.6 * M, available_options=400_000), t)
        assert close(r.after.fully_diluted, 8_721_051, 2)
        assert close(r.after.available_options, 872_105, 2)
        assert close(r.investor_shares, 2_907_017, 2)
        assert close(r.conversions[0].shares, 341_929, 2)
        assert r.conversions[0].liquidation_preference == pytest.approx(411_677, abs=1)

    def test_effective_pre_money_bridge(self) -> None:
        note = Convertible("Note", 300_000, "note", discount=0.3)
        t = RoundTerms(3 * M, 7 * M, option_pool_target=0.12, convertibles=(note,))
        r = price_round(CapTable(4 * M), t)
        assert r.effective_pre_money == pytest.approx(r.price * 4 * M)
        assert r.effective_pre_money < t.pre_money

    def test_impossible_round_is_rejected(self) -> None:
        t = RoundTerms(5 * M, 0.2 * M, option_pool_target=0.5)
        with pytest.raises(RoundInputError):
            price_round(CapTable(M), t)

    def test_overselling_post_money_safes_is_rejected(self) -> None:
        safe = Convertible("SAFE", 2 * M, cap=2 * M)
        with pytest.raises(RoundInputError):
            price_round(CapTable(M), RoundTerms(M, 10 * M, convertibles=(safe,)))

    def test_duplicate_names_are_rejected(self) -> None:
        safe = Convertible("Seed", M, cap=10 * M)
        with pytest.raises(RoundInputError):
            RoundTerms(M, 10 * M, name="Seed", convertibles=(safe,))


class TestPostMoneySafe:
    def test_ownership_equals_investment_over_cap(self) -> None:
        safes = (
            Convertible("A", 500_000, cap=5 * M),
            Convertible("B", 250_000, cap=5 * M),
        )
        r = price_round(CapTable(9 * M, available_options=M), RoundTerms(3 * M, 20 * M, convertibles=safes))
        converted = sum(c.shares for c in r.conversions)
        pre_new_money = r.before.fully_diluted + converted
        assert converted / pre_new_money == pytest.approx(0.15)

    def test_new_pool_does_not_change_safe_stake_before_new_money(self) -> None:
        safe = Convertible("SAFE", M, cap=10 * M)
        r = price_round(
            CapTable(9 * M), RoundTerms(3 * M, 20 * M, option_pool_target=0.1, convertibles=(safe,))
        )
        shares = r.conversions[0].shares
        assert shares / (9 * M + shares) == pytest.approx(0.10)

    def test_converts_at_round_price_below_cap(self) -> None:
        safe = Convertible("SAFE", 100_000, cap=100 * M)
        r = price_round(CapTable(9 * M), RoundTerms(M, 5 * M, convertibles=(safe,)))
        assert r.conversions[0].basis == "round price"
        assert r.conversions[0].price == pytest.approx(r.price)


class TestAntiDilution:
    """Example 5.3: $5M Series B at $6.75 after a $13.50 Series A."""

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [("none", 13.5), ("broad", 11.25), ("narrow", 9.0), ("full_ratchet", 6.75)],
    )
    def test_new_conversion_price(self, kind: str, expected: float) -> None:
        series_a = ShareClass(
            "A", 5 * M / 13.5, 13.5, 5 * M, anti_dilution=kind  # type: ignore[arg-type]
        )
        price = adjusted_conversion_price(series_a, 6.75, 5 * M, 5 * M / 13.5 * 4)
        assert price == pytest.approx(expected, abs=1e-6)

    def test_up_round_changes_nothing(self) -> None:
        series_a = ShareClass("A", 100.0, 2.0, 200.0, anti_dilution="full_ratchet")
        assert adjusted_conversion_price(series_a, 3.0, 100.0, 1000.0) == 2.0

    @pytest.mark.parametrize("kind", ["broad", "narrow", "full_ratchet"])
    def test_down_round_is_self_consistent(self, kind: str) -> None:
        start = price_round(CapTable(3 * M), RoundTerms(5 * M, 15 * M))
        cap = start.after.with_anti_dilution(kind)  # type: ignore[arg-type]
        down = price_round(cap, RoundTerms(5 * M, 8 * M, name="Series B"))
        assert down.antidilution_shares > 0
        assert down.price * down.after.fully_diluted == pytest.approx(13 * M)
        assert down.effective_pre_money < 8 * M
        if kind == "full_ratchet":
            adjusted = down.after.preferred[0]
            assert adjusted.current_conversion_price == pytest.approx(down.price)

    def test_ratchet_hurts_founders_most(self) -> None:
        start = price_round(CapTable(3 * M), RoundTerms(5 * M, 15 * M))
        stakes = [
            price_round(
                start.after.with_anti_dilution(kind), RoundTerms(5 * M, 8 * M, name="B")
            ).founder_ownership
            for kind in ("none", "broad", "narrow", "full_ratchet")
        ]
        assert stakes == sorted(stakes, reverse=True)


class TestSingleClassWaterfall:
    """Figure 5.9: founders' proceeds under three kinds of preferred."""

    @pytest.mark.parametrize(
        ("exit_m", "none", "full", "capped"),
        [
            (5, 0, 0, 0),
            (10, 5, 3.75, 3.75),
            (20, 15, 11.25, 11.25),
            (50, 37.5, 33.75, 35),
            (60, 45, 41.25, 45),
            (100, 75, 71.25, 75),
        ],
    )
    def test_founder_proceeds(
        self, exit_m: float, none: float, full: float, capped: float
    ) -> None:
        for kind, expected in (("none", none), ("full", full), ("capped", capped)):
            result = waterfall(simple_preferred(kind), [exit_m * M]).founders[0]
            assert result == pytest.approx(expected * M), kind

    def test_three_x_participating_example_7_4(self) -> None:
        cap = simple_preferred("full", liquidation_multiple=3.0)
        assert waterfall(cap, [15 * M]).founders[0] == pytest.approx(0)
        assert waterfall(cap, [55 * M]).founders[0] == pytest.approx(30 * M)

    def test_capped_breakpoints_example_7_3(self) -> None:
        events = {e["event"]: e["exit_value"] for e in breakpoints(simple_preferred("capped"))}
        assert events["Series A reaches its participation cap"] == pytest.approx(45 * M)
        assert events["Series A converts to common"] == pytest.approx(60 * M)


class TestMultiClassWaterfall:
    @staticmethod
    def two_series(participation: str) -> CapTable:
        """Examples 8.1 and 8.2."""
        seed = ShareClass("Seed", 500_000, 2.0, M, seniority=1)
        series_a = ShareClass(
            "A", M, 5.0, 5 * M, seniority=2, participation=participation  # type: ignore[arg-type]
        )
        return CapTable(2.5 * M, available_options=500_000, preferred=(seed, series_a))

    def test_non_participating_example_8_1(self) -> None:
        cap = self.two_series("none")
        values = [e["exit_value"] for e in breakpoints(cap)]
        assert values == pytest.approx([6 * M, 12 * M, 22.5 * M])
        w = waterfall(cap, [3 * M, 5.5 * M, 10 * M, 20 * M, 30 * M])
        assert w.payouts["A"] == pytest.approx([3 * M, 5 * M, 5 * M, 5 * M, 30 * M / 4.5])
        assert w.payouts["Seed"] == pytest.approx(
            [0, 0.5 * M, M, 15 * M / 7, 30 * M / 9]
        )

    def test_participating_example_8_2(self) -> None:
        cap = self.two_series("full")
        events = {e["event"]: e["exit_value"] for e in breakpoints(cap)}
        assert events["Seed converts to common"] == pytest.approx(14 * M)
        at_20 = waterfall(cap, [20 * M])
        assert at_20.payouts["A"][0] == pytest.approx(5 * M + 15 * M / 4.5)

    def test_step_by_step_example_1(self) -> None:
        note = Convertible(
            "Note", 200_000, "note", cap=5 * M, discount=0.2, interest_rate=0.08,
            years_outstanding=189 / 365,
        )
        t = RoundTerms(
            2.25 * M, 3.75 * M, option_pool_target=0.15, convertibles=(note,),
            shadow_series=False, participation="capped", participation_cap=3.0,
        )
        cap = price_round(CapTable(2 * M), t).after
        points = [e for e in breakpoints(cap) if e["event"].startswith("New investors")]
        values = {e["event"].split()[-1]: e["exit_value"] for e in points}
        assert values["cap"] == pytest.approx(14.5 * M, abs=0.05 * M)
        assert values["common"] == pytest.approx(18 * M, abs=0.05 * M)

    def test_step_by_step_example_3(self) -> None:
        note = Convertible(
            "Series A-1", 575_000, "note", cap=4 * M, discount=0.15,
            interest_rate=0.06, years_outstanding=414 / 365, compounding="annual",
        )
        series_a = RoundTerms(
            3.5 * M, 11.5 * M, name="Series A", option_pool_target=0.15,
            pool_in_pre_money=False, convertibles=(note,),
            convertibles_in_pre_money=False, dividend_rate=0.08,
        )
        first = price_round(CapTable(2.85 * M, available_options=150_000), series_a)
        assert close(first.after.fully_diluted, 4_969_127, 5)
        assert close(first.after.available_options, 745_369, 5)

        series_b = RoundTerms(
            10 * M, 22 * M, name="Series B", participation="full", dividend_rate=0.08
        )
        aged = first.after.with_elapsed_years(363 / 365)
        second = price_round(aged, series_b)
        assert close(second.after.fully_diluted, 7_227_822, 10)

        events = {e["event"]: e["exit_value"] for e in breakpoints(second.after)}
        assert events["All preferences covered"] == pytest.approx(14.44 * M, abs=0.01 * M)
        assert events["Series A-1 converts to common"] == pytest.approx(22.9 * M, abs=0.05 * M)
        assert events["Series A converts to common"] == pytest.approx(39.9 * M, abs=0.1 * M)

    def test_pari_passu_splits_by_preference(self) -> None:
        cap = CapTable(
            M,
            preferred=(
                ShareClass("X", 100_000, 30.0, 3 * M, seniority=1),
                ShareClass("Y", 100_000, 10.0, M, seniority=1),
            ),
        )
        w = waterfall(cap, [2 * M])
        assert w.payouts["X"][0] == pytest.approx(1.5 * M)
        assert w.payouts["Y"][0] == pytest.approx(0.5 * M)


def random_cap(rng: np.random.Generator) -> CapTable:
    classes = []
    for i in range(int(rng.integers(1, 5))):
        kind = str(rng.choice(["none", "full", "capped"]))
        multiple = float(rng.choice([1.0, 1.5, 2.0]))
        classes.append(
            ShareClass(
                f"C{i}",
                shares=float(rng.uniform(1e5, 2e6)),
                original_price=float(rng.uniform(0.5, 8)),
                invested=1.0,
                liquidation_multiple=multiple,
                participation=kind,  # type: ignore[arg-type]
                participation_cap=multiple + float(rng.uniform(0.5, 3)),
                seniority=int(rng.integers(1, 4)),
            )
        )
    return CapTable(
        founders=float(rng.uniform(1e6, 5e6)),
        available_options=float(rng.uniform(0, 1e6)),
        preferred=tuple(classes),
    )


class TestWaterfallProperties:
    @pytest.mark.parametrize("seed", range(25))
    def test_proceeds_add_up_and_never_fall(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        cap = random_cap(rng)
        exits = np.linspace(0, 200 * M, 801)
        w = waterfall(cap, exits)
        total = np.sum(list(w.payouts.values()), axis=0)
        assert total == pytest.approx(exits, abs=1e-4)
        for payout in w.payouts.values():
            assert np.all(payout >= -1e-6)
            assert np.all(np.diff(payout) >= -1e-4)

    @pytest.mark.parametrize("seed", range(15))
    def test_no_class_gains_by_switching(self, seed: int) -> None:
        """Conversion choices form an equilibrium at every exit value."""
        rng = np.random.default_rng(100 + seed)
        cap = random_cap(rng)
        engine = _Engine(cap)
        exits = np.array([0.5, 2, 5, 10, 20, 40, 80, 160]) * M
        chosen, per_share = engine.run(exits)
        options = [
            i for i, c in enumerate(cap.preferred) if c.participation != "full"
        ]
        for column, value in enumerate(exits):
            # Recover the converted set from the payouts.
            converted = np.zeros(len(cap.preferred), dtype=bool)
            for k in range(len(engine.order), -1, -1):
                mask = engine._converted_mask(k)
                trial, _ = engine._distribute(np.array([value]), mask)
                if np.allclose(trial[:, 0], chosen[:, column], atol=1e-4):
                    converted = mask
                    break
            for i in options:
                flipped = converted.copy()
                flipped[i] = not flipped[i]
                alternative, _ = engine._distribute(np.array([value]), flipped)
                assert alternative[i, 0] <= chosen[i, column] + 1e-3

    def test_brute_force_agrees_for_all_term_combinations(self) -> None:
        for kind_a, kind_b in product(["none", "capped"], repeat=2):
            cap = CapTable(
                3 * M,
                preferred=(
                    ShareClass("A", 500_000, 2.0, M, participation=kind_a, seniority=1),
                    ShareClass("B", M, 5.0, 5 * M, participation=kind_b, seniority=2),
                ),
            )
            engine = _Engine(cap)
            for value in (4 * M, 9 * M, 25 * M, 70 * M):
                payouts, _ = engine.run(np.array([value]))
                best_for_each = []
                for mask_bits in product([False, True], repeat=2):
                    trial, _ = engine._distribute(np.array([value]), np.array(mask_bits))
                    best_for_each.append(trial[:, 0])
                assert any(np.allclose(payouts[:, 0], t, atol=1e-4) for t in best_for_each)


class TestDecisionTools:
    @pytest.fixture
    def company(self) -> CapTable:
        return CapTable(8 * M, available_options=M)

    def test_minimum_pre_money_hits_target(self, company: CapTable) -> None:
        t = RoundTerms(3 * M, 9 * M, option_pool_target=0.1)
        pre_money = minimum_pre_money(company, t, 0.65)
        assert pre_money is not None
        stake = price_round(company, replace(t, pre_money=pre_money)).founder_ownership
        assert stake == pytest.approx(0.65, abs=1e-6)

    def test_minimum_pre_money_unreachable(self, company: CapTable) -> None:
        safe = Convertible("SAFE", 5 * M, cap=10 * M)
        t = RoundTerms(3 * M, 30 * M, convertibles=(safe,))
        assert minimum_pre_money(company, t, 0.60) is None

    def test_ownership_grid_marks_impossible_cells(self, company: CapTable) -> None:
        t = RoundTerms(3 * M, 9 * M, option_pool_target=0.4)
        grid = ownership_grid(company, t, [M, 3 * M], [0.5 * M, 12 * M])
        assert np.isnan(grid[1, 0])
        assert grid[0, 1] > grid[1, 1]

    def test_later_rounds_dilute_everyone_equally(self, company: CapTable) -> None:
        cap = price_round(company, RoundTerms(3 * M, 9 * M)).after
        later = add_later_rounds(cap, 10 * M, 0.25)
        assert later.founder_ownership == pytest.approx(cap.founder_ownership * 0.75)
        assert later.preferred[-1].seniority > cap.preferred[-1].seniority

    def test_exit_for_payout_inverts_the_waterfall(self, company: CapTable) -> None:
        cap = price_round(company, RoundTerms(3 * M, 9 * M, name="Seed")).after
        needed = exit_for_payout(cap, "Seed", 30 * M)
        assert needed == pytest.approx(120 * M, rel=1e-6)
        assert exit_for_payout(cap, FOUNDERS, 1e20) is None

    def test_same_seed_gives_same_exits(self) -> None:
        assumptions = ExitAssumptions(p_no_exit=0.5)
        first, second = assumptions.draw(1000), assumptions.draw(1000)
        assert np.array_equal(first, second)
        assert np.mean(first == 0) == pytest.approx(0.5, abs=0.05)

    def test_probability_matches_simulation(self) -> None:
        assumptions = ExitAssumptions(p_no_exit=0.4, median_exit=50 * M, sigma=1.5)
        exits = assumptions.draw(400_000)
        assert np.mean(exits >= 100 * M) == pytest.approx(
            assumptions.probability_at_least(100 * M), abs=0.003
        )

    def test_worse_terms_never_help_founders(self, company: CapTable) -> None:
        exits = ExitAssumptions().draw(20_000)
        clean = price_round(company, RoundTerms(3 * M, 9 * M)).after
        harsh = price_round(
            company,
            RoundTerms(3 * M, 9 * M, liquidation_multiple=2.0, participation="full"),
        ).after
        assert founder_outcome(harsh, exits).expected < founder_outcome(clean, exits).expected

    def test_levers_cover_terms_and_rank_by_value(self, company: CapTable) -> None:
        safe = Convertible("SAFE", 500_000, cap=6 * M)
        t = RoundTerms(
            3 * M, 9 * M, option_pool_target=0.15, convertibles=(safe,),
            liquidation_multiple=2.0, participation="full", dividend_rate=0.08,
            shadow_series=False,
        )
        exits = ExitAssumptions().draw(20_000)
        levers = negotiation_levers(
            company, t, exits, 50 * M, lambda c: c.with_elapsed_years(5)
        )
        names = {lever.name for lever in levers}
        assert {
            "Liquidation preference", "Participation", "Cumulative dividends",
            "Option pool timing", "Option pool size", "Preference overhang",
        } <= names
        values = [lever.expected_change for lever in levers]
        assert values == sorted(values, reverse=True)
        assert all(lever.expected_change >= -1e-6 for lever in levers)
