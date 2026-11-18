"""
founder.py: Round pricing, cap tables and exit waterfalls from a founder's seat.

The rest of the toolkit looks at returns per dollar invested. This module looks
at a single company: what a term sheet does to the cap table, and who gets paid
at each exit value.

Three ideas drive the design:

* A round is priced by solving one equation, ``price * pre_money_shares =
  pre_money``. Option pools, notes, SAFEs and antidilution all make the share
  count depend on the price, so the price is found numerically instead of with
  a separate closed form for every combination of terms.
* A waterfall is a set of conversion decisions. Preferred classes convert in
  ascending order of what they would give up per as-converted share.
* Terms are compared on identical simulated exits, so differences between two
  offers come from the terms and not from sampling noise.

Method references: Dave Lishego, *The Founder's Guide to VC Finance* (2017) for
round mechanics and waterfalls, and Y Combinator's post-money SAFE user guide
for post-money SAFE conversion.

Simplifications: taxes, debt, transaction fees and option strike proceeds are
ignored, and every option (including the unissued pool) counts as exercised by
default, which is the conservative case for founders. Results are estimates and
not legal or financial advice.

Depends on: portfolio.py (stage loss rates only).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.stats import norm

from vc_portfolio.portfolio import STAGE_LOSS_RATES

FloatArray = npt.NDArray[np.float64]
ConvertibleKind = Literal["post_money_safe", "pre_money_safe", "note"]
Participation = Literal["none", "full", "capped"]
AntiDilution = Literal["none", "broad", "narrow", "full_ratchet"]

FOUNDERS = "Founders"
OTHER_COMMON = "Other common"
OPTION_POOL = "Option pool"

# Exit defaults: median $50M with about 35% of exits under $25M, the shape of
# the 2016 PitchBook exit figures. sigma = ln(50 / 25) / 0.385.
DEFAULT_MEDIAN_EXIT = 50_000_000.0
DEFAULT_EXIT_SIGMA = 1.8
DEFAULT_MAX_EXIT = 10_000_000_000.0


class RoundInputError(ValueError):
    """Raised when round terms are invalid or cannot be satisfied."""


# ---------------------------------------------------------------------------
# Securities and cap table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Convertible:
    """A SAFE or convertible note that converts in the next priced round."""

    name: str
    amount: float
    kind: ConvertibleKind = "post_money_safe"
    cap: Optional[float] = None
    discount: float = 0.0
    interest_rate: float = 0.0
    years_outstanding: float = 0.0
    compounding: Literal["simple", "annual"] = "simple"

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise RoundInputError(f"{self.name}: amount must be positive")
        if self.cap is not None and self.cap <= 0:
            raise RoundInputError(f"{self.name}: valuation cap must be positive")
        if not 0 <= self.discount < 1:
            raise RoundInputError(f"{self.name}: discount must be between 0% and 100%")
        if self.interest_rate < 0 or self.years_outstanding < 0:
            raise RoundInputError(f"{self.name}: interest and time cannot be negative")

    @property
    def conversion_amount(self) -> float:
        """Dollars that convert: principal, plus accrued interest on a note."""
        if self.kind != "note":
            return self.amount
        r, t = self.interest_rate, self.years_outstanding
        if self.compounding == "annual":
            return self.amount * (1 + r) ** t
        return self.amount * (1 + r * t)


@dataclass(frozen=True)
class ShareClass:
    """A class of preferred stock."""

    name: str
    shares: float
    original_price: float
    invested: float
    conversion_price: Optional[float] = None
    liquidation_multiple: float = 1.0
    participation: Participation = "none"
    participation_cap: float = 3.0
    dividend_rate: float = 0.0
    seniority: int = 1
    anti_dilution: AntiDilution = "broad"
    years_accrued: float = 0.0

    @property
    def current_conversion_price(self) -> float:
        return self.conversion_price or self.original_price

    @property
    def as_converted(self) -> float:
        """Common shares this class becomes on conversion."""
        return self.shares * self.original_price / self.current_conversion_price

    @property
    def redemption_value(self) -> float:
        """Liquidation preference plus accrued cumulative dividends."""
        per_share = self.liquidation_multiple + self.dividend_rate * self.years_accrued
        return self.shares * self.original_price * per_share

    @property
    def cap_value(self) -> float:
        """Most a capped participating class can receive without converting."""
        capped = self.shares * self.original_price * self.participation_cap
        return max(capped, self.redemption_value)


@dataclass(frozen=True)
class CapTable:
    """Fully diluted ownership of a company."""

    founders: float
    other_common: float = 0.0
    available_options: float = 0.0
    preferred: tuple[ShareClass, ...] = ()

    def __post_init__(self) -> None:
        if self.founders <= 0:
            raise RoundInputError("Founder shares must be positive")
        if self.other_common < 0 or self.available_options < 0:
            raise RoundInputError("Share counts cannot be negative")

    @property
    def common_shares(self) -> float:
        return self.founders + self.other_common + self.available_options

    @property
    def fully_diluted(self) -> float:
        return self.common_shares + sum(c.as_converted for c in self.preferred)

    @property
    def founder_ownership(self) -> float:
        return self.founders / self.fully_diluted

    def holdings(self) -> dict[str, float]:
        """As-converted shares by holder, omitting empty lines."""
        rows = {
            FOUNDERS: self.founders,
            OTHER_COMMON: self.other_common,
            OPTION_POOL: self.available_options,
        }
        rows.update({c.name: c.as_converted for c in self.preferred})
        return {name: shares for name, shares in rows.items() if shares > 0}

    def ownership(self) -> dict[str, float]:
        total = self.fully_diluted
        return {name: shares / total for name, shares in self.holdings().items()}

    def with_elapsed_years(self, years: float) -> "CapTable":
        """Accrue cumulative dividends for ``years``."""
        aged = tuple(
            replace(c, years_accrued=c.years_accrued + years) for c in self.preferred
        )
        return replace(self, preferred=aged)

    def with_anti_dilution(self, kind: AntiDilution) -> "CapTable":
        """Give every preferred class the same antidilution protection."""
        classes = tuple(replace(c, anti_dilution=kind) for c in self.preferred)
        return replace(self, preferred=classes)


# ---------------------------------------------------------------------------
# Pricing a round
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundTerms:
    """The economic terms of a priced round."""

    investment: float
    pre_money: float
    name: str = "New investors"
    option_pool_target: float = 0.0
    pool_in_pre_money: bool = True
    convertibles: tuple[Convertible, ...] = ()
    convertibles_in_pre_money: bool = True
    shadow_series: bool = True
    liquidation_multiple: float = 1.0
    participation: Participation = "none"
    participation_cap: float = 3.0
    dividend_rate: float = 0.0
    anti_dilution: AntiDilution = "broad"
    pari_passu_with_prior: bool = False

    def __post_init__(self) -> None:
        if self.investment <= 0 or self.pre_money <= 0:
            raise RoundInputError("Investment and pre-money must be positive")
        if not 0 <= self.option_pool_target < 1:
            raise RoundInputError("Option pool target must be between 0% and 100%")
        if self.liquidation_multiple < 0 or self.dividend_rate < 0:
            raise RoundInputError("Preference and dividends cannot be negative")
        if self.participation == "capped" and (
            self.participation_cap < self.liquidation_multiple
        ):
            raise RoundInputError(
                "The participation cap must be at least the liquidation multiple"
            )
        names = [c.name for c in self.convertibles] + [self.name]
        if len(set(names)) != len(names):
            raise RoundInputError("Every SAFE, note and round needs a unique name")


@dataclass(frozen=True)
class Conversion:
    """How one SAFE or note converted."""

    name: str
    invested: float
    amount_converted: float
    shares: float
    price: float
    basis: Literal["cap", "discount", "round price"]
    liquidation_preference: float

    @property
    def preference_overhang(self) -> float:
        """Preference received beyond the dollars actually invested."""
        return max(0.0, self.liquidation_preference - self.invested)


@dataclass(frozen=True)
class RoundResult:
    """A priced round and everything that follows from it."""

    terms: RoundTerms
    before: CapTable
    after: CapTable
    price: float
    investor_shares: float
    new_options: float
    conversions: tuple[Conversion, ...]
    antidilution_shares: float

    @property
    def post_money(self) -> float:
        return self.price * self.after.fully_diluted

    @property
    def investor_ownership(self) -> float:
        return self.investor_shares / self.after.fully_diluted

    @property
    def founder_ownership(self) -> float:
        return self.after.founder_ownership

    @property
    def bridge(self) -> dict[str, float]:
        """Walk from the headline pre-money to what existing holders keep."""
        t = self.terms
        pool = self.price * self.before.available_options
        if t.pool_in_pre_money:
            pool += self.price * self.new_options
        converted = sum(c.shares for c in self.conversions)
        return {
            "Headline pre-money": t.pre_money,
            "Option pool": -pool,
            "SAFEs and notes": (
                -self.price * converted if t.convertibles_in_pre_money else 0.0
            ),
            "Antidilution": -self.price * self.antidilution_shares,
        }

    @property
    def effective_pre_money(self) -> float:
        """Value of the shares existing holders own, at the round price."""
        return float(sum(self.bridge.values()))


def adjusted_conversion_price(
    share_class: ShareClass,
    new_price: float,
    investment: float,
    shares_before: float,
) -> float:
    """Conversion price after a financing at ``new_price``."""
    current = share_class.current_conversion_price
    kind = share_class.anti_dilution
    if kind == "none" or new_price >= current:
        return current
    if kind == "full_ratchet":
        return new_price
    base = shares_before if kind == "broad" else share_class.as_converted
    issued = investment / new_price
    at_old_price = investment / current
    return current * (base + at_old_price) / (base + issued)


def _convert(
    convertibles: Sequence[Convertible], price: float, base_shares: float
) -> tuple[list[float], list[float], list[str]]:
    """Shares, prices and bases for each convertible at a given round price."""
    shares = [0.0] * len(convertibles)
    prices = [price] * len(convertibles)
    bases = ["round price"] * len(convertibles)
    # Post-money SAFE caps are measured after all converting securities, so the
    # share counts feed back into each other. The loop is a contraction because
    # post-money SAFEs are validated to sell less than 100% in total.
    for _ in range(1000):
        total = sum(shares)
        updated = []
        for i, c in enumerate(convertibles):
            best, basis = price, "round price"
            if c.discount > 0 and price * (1 - c.discount) < best:
                best, basis = price * (1 - c.discount), "discount"
            if c.cap is not None:
                capitalization = base_shares
                if c.kind == "post_money_safe":
                    capitalization += total
                cap_price = c.cap / capitalization
                if cap_price < best:
                    best, basis = cap_price, "cap"
            prices[i], bases[i] = best, basis
            updated.append(c.conversion_amount / best)
        change = max((abs(a - b) for a, b in zip(updated, shares)), default=0.0)
        shares = updated
        if change <= 1e-9:
            break
    return shares, prices, bases


@dataclass(frozen=True)
class _Solved:
    prior: tuple[ShareClass, ...]
    existing: float
    conv_shares: list[float]
    conv_prices: list[float]
    conv_bases: list[str]
    investor_shares: float
    new_options: float
    pre_money_shares: float


def _state(cap: CapTable, t: RoundTerms, price: float) -> _Solved:
    """Every share count implied by a candidate round price."""
    before = cap.fully_diluted
    prior = tuple(
        replace(
            c,
            conversion_price=adjusted_conversion_price(
                c, price, t.investment, before
            ),
        )
        for c in cap.preferred
    )
    existing = cap.common_shares + sum(c.as_converted for c in prior)
    conv_shares, conv_prices, conv_bases = _convert(t.convertibles, price, existing)
    converted = sum(conv_shares)
    investor_shares = t.investment / price

    new_options = 0.0
    if t.option_pool_target > 0:
        other = existing - cap.available_options + converted + investor_shares
        pool = other * t.option_pool_target / (1 - t.option_pool_target)
        new_options = max(0.0, pool - cap.available_options)

    pre_money_shares = existing
    if t.convertibles_in_pre_money:
        pre_money_shares += converted
    if t.pool_in_pre_money:
        pre_money_shares += new_options
    return _Solved(
        prior, existing, conv_shares, conv_prices, conv_bases,
        investor_shares, new_options, pre_money_shares,
    )


def price_round(cap: CapTable, terms: RoundTerms) -> RoundResult:
    """
    Price a round and return the resulting cap table.

    Solves ``price * pre_money_shares(price) = pre_money``. The left side rises
    with price, so the root is unique when one exists. There is no root when
    the pool and converting securities alone would absorb the whole pre-money.
    """
    sold = sum(
        c.conversion_amount / c.cap
        for c in terms.convertibles
        if c.kind == "post_money_safe" and c.cap is not None
    )
    if sold >= 1:
        raise RoundInputError("The post-money SAFEs sell 100% or more of the company")

    def gap(price: float) -> float:
        return price * _state(cap, terms, price).pre_money_shares - terms.pre_money

    high = terms.pre_money / cap.fully_diluted
    low = high * 1e-9
    if gap(low) >= 0:
        raise RoundInputError(
            "These terms do not fit: the option pool and converting securities "
            "take up the entire pre-money valuation. Raise the pre-money, or "
            "shrink the pool or the round."
        )
    price = high if gap(high) <= 0 else float(
        brentq(gap, low, high, xtol=high * 1e-15, rtol=1e-14, maxiter=500)
    )
    s = _state(cap, terms, price)
    return _build_result(cap, terms, price, s)


def _build_result(
    cap: CapTable, t: RoundTerms, price: float, s: _Solved
) -> RoundResult:
    top = max((c.seniority for c in cap.preferred), default=0)
    rank = max(top, 1) if t.pari_passu_with_prior else top + 1

    def preferred(
        name: str, shares: float, original_price: float, invested: float
    ) -> ShareClass:
        return ShareClass(
            name=name,
            shares=shares,
            original_price=original_price,
            invested=invested,
            liquidation_multiple=t.liquidation_multiple,
            participation=t.participation,
            participation_cap=t.participation_cap,
            dividend_rate=t.dividend_rate,
            seniority=rank,
            anti_dilution=t.anti_dilution,
        )

    new_classes = []
    conversions = []
    for c, shares, conv_price, basis in zip(
        t.convertibles, s.conv_shares, s.conv_prices, s.conv_bases
    ):
        # A shadow series sets the preference on the price actually paid. Without
        # one, discounted shares carry the full round price per share.
        lp_price = conv_price if t.shadow_series else price
        share_class = preferred(c.name, shares, lp_price, c.amount)
        new_classes.append(share_class)
        conversions.append(
            Conversion(
                name=c.name,
                invested=c.amount,
                amount_converted=c.conversion_amount,
                shares=shares,
                price=conv_price,
                basis=basis,  # type: ignore[arg-type]
                liquidation_preference=share_class.redemption_value,
            )
        )
    new_classes.append(preferred(t.name, s.investor_shares, price, t.investment))

    after = replace(
        cap,
        available_options=cap.available_options + s.new_options,
        preferred=s.prior + tuple(new_classes),
    )
    before_prior = sum(c.as_converted for c in cap.preferred)
    after_prior = sum(c.as_converted for c in s.prior)
    return RoundResult(
        terms=t,
        before=cap,
        after=after,
        price=price,
        investor_shares=s.investor_shares,
        new_options=s.new_options,
        conversions=tuple(conversions),
        antidilution_shares=after_prior - before_prior,
    )


def minimum_pre_money(
    cap: CapTable, terms: RoundTerms, founder_ownership: float
) -> Optional[float]:
    """Lowest pre-money that leaves founders at ``founder_ownership``, if any."""

    def stake(pre_money: float) -> Optional[float]:
        try:
            return price_round(cap, replace(terms, pre_money=pre_money)).founder_ownership
        except RoundInputError:
            return None

    high = terms.investment * 1e6
    best = stake(high)
    if best is None or best < founder_ownership:
        return None
    low = high
    for _ in range(200):
        low /= 2
        current = stake(low)
        if current is None or current < founder_ownership:
            break
    else:
        return low
    for _ in range(120):
        middle = (low * high) ** 0.5
        current = stake(middle)
        if current is None or current < founder_ownership:
            low = middle
        else:
            high = middle
    return high


def ownership_grid(
    cap: CapTable,
    terms: RoundTerms,
    investments: Sequence[float],
    pre_moneys: Sequence[float],
) -> FloatArray:
    """Founder ownership for each (investment, pre-money); NaN if infeasible."""
    grid = np.full((len(investments), len(pre_moneys)), np.nan)
    for i, investment in enumerate(investments):
        for j, pre_money in enumerate(pre_moneys):
            try:
                candidate = replace(terms, investment=investment, pre_money=pre_money)
                grid[i, j] = price_round(cap, candidate).founder_ownership
            except RoundInputError:
                continue
    return grid


# ---------------------------------------------------------------------------
# Later financing
# ---------------------------------------------------------------------------


def add_later_rounds(
    cap: CapTable, capital: float, dilution: float, years: float = 0.0
) -> CapTable:
    """
    Add one senior, 1x non-participating class standing in for future rounds.

    Every existing holder is diluted by ``dilution``. Dividends on existing
    classes accrue for ``years``.
    """
    if not 0 <= dilution < 1:
        raise RoundInputError("Later dilution must be between 0% and 100%")
    if capital < 0:
        raise RoundInputError("Later capital cannot be negative")
    aged = cap.with_elapsed_years(years)
    if capital == 0 or dilution == 0:
        return aged
    shares = aged.fully_diluted * dilution / (1 - dilution)
    top = max((c.seniority for c in aged.preferred), default=0)
    later = ShareClass(
        name="Later rounds",
        shares=shares,
        original_price=capital / shares,
        invested=capital,
        seniority=top + 1,
        anti_dilution="none",
    )
    return replace(aged, preferred=aged.preferred + (later,))


# ---------------------------------------------------------------------------
# Waterfall
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Waterfall:
    """Proceeds by holder at each exit value."""

    exit_values: FloatArray
    payouts: dict[str, FloatArray]

    @property
    def founders(self) -> FloatArray:
        return self.payouts[FOUNDERS]


class _Engine:
    """Waterfall for one cap table, vectorized over exit values."""

    def __init__(self, cap: CapTable, include_unissued_pool: bool = True) -> None:
        self.cap = cap
        self.classes = cap.preferred
        self.pool = cap.available_options if include_unissued_pool else 0.0
        self.common = cap.founders + cap.other_common + self.pool
        n = len(self.classes)
        self.shares = np.array([c.as_converted for c in self.classes], dtype=float)
        self.lp = np.array([c.redemption_value for c in self.classes], dtype=float)
        self.ceiling = np.array([c.cap_value for c in self.classes], dtype=float)
        self.kind = [c.participation for c in self.classes]
        self.rank = np.array([c.seniority for c in self.classes], dtype=int)

        # What a class gives up per share by converting. Full participation
        # always beats converting, so those classes never convert in a sale.
        self.threshold = np.full(n, np.inf)
        for i, c in enumerate(self.classes):
            if self.shares[i] <= 0 or c.participation == "full":
                continue
            value = self.ceiling[i] if c.participation == "capped" else self.lp[i]
            self.threshold[i] = value / self.shares[i]
        order = np.argsort(self.threshold, kind="stable")
        self.order = [int(i) for i in order if np.isfinite(self.threshold[i])]

    def _converted_mask(self, k: int) -> npt.NDArray[np.bool_]:
        mask = np.zeros(len(self.classes), dtype=bool)
        mask[self.order[:k]] = True
        return mask

    def _per_share_curve(
        self, converted: npt.NDArray[np.bool_]
    ) -> tuple[FloatArray, FloatArray, float, list[tuple[int, float]]]:
        """Residual value to per-share value, as a piecewise-linear curve."""
        base = self.common
        limited: list[tuple[int, float]] = []
        for i, kind in enumerate(self.kind):
            if converted[i] or kind == "full":
                base += self.shares[i]
            elif kind == "capped" and self.shares[i] > 0:
                room = max(0.0, self.ceiling[i] - self.lp[i])
                limited.append((i, room / self.shares[i]))
        limited.sort(key=lambda item: item[1])

        active = base + sum(self.shares[i] for i, _ in limited)
        residual, level = 0.0, 0.0
        residuals, levels = [0.0], [0.0]
        for i, limit in limited:
            residual += (limit - level) * active
            level = limit
            active -= self.shares[i]
            residuals.append(residual)
            levels.append(level)
        return np.array(residuals), np.array(levels), active, limited

    def _distribute(
        self, exits: FloatArray, converted: npt.NDArray[np.bool_]
    ) -> tuple[FloatArray, FloatArray]:
        """Payouts by class and value per common share for fixed conversions."""
        n = len(self.classes)
        paid = np.zeros((n, exits.size))
        remaining = exits.copy()
        for rank in sorted(set(self.rank.tolist()), reverse=True):
            members = np.where((self.rank == rank) & ~converted)[0]
            total = self.lp[members].sum() if members.size else 0.0
            if total <= 0:
                continue
            payment = np.minimum(remaining, total)
            paid[members] = np.outer(self.lp[members] / total, payment)
            remaining -= payment

        residuals, levels, slope_shares, limited = self._per_share_curve(converted)
        per_share = np.interp(remaining, residuals, levels)
        beyond = remaining > residuals[-1]
        per_share[beyond] = levels[-1] + (remaining[beyond] - residuals[-1]) / slope_shares

        payouts = paid
        limits = dict(limited)
        for i, kind in enumerate(self.kind):
            if converted[i]:
                payouts[i] = self.shares[i] * per_share
            elif kind == "full":
                payouts[i] = paid[i] + self.shares[i] * per_share
            elif i in limits:
                payouts[i] = paid[i] + self.shares[i] * np.minimum(per_share, limits[i])
        return payouts, per_share

    def run(self, exits: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Payouts by class and per-share common value at equilibrium."""
        best_payouts, best_share = self._distribute(
            exits, self._converted_mask(0)
        )
        for k in range(1, len(self.order) + 1):
            payouts, per_share = self._distribute(exits, self._converted_mask(k))
            converts = per_share >= self.threshold[self.order[k - 1]]
            best_payouts[:, converts] = payouts[:, converts]
            best_share[converts] = per_share[converts]
        return best_payouts, best_share

    def _exit_for_level(self, k: int, level: float) -> float:
        """Exit value where common reaches ``level`` with ``k`` classes converted."""
        converted = self._converted_mask(k)
        residuals, levels, slope_shares, _ = self._per_share_curve(converted)
        if level <= levels[-1]:
            residual = float(np.interp(level, levels, residuals))
        else:
            residual = residuals[-1] + (level - levels[-1]) * slope_shares
        return residual + float(self.lp[~converted].sum())


def waterfall(
    cap: CapTable,
    exit_values: npt.ArrayLike,
    include_unissued_pool: bool = True,
) -> Waterfall:
    """Split each exit value among holders."""
    exits = np.atleast_1d(np.asarray(exit_values, dtype=float))
    if np.any(exits < 0):
        raise RoundInputError("Exit values cannot be negative")
    engine = _Engine(cap, include_unissued_pool)
    by_class, per_share = engine.run(exits)
    payouts = {FOUNDERS: cap.founders * per_share}
    if cap.other_common > 0:
        payouts[OTHER_COMMON] = cap.other_common * per_share
    if engine.pool > 0:
        payouts[OPTION_POOL] = engine.pool * per_share
    for i, share_class in enumerate(cap.preferred):
        payouts[share_class.name] = by_class[i]
    return Waterfall(exits, payouts)


def breakpoints(cap: CapTable, include_unissued_pool: bool = True) -> list[dict[str, object]]:
    """Exit values where the shape of the waterfall changes, in order."""
    engine = _Engine(cap, include_unissued_pool)
    events: list[dict[str, object]] = []
    total = float(engine.lp.sum())
    if total > 0:
        events.append({"event": "All preferences covered", "exit_value": total})
    for i, share_class in enumerate(cap.preferred):
        if share_class.participation == "capped" and engine.shares[i] > 0:
            limit = max(0.0, engine.ceiling[i] - engine.lp[i]) / engine.shares[i]
            k = sum(engine.threshold[j] <= limit for j in engine.order)
            events.append(
                {
                    "event": f"{share_class.name} reaches its participation cap",
                    "exit_value": engine._exit_for_level(k, limit),
                }
            )
    for k, i in enumerate(engine.order, start=1):
        events.append(
            {
                "event": f"{cap.preferred[i].name} converts to common",
                "exit_value": engine._exit_for_level(k, float(engine.threshold[i])),
            }
        )
    return sorted(events, key=lambda e: float(e["exit_value"]))  # type: ignore[arg-type]


def exit_for_payout(
    cap: CapTable, holder: str, target: float, upper: float = 1e14
) -> Optional[float]:
    """Smallest exit value at which ``holder`` receives ``target``, if any."""

    def received(value: float) -> float:
        return float(waterfall(cap, [value]).payouts[holder][0])

    if target <= 0:
        return 0.0
    if received(upper) < target:
        return None
    low, high = 0.0, upper
    for _ in range(200):
        middle = 0.5 * (low + high) if low > 0 else high / 2
        if received(middle) >= target:
            high = middle
        else:
            low = middle
        if high - low <= max(1e-2, high * 1e-11):
            break
    return high


# ---------------------------------------------------------------------------
# Exit odds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitAssumptions:
    """Chance of no exit, and a lognormal exit value if there is one."""

    p_no_exit: float = STAGE_LOSS_RATES["seed"]
    median_exit: float = DEFAULT_MEDIAN_EXIT
    sigma: float = DEFAULT_EXIT_SIGMA
    max_exit: float = DEFAULT_MAX_EXIT

    def __post_init__(self) -> None:
        if not 0 <= self.p_no_exit <= 1:
            raise RoundInputError("Chance of no exit must be between 0% and 100%")
        if self.median_exit <= 0 or self.sigma <= 0 or self.max_exit <= 0:
            raise RoundInputError("Exit assumptions must be positive")

    def draw(self, n: int, seed: int = 42) -> FloatArray:
        """Simulated exit values; the same seed gives the same exits."""
        rng = np.random.default_rng(seed)
        exited = rng.uniform(size=n) >= self.p_no_exit
        values = self.median_exit * np.exp(self.sigma * rng.standard_normal(n))
        return np.where(exited, np.minimum(values, self.max_exit), 0.0)

    def probability_at_least(self, value: float) -> float:
        """Chance the exit is worth at least ``value``."""
        if value > self.max_exit:
            return 0.0
        if value <= 0:
            return 1.0
        z = (np.log(value) - np.log(self.median_exit)) / self.sigma
        return float((1 - self.p_no_exit) * norm.sf(z))


@dataclass(frozen=True)
class FounderOutcome:
    """Distribution of founder proceeds over simulated exits."""

    expected: float
    median_if_exit: float
    p90: float
    probability_nothing: float
    payouts: FloatArray


def founder_outcome(cap: CapTable, exits: FloatArray) -> FounderOutcome:
    """Summarize founder proceeds over simulated exits."""
    payouts = waterfall(cap, exits).founders
    exited = exits > 0
    median = float(np.median(payouts[exited])) if exited.any() else 0.0
    return FounderOutcome(
        expected=float(payouts.mean()),
        median_if_exit=median,
        p90=float(np.percentile(payouts, 90)),
        probability_nothing=float(np.mean(payouts <= 0)),
        payouts=payouts,
    )


# ---------------------------------------------------------------------------
# Negotiation levers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lever:
    """The effect on founders of changing one term."""

    name: str
    change: str
    ownership_change: float
    expected_change: float
    typical_exit_change: float


def _candidate_levers(t: RoundTerms) -> list[tuple[str, str, RoundTerms]]:
    levers = [
        (
            "Headline valuation",
            "Pre-money up 10% (for scale)",
            replace(t, pre_money=t.pre_money * 1.1),
        )
    ]
    if t.liquidation_multiple > 1:
        levers.append(
            (
                "Liquidation preference",
                f"{t.liquidation_multiple:g}x to 1x",
                replace(
                    t,
                    liquidation_multiple=1.0,
                    participation_cap=max(t.participation_cap, 1.0),
                ),
            )
        )
    if t.participation != "none":
        label = "Full participation" if t.participation == "full" else "Capped participation"
        levers.append(
            ("Participation", f"{label} to non-participating", replace(t, participation="none"))
        )
    if t.dividend_rate > 0:
        levers.append(
            ("Cumulative dividends", f"{t.dividend_rate:.0%} to none", replace(t, dividend_rate=0.0))
        )
    if t.option_pool_target > 0 and t.pool_in_pre_money:
        levers.append(
            ("Option pool timing", "Pre-money to post-money", replace(t, pool_in_pre_money=False))
        )
    if t.option_pool_target > 0.05:
        smaller = max(0.05, t.option_pool_target - 0.05)
        levers.append(
            (
                "Option pool size",
                f"{t.option_pool_target:.0%} to {smaller:.0%}",
                replace(t, option_pool_target=smaller),
            )
        )
    if t.convertibles and t.convertibles_in_pre_money:
        levers.append(
            (
                "SAFE and note conversion",
                "In the pre-money to alongside the round",
                replace(t, convertibles_in_pre_money=False),
            )
        )
    if t.convertibles and not t.shadow_series:
        levers.append(
            (
                "Preference overhang",
                "Add a shadow series",
                replace(t, shadow_series=True),
            )
        )
    return levers


def negotiation_levers(
    cap: CapTable,
    terms: RoundTerms,
    exits: FloatArray,
    typical_exit: float,
    to_exit_cap: Callable[[CapTable], CapTable] = lambda c: c,
) -> list[Lever]:
    """
    Founder impact of moving each term to its founder-friendly version.

    ``to_exit_cap`` applies later financing before the waterfall. Each lever is
    evaluated on its own, on the same simulated exits.
    """
    base = price_round(cap, terms)

    def measure(result: RoundResult) -> tuple[float, float]:
        exit_cap = to_exit_cap(result.after)
        expected = float(waterfall(exit_cap, exits).founders.mean())
        typical = float(waterfall(exit_cap, [typical_exit]).founders[0])
        return expected, typical

    base_expected, base_typical = measure(base)
    levers = []
    for name, change, candidate in _candidate_levers(terms):
        try:
            result = price_round(cap, candidate)
        except RoundInputError:
            continue
        expected, typical = measure(result)
        levers.append(
            Lever(
                name=name,
                change=change,
                ownership_change=result.founder_ownership - base.founder_ownership,
                expected_change=expected - base_expected,
                typical_exit_change=typical - base_typical,
            )
        )
    return sorted(levers, key=lambda lever: -lever.expected_change)
