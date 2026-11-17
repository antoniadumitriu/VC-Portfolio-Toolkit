"""
portfolio.py: Power law return simulation for VC portfolios.

No dependencies on other library modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd



# Stage-level loss rates calibrated to Correlation Ventures / CB Insights data
STAGE_LOSS_RATES: dict[str, float] = {
    "pre_seed": 0.65,
    "seed": 0.50,
    "series_a": 0.35,
    "series_b": 0.25,
    "series_c": 0.20,
    "growth": 0.15,
}

Stage = Literal["pre_seed", "seed", "series_a", "series_b", "series_c", "growth"]


# Mixture weights for the non-loss share of outcomes
ZOMBIE_SHARE = 0.25      # of non-loss outcomes, capped at 20% overall
ZOMBIE_CAP = 0.20
LOGNORMAL_SHARE = 0.65   # of non-loss outcomes; the rest goes to the Pareto tail
MAX_LOSS_RATE = 0.95


def draw_exit_multiples(
    rng: np.random.Generator,
    stage: str,
    n: int,
    loss_rate_adjustment: float = 0.0,
    loss_rate_multiplier: float = 1.0,
) -> npt.NDArray[np.float64]:
    """
    Draw gross exit multiples from the 4-component mixture.

    Components (weights are shares of all outcomes):
        p_loss                        -> 0x (complete loss)
        min(0.20, 0.25 * (1-p_loss))  -> Uniform(0.5, 1.5) zombie returns
        0.65 * (1-p_loss)             -> LogNormal(mu=ln 2, sigma=0.85), median 2x
        remainder                     -> Pareto(alpha=1.5, xm=2x) home runs

    ``p_loss`` is the stage loss rate, scaled by ``loss_rate_multiplier`` and
    shifted by ``loss_rate_adjustment``, then clipped to [0, 0.95].
    """
    base = STAGE_LOSS_RATES.get(stage, 0.40)
    p_loss = float(np.clip(base * loss_rate_multiplier + loss_rate_adjustment, 0.0, MAX_LOSS_RATE))
    p_rest = 1.0 - p_loss
    p_zombie = min(ZOMBIE_CAP, p_rest * ZOMBIE_SHARE)
    p_lognorm = p_rest * LOGNORMAL_SHARE

    u = rng.uniform(size=n)
    zombie = rng.uniform(0.5, 1.5, size=n)
    lognorm = rng.lognormal(mean=np.log(2.0), sigma=0.85, size=n)
    pareto = 2.0 / (1.0 - rng.uniform(size=n)) ** (1.0 / 1.5)

    return np.select(
        [u < p_loss, u < p_loss + p_zombie, u < p_loss + p_zombie + p_lognorm],
        [0.0, zombie, lognorm],
        default=pareto,
    ).astype(np.float64)


def portfolio_irrs(
    investments: npt.NDArray[np.float64],
    entry_years: npt.NDArray[np.float64],
    exit_years: npt.NDArray[np.float64],
    exit_values: npt.NDArray[np.float64],
    lo: float = -0.999,
    hi: float = 1000.0,
    n_grid: int = 96,
    n_iter: int = 60,
) -> npt.NDArray[np.float64]:
    """
    Vectorized IRR for many simulated portfolios.

    Each company invests ``investments[j]`` at ``entry_years[j]`` and returns
    ``exit_values[i, j]`` at ``exit_years[j]`` (years, may be fractional).
    For each row, NPV is evaluated on a grid in log(1 + r); the sign change
    closest to r = 0 is then refined by bisection. With staggered deals the
    NPV curve can have several roots, and this picks the economically
    meaningful one, matching ``metrics.calc_irr``.

    Returns -1.0 for rows with no proceeds, and the bound (lo or hi) when
    NPV never changes sign inside it.
    """
    values = np.atleast_2d(np.asarray(exit_values, dtype=np.float64))
    origin = float(np.min(entry_years))
    t0 = np.asarray(entry_years, dtype=np.float64) - origin
    t1 = np.asarray(exit_years, dtype=np.float64) - origin
    inv = np.asarray(investments, dtype=np.float64)

    def npv(g: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """NPV for each row at log-growth g (same shape as rows, or grid)."""
        g = g[..., None]
        return (values[:, None, :] * np.exp(-g * t1)).sum(-1) - (inv * np.exp(-g * t0)).sum(-1)

    n = values.shape[0]
    grid = np.linspace(np.log1p(lo), np.log1p(hi), n_grid)
    f = npv(np.broadcast_to(grid, (n, n_grid)))  # (n, n_grid)

    crosses = np.sign(f[:, :-1]) * np.sign(f[:, 1:]) <= 0
    # Distance of each bracket from r = 0, inf where there is no sign change
    dist = np.where(crosses, np.abs(0.5 * (grid[:-1] + grid[1:])), np.inf)
    k = np.argmin(dist, axis=1)
    has_root = np.isfinite(dist[np.arange(n), k])

    a = grid[k].copy()
    b = grid[k + 1].copy()
    f_a = f[np.arange(n), k]
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        f_m = npv(m[:, None])[:, 0]
        left = np.sign(f_m) == np.sign(f_a)
        a = np.where(left, m, a)
        f_a = np.where(left, f_m, f_a)
        b = np.where(left, b, m)

    irr = np.expm1(0.5 * (a + b))
    # No root in range: NPV positive everywhere means returns beyond hi
    irr = np.where(has_root, irr, np.where(f[:, -1] > 0, hi, lo))
    irr[values.sum(axis=1) <= 0] = -1.0
    return irr


def simulate_portfolio(
    companies: list[Company],
    n_simulations: int,
    rng: np.random.Generator,
    return_multiplier: float = 1.0,
    loss_rate_adjustment: float = 0.0,
    loss_rate_multiplier: float = 1.0,
    holding_period_multiplier: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Simulate portfolio MOIC and IRR across ``n_simulations`` draws.

    Periods on ``Company`` are in years. Returns (moics, irrs).
    """
    if not companies:
        raise ValueError("Portfolio is empty. Add companies before simulating.")

    investments = np.array([c.investment_amount for c in companies], dtype=np.float64)
    total = float(investments.sum())
    entry = np.array([c.entry_period for c in companies], dtype=np.float64)
    hold = np.array([c.holding_period for c in companies], dtype=np.float64)
    exit_ = entry + np.maximum(hold * holding_period_multiplier, 1e-6)

    multiples = np.column_stack(
        [
            draw_exit_multiples(
                rng, c.stage, n_simulations, loss_rate_adjustment, loss_rate_multiplier
            )
            for c in companies
        ]
    ) * return_multiplier
    values = multiples * investments

    moics = values.sum(axis=1) / total if total > 0 else np.zeros(n_simulations)
    irrs = portfolio_irrs(investments, entry, exit_, values)
    return moics, irrs


def summarize_simulation(
    moics: npt.NDArray[np.float64],
    irrs: npt.NDArray[np.float64],
) -> dict[str, object]:
    """Percentiles and probabilities for simulated MOICs and IRRs."""
    out: dict[str, object] = {}
    for p in (10, 25, 50, 75, 90):
        out[f"moic_p{p}"] = float(np.percentile(moics, p))
    for p in (10, 25, 50, 75, 90):
        out[f"irr_p{p}"] = float(np.percentile(irrs, p))
    out["loss_probability"] = float(np.mean(moics < 1.0))
    out["home_run_probability"] = float(np.mean(moics > 3.0))
    out["raw_moics"] = moics
    out["raw_irrs"] = irrs
    return out


@dataclass
class Company:
    """Represents a single portfolio company investment."""

    name: str
    investment_amount: float
    stage: Stage
    sector: str
    entry_valuation: float
    entry_period: float  # years from fund start
    expected_exit_period: float  # years from fund start
    ownership_pct: float  # post-money ownership percentage (e.g. 0.15 = 15%)
    reserved_capital: float = 0.0  # reserved for follow-on

    @property
    def total_committed(self) -> float:
        return self.investment_amount + self.reserved_capital

    @property
    def holding_period(self) -> float:
        return self.expected_exit_period - self.entry_period


class Portfolio:
    """
    VC portfolio modeler with power law exit simulation.

    Uses the modern NumPy Generator API (default_rng) for reproducibility.
    Exit multiples follow a 4-component mixture:
        - p_loss (stage loss rate) at 0x
        - zombie returns: Uniform(0.5, 1.5)
        - 65% of non-loss outcomes: LogNormal with median 2x
        - remainder: Pareto(alpha=1.5, xm=2x) home runs
    See ``draw_exit_multiples`` for exact weights.

    Supports method chaining:

        portfolio = (
            Portfolio(seed=42)
            .add_company(company_a)
            .add_companies([company_b, company_c])
        )
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._companies: list[Company] = []

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(cls, table: pd.DataFrame, seed: Optional[int] = None) -> Portfolio:
        """
        Build a portfolio from a table with one row per company.

        See ``parse_portfolio_table`` for the accepted columns and formats.
        Raises ``PortfolioInputError`` listing every problem found.
        """
        clean = parse_portfolio_table(table)
        companies = [
            Company(
                name=r.name,
                investment_amount=r.investment_amount,
                stage=r.stage,
                sector=r.sector,
                entry_valuation=r.entry_valuation,
                entry_period=r.entry_year,
                expected_exit_period=r.exit_year,
                ownership_pct=r.ownership_pct,
                reserved_capital=r.reserved_capital,
            )
            for r in clean.itertuples(index=False)
        ]
        return cls(seed=seed).add_companies(companies)

    @classmethod
    def from_csv(cls, source: object, seed: Optional[int] = None) -> Portfolio:
        """Build a portfolio from a CSV path or file-like object."""
        return cls.from_dataframe(pd.read_csv(source), seed=seed)  # type: ignore[call-overload]


    def add_company(self, company: Company) -> "Portfolio":
        """Add a single company to the portfolio."""
        self._companies.append(company)
        return self

    def add_companies(self, companies: list[Company]) -> "Portfolio":
        """Add multiple companies to the portfolio."""
        self._companies.extend(companies)
        return self

    # ------------------------------------------------------------------
    # Exit multiple simulation
    # ------------------------------------------------------------------

    def _draw_exit_multiple(
        self,
        stage: Stage,
        n: int = 1,
    ) -> npt.NDArray[np.float64]:
        """Draw exit multiples for one stage. See ``draw_exit_multiples``."""
        return draw_exit_multiples(self._rng, stage, n)

    # ------------------------------------------------------------------
    # Portfolio simulation
    # ------------------------------------------------------------------

    def simulate_exits(
        self,
        n_simulations: int = 10_000,
    ) -> dict[str, object]:
        """
        Run Monte Carlo exit simulations across all portfolio companies.

        Each simulation draws an exit multiple for every company, then
        computes portfolio MOIC and an IRR from the actual cash flow timing
        (investment at entry, proceeds at exit, in years).

        Returns
        -------
        dict with keys:
            moic_p10 ... moic_p90, irr_p10 ... irr_p90
            loss_probability (fraction of simulations with MOIC < 1)
            home_run_probability (fraction with MOIC > 3)
            raw_moics, raw_irrs (ndarrays of all simulated values)
        """
        moics, irrs = simulate_portfolio(self._companies, n_simulations, self._rng)
        return summarize_simulation(moics, irrs)

    # ------------------------------------------------------------------
    # Portfolio analytics
    # ------------------------------------------------------------------

    def reserve_ratio(self) -> dict[str, float]:
        """Return reserve ratio stats across portfolio."""
        if not self._companies:
            return {}

        total_initial = sum(c.investment_amount for c in self._companies)
        total_reserved = sum(c.reserved_capital for c in self._companies)
        total_committed = sum(c.total_committed for c in self._companies)

        return {
            "total_initial_investment": total_initial,
            "total_reserved": total_reserved,
            "total_committed": total_committed,
            "reserve_ratio": total_reserved / total_initial if total_initial > 0 else 0.0,
        }

    def follow_on_strategy(
        self,
        strategy: Literal["pro_rata", "top_up", "none"] = "pro_rata",
    ) -> pd.DataFrame:
        """
        Generate follow-on investment schedule by strategy.

        Parameters
        ----------
        strategy:
            'pro_rata': maintain ownership through each round
            'top_up': invest reserved capital in top performers only
            'none': no follow-on

        Returns
        -------
        DataFrame with columns: company, initial, follow_on, total, period
        """
        rows = []
        for company in self._companies:
            follow_on = 0.0
            if strategy == "pro_rata":
                follow_on = company.reserved_capital
            elif strategy == "top_up":
                # Invest in companies at Series B+ stages
                if company.stage in ("series_b", "series_c", "growth"):
                    follow_on = company.reserved_capital
            # else "none": no follow-on

            rows.append(
                {
                    "company": company.name,
                    "stage": company.stage,
                    "initial_investment": company.investment_amount,
                    "follow_on": follow_on,
                    "total_committed": company.investment_amount + follow_on,
                    "entry_period": company.entry_period,
                }
            )

        return pd.DataFrame(rows)

    def portfolio_breakdown(self) -> pd.DataFrame:
        """Return company-level breakdown DataFrame."""
        if not self._companies:
            return pd.DataFrame()

        rows = []
        total_invested = sum(c.investment_amount for c in self._companies)

        for company in self._companies:
            rows.append(
                {
                    "company": company.name,
                    "stage": company.stage,
                    "sector": company.sector,
                    "investment_amount": company.investment_amount,
                    "ownership_pct": company.ownership_pct,
                    "entry_valuation": company.entry_valuation,
                    "entry_period": company.entry_period,
                    "expected_exit_period": company.expected_exit_period,
                    "holding_period": company.holding_period,
                    "portfolio_weight": (
                        company.investment_amount / total_invested
                        if total_invested > 0
                        else 0.0
                    ),
                }
            )

        return pd.DataFrame(rows)

    def concentration_analysis(self) -> dict[str, object]:
        """
        Compute portfolio concentration metrics.

        Returns
        -------
        dict with:
            herfindahl_index: HHI of portfolio weights (0=diversified, 1=concentrated)
            top_3_concentration: % of portfolio in top 3 positions
            stage_diversity: number of distinct stages represented
            sector_diversity: number of distinct sectors represented
            stage_counts: dict of stage → count
            sector_counts: dict of sector → count
        """
        if not self._companies:
            return {}

        investments = np.array([c.investment_amount for c in self._companies])
        total = investments.sum()
        weights = investments / total if total > 0 else np.zeros_like(investments)

        # Herfindahl-Hirschman Index
        hhi = float(np.sum(weights**2))

        # Top 3 concentration
        sorted_weights = np.sort(weights)[::-1]
        top3 = float(sorted_weights[:3].sum())

        # Stage and sector diversity
        stages = [c.stage for c in self._companies]
        sectors = [c.sector for c in self._companies]

        from collections import Counter
        stage_counts = dict(Counter(stages))
        sector_counts = dict(Counter(sectors))

        return {
            "herfindahl_index": hhi,
            "top_3_concentration": top3,
            "stage_diversity": len(set(stages)),
            "sector_diversity": len(set(sectors)),
            "stage_counts": stage_counts,
            "sector_counts": sector_counts,
        }

    def __len__(self) -> int:
        return len(self._companies)

    def __repr__(self) -> str:
        total = sum(c.investment_amount for c in self._companies)
        return (
            f"Portfolio(n_companies={len(self._companies)}, "
            f"total_invested=${total:,.0f})"
        )


# ---------------------------------------------------------------------------
# Loading portfolios from tables (CSV, spreadsheets, editable UI tables)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["name", "investment_amount", "stage", "entry_year", "exit_year"]
OPTIONAL_COLUMNS: dict[str, object] = {
    "sector": "unspecified",
    "entry_valuation": 0.0,
    "ownership_pct": 0.0,
    "reserved_capital": 0.0,
    "expected_multiple": np.nan,
}
TABLE_COLUMNS = REQUIRED_COLUMNS + list(OPTIONAL_COLUMNS)

_COLUMN_ALIASES = {
    "company": "name",
    "company_name": "name",
    "amount": "investment_amount",
    "investment": "investment_amount",
    "check_size": "investment_amount",
    "entry_period": "entry_year",
    "entry": "entry_year",
    "exit_period": "exit_year",
    "expected_exit_period": "exit_year",
    "expected_exit_year": "exit_year",
    "exit": "exit_year",
    "valuation": "entry_valuation",
    "post_money": "entry_valuation",
    "ownership": "ownership_pct",
    "reserves": "reserved_capital",
    "multiple": "expected_multiple",
    "exit_multiple": "expected_multiple",
}


class PortfolioInputError(ValueError):
    """Raised when a portfolio table has problems. ``errors`` lists them all."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Portfolio input has problems:\n- " + "\n- ".join(errors))


def _normalize_key(text: object) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def _to_number(value: object) -> float:
    """Parse numbers like 2000000, '2,000,000', '$2.5M', '15%'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    if text == "":
        return float("nan")
    scale = 1.0
    if text[-1] in "kKmMbB%":
        scale = {"k": 1e3, "m": 1e6, "b": 1e9, "%": 0.01}[text[-1].lower()]
        text = text[:-1]
    return float(text) * scale


def expected_exit_multiple(stage: str) -> float:
    """Mean gross exit multiple of the mixture for a stage (analytic)."""
    p_loss = STAGE_LOSS_RATES.get(stage, 0.40)
    p_rest = 1.0 - p_loss
    p_zombie = min(ZOMBIE_CAP, p_rest * ZOMBIE_SHARE)
    p_lognorm = p_rest * LOGNORMAL_SHARE
    p_pareto = max(0.0, p_rest - p_zombie - p_lognorm)
    lognorm_mean = 2.0 * np.exp(0.85**2 / 2)
    pareto_mean = 1.5 * 2.0 / (1.5 - 1.0)
    return float(p_zombie * 1.0 + p_lognorm * lognorm_mean + p_pareto * pareto_mean)


def parse_portfolio_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a portfolio table.

    Required columns: name, investment_amount, stage, entry_year, exit_year.
    Optional: sector, entry_valuation, ownership_pct, reserved_capital,
    expected_multiple. Headers are case-insensitive and common aliases
    ("company", "amount", "ownership", ...) are accepted. Amounts may be
    written like "2,000,000" or "$2M"; ownership as 0.15, 15 or "15%".
    Stages accept forms like "Series A", "pre-seed" or "series_a".
    Rows that are completely empty are dropped.

    Returns a DataFrame with exactly ``TABLE_COLUMNS``.
    Raises ``PortfolioInputError`` listing every problem found.
    """
    df = table.copy()
    df.columns = [_COLUMN_ALIASES.get(_normalize_key(c), _normalize_key(c)) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(how="all")
    df = df[~df.apply(lambda r: all(str(v).strip() in ("", "nan", "None") for v in r), axis=1)]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PortfolioInputError([f"Missing column(s): {', '.join(missing)}"])
    if df.empty:
        raise PortfolioInputError(["The portfolio has no companies."])

    for col, default in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            df[col] = default

    errors: list[str] = []
    rows = []
    stages = list(STAGE_LOSS_RATES)
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        label = f"Row {i}"
        name = str(r["name"]).strip() if pd.notna(r["name"]) else ""
        if name:
            label += f" ({name})"
        else:
            errors.append(f"{label}: company name is empty")

        stage = _normalize_key(r["stage"]) if pd.notna(r["stage"]) else ""
        if stage not in STAGE_LOSS_RATES:
            errors.append(f"{label}: stage '{r['stage']}' not recognized; use one of {', '.join(stages)}")

        nums: dict[str, float] = {}
        for col in ["investment_amount", "entry_year", "exit_year", "entry_valuation",
                    "ownership_pct", "reserved_capital", "expected_multiple"]:
            try:
                nums[col] = _to_number(r[col])
            except (TypeError, ValueError):
                errors.append(f"{label}: '{r[col]}' is not a number ({col})")
                nums[col] = float("nan")

        if not nums["investment_amount"] > 0:
            errors.append(f"{label}: investment_amount must be a positive number")
        if np.isnan(nums["entry_year"]) or nums["entry_year"] < 0:
            errors.append(f"{label}: entry_year must be 0 or more (years from fund start)")
        if np.isnan(nums["exit_year"]):
            errors.append(f"{label}: exit_year is missing")
        elif not np.isnan(nums["entry_year"]) and nums["exit_year"] <= nums["entry_year"]:
            errors.append(f"{label}: exit_year must be after entry_year")

        for col in ["entry_valuation", "reserved_capital"]:
            if np.isnan(nums[col]):
                nums[col] = 0.0
            elif nums[col] < 0:
                errors.append(f"{label}: {col} cannot be negative")
        own = 0.0 if np.isnan(nums["ownership_pct"]) else nums["ownership_pct"]
        if own > 1:
            own /= 100.0  # entered as a percentage, e.g. 15
        if not 0 <= own <= 1:
            errors.append(f"{label}: ownership_pct must be between 0 and 100%")
        mult = nums["expected_multiple"]
        if not np.isnan(mult) and mult < 0:
            errors.append(f"{label}: expected_multiple cannot be negative")

        sector = str(r["sector"]).strip() if pd.notna(r["sector"]) and str(r["sector"]).strip() else "unspecified"
        rows.append(
            {
                "name": name,
                "investment_amount": nums["investment_amount"],
                "stage": stage,
                "entry_year": nums["entry_year"],
                "exit_year": nums["exit_year"],
                "sector": sector,
                "entry_valuation": nums["entry_valuation"],
                "ownership_pct": own,
                "reserved_capital": nums["reserved_capital"],
                "expected_multiple": mult,
            }
        )

    if errors:
        raise PortfolioInputError(errors)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)
