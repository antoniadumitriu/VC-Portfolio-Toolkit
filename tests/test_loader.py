"""Tests for loading portfolios from tables and CSV files."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vc_portfolio import Portfolio, PortfolioInputError, parse_portfolio_table
from vc_portfolio.portfolio import draw_exit_multiples, expected_exit_multiple

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_portfolio.csv"


def test_sample_csv_loads() -> None:
    p = Portfolio.from_csv(SAMPLE, seed=1)
    assert len(p) == 10
    assert p.simulate_exits(500)["moic_p50"] > 0


def test_friendly_headers_and_formats() -> None:
    df = pd.DataFrame(
        {
            "Company": ["A", "B"],
            "Amount": ["$2M", "1,500,000"],
            "Stage": ["Series A", "pre-seed"],
            "Entry Year": [0, 1],
            "Exit Year": [6, 8.5],
            "Ownership": ["15%", 12],
        }
    )
    clean = parse_portfolio_table(df)
    assert clean["investment_amount"].tolist() == [2_000_000, 1_500_000]
    assert clean["stage"].tolist() == ["series_a", "pre_seed"]
    assert clean["ownership_pct"].tolist() == pytest.approx([0.15, 0.12])
    assert clean["sector"].tolist() == ["unspecified", "unspecified"]
    assert clean["expected_multiple"].isna().all()


def test_blank_rows_are_dropped() -> None:
    df = pd.DataFrame(
        {"name": ["A", None], "investment_amount": [1e6, None], "stage": ["seed", None],
         "entry_year": [0, None], "exit_year": [5, None]}
    )
    assert len(parse_portfolio_table(df)) == 1


def test_all_errors_reported_together() -> None:
    df = pd.DataFrame(
        {"name": ["A", ""], "investment_amount": [-5, "abc"], "stage": ["seed", "angel"],
         "entry_year": [3, 0], "exit_year": [2, 5]}
    )
    with pytest.raises(PortfolioInputError) as exc:
        parse_portfolio_table(df)
    text = "\n".join(exc.value.errors)
    assert "positive" in text
    assert "after entry_year" in text
    assert "angel" in text
    assert "name is empty" in text
    assert "not a number" in text


def test_missing_columns() -> None:
    with pytest.raises(PortfolioInputError, match="exit_year"):
        Portfolio.from_csv(io.StringIO("name,investment_amount,stage,entry_year\nA,1,seed,0\n"))


def test_expected_multiple_matches_simulation() -> None:
    rng = np.random.default_rng(0)
    for stage in ["seed", "series_b"]:
        sim = draw_exit_multiples(rng, stage, 400_000).mean()
        # Heavy Pareto tail: allow a loose tolerance
        assert sim == pytest.approx(expected_exit_multiple(stage), rel=0.08)
