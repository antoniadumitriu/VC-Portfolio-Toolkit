"""Smoke tests for the Streamlit app using Streamlit's AppTest harness."""
from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).parent.parent / "streamlit_app.py")


def _run(**sidebar):
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.run()
    for label, value in sidebar.items():
        widget = next(w for w in at.sidebar.radio if w.label == label)
        widget.set_value(value)
    if sidebar:
        at.run()
    return at


def test_sample_portfolio_renders() -> None:
    at = _run()
    assert not at.exception
    assert not at.error
    labels = [m.label for m in at.metric]
    assert "Median MOIC" in labels
    assert "Final net TVPI" in labels


def test_blank_template_renders() -> None:
    at = _run(**{"Start from": "Blank template"})
    assert not at.exception
    assert not at.error


def test_upload_prompt_without_file() -> None:
    at = _run(**{"Start from": "Upload a file"})
    assert not at.exception
    assert at.info


def _founder_page():
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page("pages/founder_round_modeler.py")
    at.run()
    return at


def test_founder_page_renders() -> None:
    at = _founder_page()
    assert not at.exception
    assert not at.error
    labels = [m.label for m in at.metric]
    assert "Founders after the round" in labels
    assert "Effective pre-money" in labels
    assert "Chance of nothing" in labels


def test_founder_page_reports_impossible_terms() -> None:
    at = _founder_page()
    next(w for w in at.number_input if w.key == "preA").set_value(0.1)
    next(w for w in at.number_input if w.key == "poolA").set_value(50.0)
    at.run()
    assert not at.exception
    assert at.error


def test_founder_page_handles_each_preferred_type() -> None:
    at = _founder_page()
    for kind in ("Participating, capped", "Fully participating", "Non-participating"):
        next(w for w in at.selectbox if w.key == "typeA").set_value(kind)
        at.run()
        assert not at.exception
        assert not at.error


def test_founder_page_second_offer_and_extreme_inputs() -> None:
    at = _founder_page()
    next(w for w in at.radio if w.label.startswith("Offer to examine")).set_value("Offer B")
    next(w for w in at.number_input if w.key == "lpB").set_value(0.0)
    next(w for w in at.number_input if w.label == "Capital raised ($M)").set_value(0.0)
    at.run()
    assert not at.exception
    assert not at.error
