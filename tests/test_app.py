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
