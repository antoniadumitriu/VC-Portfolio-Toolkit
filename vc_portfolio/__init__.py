"""
vc_portfolio — A polished Python library for VC fund portfolio analysis.

Public API surface:

    from vc_portfolio import Fund, FundConfig
    from vc_portfolio import Portfolio, Company
    from vc_portfolio import JCurve, CompanyTimeline
    from vc_portfolio import ScenarioSet, MonteCarloSimulator, SensitivityAnalysis
    from vc_portfolio import BEAR_SCENARIO, BASE_SCENARIO, BULL_SCENARIO
    from vc_portfolio import metrics
    from vc_portfolio import visualization as viz
"""
from __future__ import annotations

# Core data classes and engines
from vc_portfolio.fund import Fund, FundConfig
from vc_portfolio.jcurve import CompanyTimeline, JCurve
from vc_portfolio.portfolio import Company, Portfolio, STAGE_LOSS_RATES
from vc_portfolio.scenarios import (
    BASE_SCENARIO,
    BEAR_SCENARIO,
    BULL_SCENARIO,
    MonteCarloSimulator,
    Scenario,
    ScenarioResults,
    ScenarioSet,
    SensitivityAnalysis,
)

# Submodules available for direct import
from vc_portfolio import metrics
from vc_portfolio import visualization

__version__ = "0.1.0"
__author__ = "vc-portfolio-toolkit"

__all__ = [
    # Fund
    "Fund",
    "FundConfig",
    # Portfolio
    "Company",
    "Portfolio",
    "STAGE_LOSS_RATES",
    # J-Curve
    "CompanyTimeline",
    "JCurve",
    # Scenarios
    "Scenario",
    "ScenarioSet",
    "ScenarioResults",
    "MonteCarloSimulator",
    "SensitivityAnalysis",
    "BEAR_SCENARIO",
    "BASE_SCENARIO",
    "BULL_SCENARIO",
    # Submodules
    "metrics",
    "visualization",
    # Version
    "__version__",
]
