# tests/test_stress_testing.py
import pandas as pd
import numpy as np
import pytest
from src.risk.var_models import portfolio_returns_series
from src.scenarios.stress_testing import stress_scenarios
from src.config import DEFAULT_TICKERS, DEFAULT_WEIGHTS

def test_covid_scenario_matches_notebook():
    log_returns = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
    w = np.array([DEFAULT_WEIGHTS[t] for t in DEFAULT_TICKERS])
    pr = portfolio_returns_series(log_returns, w)
    results = stress_scenarios(pr)
    covid = results["COVID Crash"]
    assert covid["worst_day"] == pytest.approx(-11.74, abs=0.05)
    assert covid["pre_var_99"] == pytest.approx(0.029826, abs=1e-3)