# tests/test_var_models.py
import numpy as np
import pandas as pd
import pytest
from src.risk.var_models import (
    portfolio_returns_series, historical_var, historical_es,
    parametric_var, monte_carlo_var, ewma_var_current,
)
from src.config import DEFAULT_TICKERS, DEFAULT_WEIGHTS

@pytest.fixture(scope="module")
def port_ret():
    log_returns = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
    w = np.array([DEFAULT_WEIGHTS[t] for t in DEFAULT_TICKERS])
    return portfolio_returns_series(log_returns, w), log_returns, w

def test_historical_var_95(port_ret):
    pr, _, _ = port_ret
    assert historical_var(pr, 0.95) == pytest.approx(0.018045, abs=1e-4)

def test_historical_var_99(port_ret):
    pr, _, _ = port_ret
    assert historical_var(pr, 0.99) == pytest.approx(0.034161, abs=1e-4)

def test_parametric_var_95(port_ret):
    _, lr, w = port_ret
    assert parametric_var(lr, w, 0.95) == pytest.approx(0.020046, abs=1e-4)

def test_monte_carlo_var_95_matches_notebook_seed(port_ret):
    _, lr, w = port_ret
    assert monte_carlo_var(lr, w, 0.95, 10_000, seed=42) == pytest.approx(0.020190, abs=5e-4)

def test_ewma_var_95(port_ret):
    pr, _, _ = port_ret
    _, var, _ = ewma_var_current(pr, 0.95)
    assert var == pytest.approx(0.014640, abs=1e-4)