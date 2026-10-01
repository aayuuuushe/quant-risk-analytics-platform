import pandas as pd
import numpy as np
import pytest
from src.validation.backtesting import kupiec_test, christoffersen_test, verdict


def test_kupiec_matches_notebook_historical_95():
    result = kupiec_test(n_obs=2264, n_breaches=109, confidence=0.95)
    assert result["p_value"] == pytest.approx(0.6837, abs=1e-3)
    assert result["passed"] is True


def test_verdict_logic():
    assert verdict(True, True) == "PASS"
    assert verdict(True, False) == "CONDITIONAL"
    assert verdict(False, True) == "FAIL"
    assert verdict(False, False) == "FAIL"


def test_christoffersen_independent_breaches():
    rng = np.random.default_rng(7)
    b = pd.Series(rng.choice([0, 1], size=300, p=[0.9, 0.1]))
    result = christoffersen_test(b)
    assert result["passed"] is True


def test_christoffersen_clustered_breaches():
    # All breaches in one block — maximally clustered
    b = pd.Series([0] * 90 + [1] * 10)
    result = christoffersen_test(b)
    assert result["passed"] is False