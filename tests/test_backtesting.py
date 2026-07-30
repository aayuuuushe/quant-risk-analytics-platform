# tests/test_backtesting.py
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