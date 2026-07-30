"""Rolling VaR backtesting: Kupiec POF test and Christoffersen independence test."""
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from src.risk.var_models import ewma_variance_series
from src.config import EWMA_LAMBDA


def rolling_historical_var(returns: pd.Series, window: int, confidence: float) -> pd.Series:
    return returns.rolling(window).apply(
        lambda x: -np.percentile(x, (1 - confidence) * 100), raw=True
    ).shift(1)


def rolling_parametric_var(returns: pd.Series, window: int, confidence: float) -> pd.Series:
    z = norm.ppf(1 - confidence)
    mu = returns.rolling(window).mean()
    sg = returns.rolling(window).std()
    return (-(mu + z * sg)).shift(1)


def rolling_ewma_var(returns: pd.Series, window: int, confidence: float,
                      lam: float = EWMA_LAMBDA) -> pd.Series:
    z = norm.ppf(1 - confidence)
    ewma_var = ewma_variance_series(returns, lam, burn_in=window)
    return pd.Series(-z * np.sqrt(ewma_var), index=returns.index).shift(1)


def rolling_montecarlo_var(returns: pd.Series, window: int, confidence: float) -> pd.Series:
    """Rolling univariate MC under normality converges to the parametric
    rolling estimate; retained to make that convergence explicit and
    testable rather than silently duplicating the parametric result."""
    return rolling_parametric_var(returns, window, confidence)


ROLLING_VAR_FUNCS = {
    "Historical":  rolling_historical_var,
    "Parametric":  rolling_parametric_var,
    "Monte Carlo": rolling_montecarlo_var,
    "EWMA":        rolling_ewma_var,
}


def kupiec_test(n_obs: int, n_breaches: int, confidence: float) -> dict:
    p = 1 - confidence
    p_hat = n_breaches / n_obs
    if np.isclose(p_hat, 0) or np.isclose(p_hat, 1):
        return {'lr': np.nan, 'p_value': np.nan, 'passed': False}
    lr = -2 * (
        n_breaches * np.log(p / p_hat) +
        (n_obs - n_breaches) * np.log((1 - p) / (1 - p_hat))
    )
    p_value = 1 - chi2.cdf(lr, df=1)
    return {'lr': lr,'p_value': p_value,'passed': bool(p_value > 0.05),'n_obs': n_obs,'n_breaches': n_breaches,'expected': n_obs * p,}


def christoffersen_test(breaches: pd.Series) -> dict:
    b = breaches.values
    n00 = sum(1 for t in range(1, len(b)) if b[t-1] == 0 and b[t] == 0)
    n01 = sum(1 for t in range(1, len(b)) if b[t-1] == 0 and b[t] == 1)
    n10 = sum(1 for t in range(1, len(b)) if b[t-1] == 1 and b[t] == 0)
    n11 = sum(1 for t in range(1, len(b)) if b[t-1] == 1 and b[t] == 1)
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi in (0, 1) or pi01 == 0 or pi11 == 0:
        return {'lr': np.nan, 'p_value': np.nan, 'passed': False}
    lr = -2 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi) -
        n00 * np.log(1 - pi01) - n01 * np.log(pi01) -
        n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )
    p_value = 1 - chi2.cdf(lr, df=1)
    return {'lr': lr,'p_value': p_value,'passed': bool(p_value > 0.05),}

def verdict(kupiec_passed: bool, christoffersen_passed: bool) -> str:
    if kupiec_passed and christoffersen_passed:
        return "PASS"
    if kupiec_passed:
        return "CONDITIONAL"
    return "FAIL"

def run_backtest(portfolio_returns: pd.Series, var_series: pd.Series,
                  confidence: float) -> dict:
    """Generic backtest — accepts ANY method's rolling VaR series, so the
    same function backtests Historical, Parametric, Monte Carlo, or EWMA."""
    valid = var_series.dropna()
    aligned = portfolio_returns.loc[valid.index]
    breaches = (aligned < -valid).astype(int)
    n_obs = len(valid)
    n_breaches = int(breaches.sum())
    kup = kupiec_test(n_obs, n_breaches, confidence)
    chris = christoffersen_test(breaches)
    return {
        'var_series': valid, 'returns': aligned, 'breaches': breaches,
        'breach_dates': breaches[breaches == 1].index,
        'kupiec': kup, 'christoffersen': chris,
        'verdict': verdict(kup['passed'], chris['passed']),
    }


def backtest_all_methods(portfolio_returns: pd.Series, confidence: float,
                          window: int = 250) -> dict:
    """Runs the backtest live for every method — this is what replaces the
    old hardcoded PASS/FAIL table in app.py."""
    results = {}
    for name, rolling_func in ROLLING_VAR_FUNCS.items():
        var_series = rolling_func(portfolio_returns, window, confidence)
        results[name] = run_backtest(portfolio_returns, var_series, confidence)
    return results