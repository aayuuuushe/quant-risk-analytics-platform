"""Rolling VaR/ES backtesting.

Every forecast series is aligned so the value at date t was computable at the
close of t-1 (no look-ahead). A 'breach' (exception) is a day whose loss is
bigger than that day's VaR forecast.

Tests
  * Kupiec POF         - did we get the right NUMBER of breaches?
  * Christoffersen     - are breaches independent (not bunched together)?
  * Conditional cover. - both at once (sum of the two statistics, 2 d.o.f.)
  * ES tail check      - on breach days, how big were losses vs the model's ES?
Monte Carlo is not backtested: under a normal model it is the Parametric
method (same distribution), so a separate row would just repeat it.
"""
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

from src.config import BACKTEST_WINDOW, EWMA_LAMBDA
from src.risk.var_models import ewma_variance_series


# ------------------------------------------------------------ rolling forecasts
def rolling_historical_var(returns: pd.Series, window: int, confidence: float) -> pd.Series:
    q = (1 - confidence) * 100
    return returns.rolling(window).apply(
        lambda x: -np.percentile(x, q), raw=True).shift(1)


def rolling_historical_es(returns: pd.Series, window: int, confidence: float) -> pd.Series:
    q = (1 - confidence) * 100

    def _es(x):
        return -x[x <= np.percentile(x, q)].mean()
    return returns.rolling(window).apply(_es, raw=True).shift(1)


def rolling_parametric_var(returns: pd.Series, window: int, confidence: float) -> pd.Series:
    z = norm.ppf(1 - confidence)
    mu, sg = returns.rolling(window).mean(), returns.rolling(window).std()
    return (-(mu + z * sg)).shift(1)


def rolling_parametric_es(returns: pd.Series, window: int, confidence: float) -> pd.Series:
    z = norm.ppf(1 - confidence)
    mu, sg = returns.rolling(window).mean(), returns.rolling(window).std()
    return (-mu + sg * norm.pdf(z) / (1 - confidence)).shift(1)


def _ewma_sigma(returns: pd.Series, window: int, lam: float) -> pd.Series:
    # ewma_variance_series[t] is ALREADY a forecast for day t made with returns
    # through t-1, so there must be no further shift here.
    v = ewma_variance_series(returns, lam, burn_in=window)
    return pd.Series(np.sqrt(v), index=returns.index)


def rolling_ewma_var(returns: pd.Series, window: int, confidence: float,
                     lam: float = EWMA_LAMBDA) -> pd.Series:
    return -norm.ppf(1 - confidence) * _ewma_sigma(returns, window, lam)


def rolling_ewma_es(returns: pd.Series, window: int, confidence: float,
                    lam: float = EWMA_LAMBDA) -> pd.Series:
    z = norm.ppf(1 - confidence)
    return _ewma_sigma(returns, window, lam) * norm.pdf(z) / (1 - confidence)


ROLLING_VAR_FUNCS = {
    "Historical": rolling_historical_var,
    "Parametric": rolling_parametric_var,
    "EWMA":       rolling_ewma_var,
}
ROLLING_ES_FUNCS = {
    "Historical": rolling_historical_es,
    "Parametric": rolling_parametric_es,
    "EWMA":       rolling_ewma_es,
}


# ------------------------------------------------------------ statistical tests
def kupiec_test(n_obs: int, n_breaches: int, confidence: float) -> dict:
    """Kupiec proportion-of-failures likelihood-ratio test (chi-square, 1 d.o.f.).
    xlogy(0, 0) = 0, so zero breaches is handled correctly (it is a legitimate
    outcome: the model was too cautious)."""
    if n_obs <= 0:
        return {'lr': np.nan, 'p_value': np.nan, 'passed': False,
                'n_obs': n_obs, 'n_breaches': n_breaches, 'expected': np.nan}
    p, x, n = 1 - confidence, n_breaches, n_obs
    p_hat = x / n
    lr = -2 * (xlogy(x, p) + xlogy(n - x, 1 - p)
               - xlogy(x, p_hat) - xlogy(n - x, 1 - p_hat))
    lr = float(max(lr, 0.0))
    p_value = float(chi2.sf(lr, df=1))
    return {'lr': lr, 'p_value': p_value, 'passed': bool(p_value > 0.05),
            'n_obs': n, 'n_breaches': x, 'expected': n * p}


def christoffersen_test(breaches: pd.Series) -> dict:
    """Christoffersen independence test on the 0/1 breach sequence.
    Compares P(breach tomorrow | breach today) with P(breach tomorrow | no
    breach today). Zero consecutive breaches is the best case, not a failure."""
    b = np.asarray(breaches).astype(int)
    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    n = n00 + n01 + n10 + n11
    if n == 0:
        return {'lr': np.nan, 'p_value': np.nan, 'passed': False,
                'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11}
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / n
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
              + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr = float(max(-2 * (ll_null - ll_alt), 0.0))
    p_value = float(chi2.sf(lr, df=1))
    return {'lr': lr, 'p_value': p_value, 'passed': bool(p_value > 0.05),
            'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11}


def conditional_coverage_test(kupiec: dict, christoffersen: dict) -> dict:
    """Joint test of correct frequency AND independence: LR_cc = LR_uc + LR_ind."""
    lr = kupiec['lr'] + christoffersen['lr']
    p_value = float(chi2.sf(lr, df=2)) if np.isfinite(lr) else np.nan
    return {'lr': lr, 'p_value': p_value,
            'passed': bool(p_value > 0.05) if np.isfinite(p_value) else False}


def es_tail_check(returns: pd.Series, var_series: pd.Series,
                  es_series: pd.Series) -> dict:
    """Simple ES check (the statistic behind Acerbi-Szekely's Z1 test).
    On breach days compare the actual loss with the ES the model forecast for
    that day. Ratio near 1 = ES realistic; above 1 = tail losses were larger
    than the model said. No p-value is attached (that needs simulation)."""
    loss = -returns.loc[var_series.index]
    breach = loss > var_series
    n_b = int(breach.sum())
    if n_b == 0:
        return {'n_breaches': 0, 'avg_loss_on_breach': np.nan,
                'avg_forecast_es_on_breach': np.nan, 'ratio': np.nan}
    es_b = es_series.loc[var_series.index][breach]
    return {'n_breaches': n_b,
            'avg_loss_on_breach': float(loss[breach].mean()),
            'avg_forecast_es_on_breach': float(es_b.mean()),
            'ratio': float((loss[breach] / es_b).mean())}


def verdict(kupiec_passed: bool, christoffersen_passed: bool) -> str:
    """PASS = right frequency and independent; CLUSTERED = right frequency but
    breaches bunch together; FAIL = wrong frequency."""
    if kupiec_passed and christoffersen_passed:
        return "PASS"
    if kupiec_passed:
        return "CLUSTERED"
    return "FAIL"


# ------------------------------------------------------------ orchestration
def run_backtest(portfolio_returns: pd.Series, var_series: pd.Series,
                 confidence: float, es_series: pd.Series = None) -> dict:
    valid = var_series.dropna()
    aligned = portfolio_returns.loc[valid.index]
    breaches = (-aligned > valid).astype(int)
    n_obs, n_breaches = len(valid), int(breaches.sum())
    kup = kupiec_test(n_obs, n_breaches, confidence)
    chris = christoffersen_test(breaches)
    out = {
        'var_series': valid, 'returns': aligned, 'breaches': breaches,
        'breach_dates': breaches[breaches == 1].index,
        'kupiec': kup, 'christoffersen': chris,
        'conditional_coverage': conditional_coverage_test(kup, chris),
        'verdict': verdict(kup['passed'], chris['passed']),
    }
    if es_series is not None:
        out['es_check'] = es_tail_check(aligned, valid, es_series)
    return out


def backtest_all_methods(portfolio_returns: pd.Series, confidence: float,
                         window: int = BACKTEST_WINDOW) -> dict:
    """Backtest every method on EXACTLY the same dates."""
    var_all = {n: f(portfolio_returns, window, confidence)
               for n, f in ROLLING_VAR_FUNCS.items()}
    es_all = {n: f(portfolio_returns, window, confidence)
              for n, f in ROLLING_ES_FUNCS.items()}
    common = None
    for n in var_all:
        ok = var_all[n].dropna().index.intersection(es_all[n].dropna().index)
        common = ok if common is None else common.intersection(ok)
    return {n: run_backtest(portfolio_returns, var_all[n].loc[common],
                            confidence, es_all[n].loc[common])
            for n in var_all}
