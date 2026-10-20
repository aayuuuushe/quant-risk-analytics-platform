"""Core VaR/ES models: historical, parametric, Monte Carlo, EWMA.

CONVENTION (used by every model, backtest and stress test in this project)
  * R  = the portfolio's simple (percentage) return for one day
         R = sum_i w_i * (exp(logret_i) - 1)   -> what the money actually did
  * L  = -R is the loss. VaR and ES are reported as positive loss fractions;
         multiply by portfolio value to get dollars.
  * One-day horizon. Weights are held constant (daily rebalancing).
The data file stores asset LOG returns; they are converted to simple returns
once, here, so that all methods describe the same portfolio return series.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import MC_SEED, EWMA_LAMBDA, EWMA_BURN_IN


# ---------------------------------------------------------------- building blocks
def asset_simple_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert asset log returns to simple returns: exp(r) - 1."""
    return np.expm1(log_returns)


def _as_weights(weights, n_assets: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != n_assets:
        raise ValueError(f"Expected {n_assets} weights, got shape {w.shape}.")
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights contain NaN or infinite values.")
    return w


def portfolio_returns_series(log_returns: pd.DataFrame,
                             weights: np.ndarray) -> pd.Series:
    """Daily portfolio simple return: sum of weight x asset simple return."""
    w = _as_weights(weights, log_returns.shape[1])
    return asset_simple_returns(log_returns).dot(w)


def portfolio_mu(log_returns: pd.DataFrame, weights: np.ndarray) -> float:
    w = _as_weights(weights, log_returns.shape[1])
    return float(w @ asset_simple_returns(log_returns).mean().values)


def portfolio_sigma(log_returns: pd.DataFrame, weights: np.ndarray) -> float:
    """sqrt(w' Sigma w) with Sigma the covariance of asset simple returns.
    Equals the sample std of portfolio_returns_series exactly."""
    w = _as_weights(weights, log_returns.shape[1])
    cov = asset_simple_returns(log_returns).cov().values
    return float(np.sqrt(w @ cov @ w))


# ---------------------------------------------------------------- historical
def historical_var(portfolio_returns: pd.Series, confidence: float) -> float:
    return float(-np.percentile(portfolio_returns, (1 - confidence) * 100))


def historical_es(portfolio_returns: pd.Series, confidence: float) -> float:
    """Average loss on days at least as bad as the VaR day."""
    var = historical_var(portfolio_returns, confidence)
    tail = portfolio_returns[portfolio_returns <= -var]
    return float(-tail.mean())


# ---------------------------------------------------------------- parametric
def parametric_var(log_returns: pd.DataFrame, weights: np.ndarray,
                   confidence: float) -> float:
    mu_p = portfolio_mu(log_returns, weights)
    sigma_p = portfolio_sigma(log_returns, weights)
    z = norm.ppf(1 - confidence)
    return float(-(mu_p + z * sigma_p))


def parametric_es(log_returns: pd.DataFrame, weights: np.ndarray,
                  confidence: float) -> float:
    mu_p = portfolio_mu(log_returns, weights)
    sigma_p = portfolio_sigma(log_returns, weights)
    z = norm.ppf(1 - confidence)
    return float(-mu_p + sigma_p * norm.pdf(z) / (1 - confidence))


# ---------------------------------------------------------------- Monte Carlo
def _simulate_portfolio(log_returns: pd.DataFrame, weights: np.ndarray,
                        n_simulations: int, seed: int) -> np.ndarray:
    """Draw asset simple returns from a multivariate normal with the
    historical mean and covariance, and combine them with the weights."""
    w = _as_weights(weights, log_returns.shape[1])
    simple = asset_simple_returns(log_returns)
    # Draw in a fixed (alphabetical) asset order so the result does not depend on
    # the order the columns happen to be in; the same seed then always gives the
    # same simulated portfolio for the same set of assets.
    order = np.argsort(simple.columns.astype(str).to_numpy(), kind="stable")
    simple, w = simple.iloc[:, order], w[order]
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(simple.mean().values, simple.cov().values,
                                    n_simulations)
    return draws @ w


def monte_carlo_var(log_returns, weights, confidence,
                    n_simulations: int = 10_000, seed: int = MC_SEED) -> float:
    sim = _simulate_portfolio(log_returns, weights, n_simulations, seed)
    return float(-np.percentile(sim, (1 - confidence) * 100))


def monte_carlo_full(log_returns, weights, confidence,
                     n_simulations: int = 10_000, seed: int = MC_SEED) -> dict:
    sim = _simulate_portfolio(log_returns, weights, n_simulations, seed)
    var = float(-np.percentile(sim, (1 - confidence) * 100))
    es = float(-sim[sim <= -var].mean())
    return {'simulated': sim, 'var': var, 'es': es}


# ---------------------------------------------------------------- EWMA
def ewma_variance_series(portfolio_returns: pd.Series, lam: float = EWMA_LAMBDA,
                         burn_in: int = EWMA_BURN_IN) -> np.ndarray:
    """EWMA variance FORECAST for each day t, made with returns through t-1:
        var[t] = lam * var[t-1] + (1 - lam) * r[t-1]**2
    Seeded with the variance of the first `burn_in` returns, so the first
    usable forecast is for day index `burn_in`. Earlier entries are NaN."""
    r = np.asarray(portfolio_returns, dtype=float)
    n = len(r)
    if n <= burn_in:
        raise ValueError(f"Need more than {burn_in} returns, got {n}.")
    var = np.full(n, np.nan)
    var[burn_in] = np.var(r[:burn_in])
    for t in range(burn_in + 1, n):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return var


def ewma_next_day_variance(portfolio_returns: pd.Series, lam: float = EWMA_LAMBDA,
                           burn_in: int = EWMA_BURN_IN) -> float:
    """Variance forecast for the day AFTER the last observation (includes the
    last observed return)."""
    r = np.asarray(portfolio_returns, dtype=float)
    last_forecast = ewma_variance_series(portfolio_returns, lam, burn_in)[-1]
    return float(lam * last_forecast + (1 - lam) * r[-1] ** 2)


def ewma_var_current(portfolio_returns: pd.Series, confidence: float,
                     lam: float = EWMA_LAMBDA, burn_in: int = EWMA_BURN_IN) -> tuple:
    """(next-day volatility, VaR, ES) under zero-mean normal returns."""
    vol = float(np.sqrt(ewma_next_day_variance(portfolio_returns, lam, burn_in)))
    z = norm.ppf(1 - confidence)
    var = -z * vol
    es = vol * norm.pdf(z) / (1 - confidence)
    return vol, float(var), float(es)


def ewma_rolling_var(portfolio_returns: pd.Series, confidence: float,
                     lam: float = EWMA_LAMBDA, burn_in: int = EWMA_BURN_IN) -> tuple:
    """(volatility series, VaR series). Value at date t is the forecast that
    was available at the close of t-1."""
    ewma_var = ewma_variance_series(portfolio_returns, lam, burn_in)
    idx = portfolio_returns.index
    ewma_vol = pd.Series(np.sqrt(ewma_var), index=idx)
    z = norm.ppf(1 - confidence)
    return ewma_vol, pd.Series(-z * ewma_vol.values, index=idx)
