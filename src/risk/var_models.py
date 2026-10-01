"""Core VaR/ES models: historical, parametric, Monte Carlo, EWMA."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import MC_SEED, EWMA_LAMBDA, EWMA_BURN_IN


def portfolio_returns_series(log_returns: pd.DataFrame,
                              weights: np.ndarray) -> pd.Series:
    simple_returns = np.exp(log_returns) - 1
    portfolio_simple = simple_returns.dot(weights)
    return np.log1p(portfolio_simple)


def portfolio_mu(log_returns: pd.DataFrame, weights: np.ndarray) -> float:
    return weights @ log_returns.mean().values


def portfolio_sigma(log_returns: pd.DataFrame, weights: np.ndarray) -> float:
    cov = log_returns.cov().values
    return np.sqrt(weights @ cov @ weights)


def historical_var(portfolio_returns: pd.Series, confidence: float) -> float:
    return -np.percentile(portfolio_returns, (1 - confidence) * 100)


def historical_es(portfolio_returns: pd.Series, confidence: float) -> float:
    var = historical_var(portfolio_returns, confidence)
    tail = portfolio_returns[portfolio_returns <= -var]
    return -tail.mean()


def parametric_var(log_returns: pd.DataFrame, weights: np.ndarray,
                    confidence: float) -> float:
    mu_p = portfolio_mu(log_returns, weights)
    sigma_p = portfolio_sigma(log_returns, weights)
    z = norm.ppf(1 - confidence)
    return -(mu_p + z * sigma_p)


def parametric_es(log_returns: pd.DataFrame, weights: np.ndarray,
                   confidence: float) -> float:
    mu_p = portfolio_mu(log_returns, weights)
    sigma_p = portfolio_sigma(log_returns, weights)
    z = norm.ppf(1 - confidence)
    return -mu_p + sigma_p * norm.pdf(z) / (1 - confidence)


def monte_carlo_var(log_returns, weights, confidence,
                     n_simulations: int = 10_000, seed: int = 42) -> float:
    np.random.seed(seed)  
    mu = log_returns.mean().values
    cov = log_returns.cov().values
    simulated = np.random.multivariate_normal(mu, cov, n_simulations)
    port_sim = simulated @ weights
    return -np.percentile(port_sim, (1 - confidence) * 100)


def monte_carlo_full(log_returns, weights, confidence,
                      n_simulations: int = 10_000, seed: int = 42) -> dict:
    np.random.seed(seed)
    mu = log_returns.mean().values
    cov = log_returns.cov().values
    simulated = np.random.multivariate_normal(mu, cov, n_simulations)
    port_sim = simulated @ weights
    var = -np.percentile(port_sim, (1 - confidence) * 100)
    es = -port_sim[port_sim <= -var].mean()
    return {'simulated': port_sim, 'var': var, 'es': es}


def ewma_variance_series(portfolio_returns: pd.Series,lam: float = EWMA_LAMBDA,
                         burn_in: int = EWMA_BURN_IN) -> np.ndarray:
    r = portfolio_returns.values
    n = len(r)
    ewma_var = np.full(n, np.nan)
    ewma_var[burn_in] = np.var(r[:burn_in])
    for t in range(burn_in + 1, n):
        ewma_var[t] = lam * ewma_var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return ewma_var

def ewma_var_current(portfolio_returns: pd.Series,confidence: float,
                     lam: float = EWMA_LAMBDA,burn_in: int = EWMA_BURN_IN) -> tuple:
    ewma_var = ewma_variance_series(portfolio_returns, lam, burn_in)
    current_vol = np.sqrt(ewma_var[-1])
    z = norm.ppf(1 - confidence)
    var = -z * current_vol
    es = current_vol * norm.pdf(z) / (1 - confidence)
    return current_vol, var, es


def ewma_rolling_var(portfolio_returns: pd.Series, confidence: float,
                      lam: float = EWMA_LAMBDA, burn_in: int = EWMA_BURN_IN) -> tuple:
    ewma_var = ewma_variance_series(portfolio_returns, lam, burn_in)
    idx = portfolio_returns.index
    ewma_vol = pd.Series(np.sqrt(ewma_var), index=idx)
    z = norm.ppf(1 - confidence)
    rolling_var = pd.Series(-z * ewma_vol.values, index=idx)
    return ewma_vol, rolling_var