"""Portfolio-level risk metrics: diversification benefit."""
import numpy as np
import pandas as pd

from src.risk.var_models import (
    historical_var, portfolio_returns_series, asset_simple_returns,
)


def individual_historical_vars(log_returns: pd.DataFrame,
                               confidence: float) -> list:
    """Standalone historical VaR of each asset (simple returns), in column order."""
    simple = asset_simple_returns(log_returns)
    return [historical_var(simple[t], confidence) for t in simple.columns]


def diversification_benefit(log_returns: pd.DataFrame, weights: np.ndarray,
                            confidence: float) -> float:
    """Weighted average of standalone asset VaRs minus portfolio VaR: the risk
    reduction achieved purely by holding the assets together."""
    individual = individual_historical_vars(log_returns, confidence)
    weighted_avg = float(np.dot(weights, individual))
    port_var = historical_var(portfolio_returns_series(log_returns, weights),
                              confidence)
    return weighted_avg - port_var
