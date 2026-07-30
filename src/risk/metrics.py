"""Portfolio-level risk metrics: diversification benefit and decomposition."""
import numpy as np
import pandas as pd

from src.risk.var_models import historical_var, portfolio_returns_series


def diversification_benefit(log_returns: pd.DataFrame, weights: np.ndarray,
                             confidence: float) -> float:
    """
    Weighted-average of standalone asset VaRs minus portfolio VaR —
    the risk reduction achieved purely through diversification.
    """
    tickers = log_returns.columns.tolist()
    individual_vars = [
        -np.percentile(log_returns[t], (1 - confidence) * 100)
        for t in tickers
    ]
    weighted_avg = float(np.dot(weights, individual_vars))
    port_ret = portfolio_returns_series(log_returns, weights)
    port_var = historical_var(port_ret, confidence)
    return weighted_avg - port_var