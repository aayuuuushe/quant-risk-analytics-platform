"""Historical scenario and hypothetical shock stress testing."""
import numpy as np
import pandas as pd

SCENARIOS = {
    'COVID Crash':     ('2020-02-20', '2020-03-23'),
    '2018 Selloff':    ('2018-10-01', '2018-12-31'),
    '2022 Rate Shock': ('2022-01-01', '2022-10-31'),
    'China Selloff':   ('2015-08-01', '2015-09-30'),
}


def pre_crisis_var(returns: pd.Series, crisis_start: str,
                    confidence: float, window: int = 250) -> float:
    """
    VaR estimated only from the `window` trading days immediately
    preceding `crisis_start` — avoids look-ahead from including the
    crisis itself in the estimation sample.
    """
    cutoff = pd.Timestamp(crisis_start) - pd.Timedelta(days=1)
    history = returns.loc[:cutoff].tail(window)
    return -np.percentile(history, (1 - confidence) * 100)


def stress_scenarios(portfolio_returns: pd.Series,
                      scenarios: dict = None) -> dict:
    scenarios = scenarios or SCENARIOS
    results = {}
    for name, (start, end) in scenarios.items():
        period = portfolio_returns.loc[start:end]
        if len(period) < 5:
            continue
        results[name] = {
            'returns': period,
            'total': (np.exp(period.sum()) - 1) * 100,
            'worst_day': period.min() * 100,
            'pre_var_99': pre_crisis_var(portfolio_returns, start, 0.99),
        }
    return results

def worst_drawdown_windows(returns: pd.Series, n_windows: int = 6,
                            window: int = 40) -> pd.Series:
    """
    Locates the n worst rolling drawdowns in the sample — used to verify
    the hand-picked scenario windows line up with the largest real
    drawdowns in this portfolio's own history.
    """
    cum = returns.cumsum()
    rolling_peak = cum.rolling(window, min_periods=1).max()
    drawdown = cum - rolling_peak
    return drawdown.nsmallest(n_windows)