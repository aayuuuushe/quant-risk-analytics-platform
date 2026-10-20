"""Historical scenario replay and hypothetical shocks."""
import numpy as np
import pandas as pd

from src.config import PRE_CRISIS_WINDOW, MIN_PRE_CRISIS_OBS

SCENARIOS = {
    'COVID Crash':     ('2020-02-20', '2020-03-23'),
    '2018 Selloff':    ('2018-10-01', '2018-12-31'),
    '2022 Rate Shock': ('2022-01-01', '2022-10-31'),
    'China Selloff':   ('2015-08-01', '2015-09-30'),
}


def pre_crisis_history(returns: pd.Series, crisis_start: str,
                       window: int = PRE_CRISIS_WINDOW) -> pd.Series:
    """The returns a risk manager could see on the day before the crisis began."""
    cutoff = pd.Timestamp(crisis_start) - pd.Timedelta(days=1)
    return returns.loc[:cutoff].tail(window)


def pre_crisis_var(returns: pd.Series, crisis_start: str, confidence: float,
                   window: int = PRE_CRISIS_WINDOW) -> float:
    """VaR from the days before the crisis. NaN if there are too few days to
    estimate it (instead of crashing or giving a meaningless number)."""
    history = pre_crisis_history(returns, crisis_start, window)
    if len(history) < MIN_PRE_CRISIS_OBS:
        return float('nan')
    return float(-np.percentile(history, (1 - confidence) * 100))


def stress_scenarios(portfolio_returns: pd.Series, scenarios: dict = None) -> dict:
    scenarios = scenarios or SCENARIOS
    results = {}
    for name, (start, end) in scenarios.items():
        period = portfolio_returns.loc[start:end]
        if len(period) < 5:
            continue
        pre_var = pre_crisis_var(portfolio_returns, start, 0.99)
        n_pre = len(pre_crisis_history(portfolio_returns, start))
        breaches = int((-period > pre_var).sum()) if np.isfinite(pre_var) else None
        results[name] = {
            'returns': period,
            'total': float(((1 + period).prod() - 1) * 100),   # compounded
            'worst_day': float(period.min() * 100),
            'pre_var_99': pre_var,
            'pre_obs': n_pre,
            'n_days': len(period),
            'breaches': breaches,
            'expected_breaches': len(period) * 0.01,
        }
    return results


def worst_drawdown_windows(returns: pd.Series, n_windows: int = 5) -> pd.DataFrame:
    """The worst DISTINCT peak-to-trough episodes of the portfolio's own history
    (one row per episode), used as a cross-check of the scenario windows."""
    wealth = (1 + returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    under = dd < 0
    episode = (~under).cumsum()
    rows = []
    for _, grp in dd[under].groupby(episode[under]):
        first_pos = wealth.index.get_loc(grp.index[0])
        last_pos = wealth.index.get_loc(grp.index[-1])
        rows.append({
            'peak_date': wealth.index[first_pos - 1].date(),
            'trough_date': grp.idxmin().date(),
            'recovery_date': (wealth.index[last_pos + 1].date()
                              if last_pos + 1 < len(wealth) else None),
            'max_drawdown_pct': round(float(grp.min() * 100), 2),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values('max_drawdown_pct').head(n_windows).reset_index(drop=True)
