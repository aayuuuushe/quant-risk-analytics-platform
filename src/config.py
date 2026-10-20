"""Centralized configuration — single source of truth for default portfolio
parameters and model hyperparameters used across notebooks, src, and the app."""

DEFAULT_TICKERS = ['AAPL', 'JPM', 'XOM', 'JNJ', 'GS']
DEFAULT_WEIGHTS = {t: 0.20 for t in DEFAULT_TICKERS}

DEFAULT_PORTFOLIO_VALUE = 1_000_000
DEFAULT_CONFIDENCE = 0.95

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2024-12-31"

MC_SIMULATION_OPTIONS = [1_000, 5_000, 10_000, 50_000]
DEFAULT_MC_SIMULATIONS = 10_000
MC_SEED = 42

EWMA_LAMBDA = 0.94
EWMA_BURN_IN = 250
BACKTEST_WINDOW = 250

# Stress testing: each scenario is judged against a VaR estimated from the
# PRE_CRISIS_WINDOW days before it started. With fewer than MIN_PRE_CRISIS_OBS
# days the benchmark is not computed at all.
PRE_CRISIS_WINDOW = 250
MIN_PRE_CRISIS_OBS = 60

# Data sufficiency (trading days of returns)
HARD_MIN_OBS = 500    # below this the app refuses to run (250 burn-in + 250 tests)
SOFT_MIN_OBS = 1000   # below this the app warns that 99% results are unreliable

THEME = {
    "bg": "#0F1117",
    "sidebar_bg": "#1A1D27",
    "card_bg": "#1E2130",
    "border": "#2A2D3E",
    "text_muted": "#8B9DC3",
    "accent": "#00A3FF",
}
