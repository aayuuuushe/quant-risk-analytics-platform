"""Shared Streamlit presentation layer: theming, sidebar controls, session
state. No risk calculations happen here — see src/ for all business logic."""
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_TICKERS, DEFAULT_PORTFOLIO_VALUE, DEFAULT_CONFIDENCE,
    DEFAULT_START_DATE, DEFAULT_END_DATE, MC_SIMULATION_OPTIONS,
    DEFAULT_MC_SIMULATIONS, THEME,
)
from src.data.loader import load_data
from src.risk.var_models import portfolio_returns_series

PLATFORM_CSS = f"""
<style>
.stApp {{ background-color: {THEME['bg']}; }}
[data-testid="stSidebar"] {{
    background-color: {THEME['sidebar_bg']};
    border-right: 1px solid {THEME['border']};
}}
[data-testid="metric-container"] {{
    background-color: {THEME['card_bg']};
    border: 1px solid {THEME['border']};
    padding: 16px;
    border-radius: 4px;
}}
[data-testid="metric-container"] label {{
    color: {THEME['text_muted']} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: #FFFFFF !important;
    font-size: 22px !important;
    font-weight: 600;
}}
h1, h2, h3 {{ color: #FFFFFF; }}
hr {{ border-color: {THEME['border']}; }}
.stTabs [data-baseweb="tab"] {{ color: {THEME['text_muted']}; }}
.stTabs [aria-selected="true"] {{
    color: {THEME['accent']} !important;
    border-bottom: 2px solid {THEME['accent']} !important;
}}
</style>
"""


def inject_theme() -> None:
    st.markdown(PLATFORM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading market data...")
def _get_data_cached(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    return load_data(list(tickers), start, end)


def render_sidebar() -> dict:
    """Renders portfolio controls, loads data, computes the portfolio return
    series, and returns/stores everything downstream pages need."""
    with st.sidebar:
        st.markdown(
            f"<div style='font-family:monospace; font-size:13px; "
            f"font-weight:700; color:{THEME['accent']}; letter-spacing:0.15em; "
            f"text-transform:uppercase; padding-bottom:8px; "
            f"border-bottom:1px solid {THEME['border']}; margin-bottom:16px;'>"
            f"⬛ QUANT RISK PLATFORM</div>",
            unsafe_allow_html=True
        )

        st.markdown("**Portfolio Settings**")

        portfolio_value = st.number_input(
            "Portfolio Value ($)", min_value=100_000, max_value=100_000_000,
            value=DEFAULT_PORTFOLIO_VALUE, step=100_000
        )

        selected_tickers = st.multiselect(
            "Assets", options=DEFAULT_TICKERS, default=DEFAULT_TICKERS
        )
        if not selected_tickers:
            st.warning("Select at least one asset")
            st.stop()

        st.markdown("**Weights**")
        n = len(selected_tickers)
        default_w = round(1.0 / n, 2)
        raw_weights = {
            t: st.number_input(t, min_value=0.0, max_value=1.0,
                                value=default_w, step=0.05, key=f"w_{t}")
            for t in selected_tickers
        }

        total_w = sum(raw_weights.values())
        if total_w <= 0:
            st.error("Weights must sum to a positive number.")
            st.stop()
        if abs(total_w - 1.0) > 0.01:
            st.info(f"Weights sum to {total_w:.2f} — normalized to 100% for all calculations below.")
        weights = {t: v / total_w for t, v in raw_weights.items()}

        n_simulations = st.selectbox(
            "Monte Carlo Simulations", options=MC_SIMULATION_OPTIONS,
            index=MC_SIMULATION_OPTIONS.index(DEFAULT_MC_SIMULATIONS)
        )

        start_date = st.date_input("Start Date", value=pd.Timestamp(DEFAULT_START_DATE))
        end_date = st.date_input("End Date", value=pd.Timestamp(DEFAULT_END_DATE))
        st.markdown("---")

    log_returns = _get_data_cached(tuple(selected_tickers), str(start_date), str(end_date))

    MIN_OBS = 260  # EWMA/backtest burn-in (250) plus a small buffer
    if len(log_returns) < MIN_OBS:
        st.error(
            f"Selected date range has only {len(log_returns)} trading days. "
            f"At least {MIN_OBS} are required for EWMA and backtesting calculations — "
            f"please widen the date range."
        )
        st.stop()

    w = np.array([weights[t] for t in selected_tickers])
    portfolio_returns = portfolio_returns_series(log_returns, w)

    cfg = dict(
        portfolio_value=portfolio_value,
        tickers=selected_tickers, weights=weights, w=w,
        n_simulations=n_simulations, start_date=str(start_date), end_date=str(end_date),
        log_returns=log_returns, portfolio_returns=portfolio_returns,)
    st.session_state.update(cfg)
    return cfg