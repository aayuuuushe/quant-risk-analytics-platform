"""Quant Risk Analytics Platform — Home / Overview page."""
import streamlit as st
import numpy as np
import pandas as pd

from ui_helpers import inject_theme, render_sidebar
from src.risk.var_models import (
    historical_var, parametric_var, monte_carlo_var, ewma_var_current,
)
from src.validation.backtesting import backtest_all_methods
from src.visualization.charts import cumulative_performance_chart, var_comparison_chart

st.set_page_config(
    page_title="Quant Risk Platform", page_icon="📊",
    layout="wide", initial_sidebar_state="expanded"
)
inject_theme()

cfg = render_sidebar()

log_returns = cfg["log_returns"]
port_ret = cfg["portfolio_returns"]
w = cfg["w"]
pv = cfg["portfolio_value"]
n_sim = cfg["n_simulations"]

h_var_95 = historical_var(port_ret, 0.95)
h_var_99 = historical_var(port_ret, 0.99)

p_var_95 = parametric_var(log_returns, w, 0.95)
p_var_99 = parametric_var(log_returns, w, 0.99)

mc_var_95 = monte_carlo_var(log_returns, w, 0.95, n_sim)
mc_var_99 = monte_carlo_var(log_returns, w, 0.99, n_sim)

ewma_vol_95, ewma_95, _ = ewma_var_current(port_ret, 0.95)
ewma_vol_99, ewma_99, _ = ewma_var_current(port_ret, 0.99)

with st.sidebar:
    st.markdown("**Live Risk Summary**")
    st.metric("Daily Volatility", f"{port_ret.std()*100:.2f}%")
    st.metric("95% Historical VaR", f"${h_var_95*pv:,.0f}")
    st.metric("99% Historical VaR", f"${h_var_99*pv:,.0f}")
st.markdown("# QUANT RISK ANALYTICS PLATFORM")
st.markdown(
    f"<p style='color:#8B9DC3; font-size:13px; margin-top:-10px;'>"
    f"Market Risk Modelling · {len(cfg['tickers'])}-Asset Equity Portfolio · "
    f"{cfg['start_date']} – {cfg['end_date']} · ${pv:,.0f}</p>",
    unsafe_allow_html=True
)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Portfolio Value", f"${pv:,.0f}")
    st.metric("Daily Volatility", f"{port_ret.std()*100:.2f}%")

with col2:
    st.metric(
        "Historical VaR (95% / 99%)",
        f"${h_var_95*pv:,.0f}",
        f"{h_var_95*100:.2f}%"
    )
    st.caption(
        f"99%: ${h_var_99*pv:,.0f} ({h_var_99*100:.2f}%)"
    )

    st.metric(
        "Parametric VaR (95% / 99%)",
        f"${p_var_95*pv:,.0f}",
        f"{p_var_95*100:.2f}%"
    )
    st.caption(
        f"99%: ${p_var_99*pv:,.0f} ({p_var_99*100:.2f}%)"
    )

with col3:
    st.metric(
        "Monte Carlo VaR (95% / 99%)",
        f"${mc_var_95*pv:,.0f}",
        f"{mc_var_95*100:.2f}%"
    )
    st.caption(
        f"99%: ${mc_var_99*pv:,.0f} ({mc_var_99*100:.2f}%)"
    )

    st.metric(
        "EWMA VaR (95% / 99%)",
        f"${ewma_95*pv:,.0f}",
        f"{ewma_95*100:.2f}%"
    )
    st.caption(
        f"99%: ${ewma_99*pv:,.0f} ({ewma_99*100:.2f}%)"
    )
st.markdown("---")
st.markdown("#### Cumulative Performance")
st.plotly_chart(cumulative_performance_chart(log_returns), use_container_width=True)

st.markdown("---")
st.markdown("#### VaR Comparison — All Methods")

st.plotly_chart(
    var_comparison_chart(
        ["Historical", "Parametric", "Monte Carlo", "EWMA"],
        [
            h_var_95 * 100,
            p_var_95 * 100,
            mc_var_95 * 100,
            ewma_95 * 100,
        ],
        [
            h_var_99 * 100,
            p_var_99 * 100,
            mc_var_99 * 100,
            ewma_99 * 100,
        ],
    ),
    use_container_width=True,
)

st.markdown("---")
st.markdown("#### Model Validation — Live Backtest Results")
st.caption(
    "Computed live for both 95% and 99% rolling VaR over the current "
    "asset/date selection — not a static table.")

backtests_95 = backtest_all_methods(port_ret, 0.95)
backtests_99 = backtest_all_methods(port_ret, 0.99)

summary_rows = []

for cl_val, backtests in [
    (0.95, backtests_95),
    (0.99, backtests_99),
]:
    for method, result in backtests.items():
        kup = result["kupiec"]
        chris = result["christoffersen"]

        summary_rows.append(
            {
                "Method": method,
                "CL": f"{cl_val*100:.0f}%",
                "Backtest": "PASS" if kup["passed"] else "FAIL",
                "Kupiec p-value": (
                    f"{kup['p_value']:.3f}"
                    if pd.notna(kup["p_value"])
                    else "N/A"
                ),
                "Christoffersen": (
                    "PASS" if chris["passed"] else "FAIL"
                ),
                "Verdict": result["verdict"],
            }
        )

st.dataframe(
    pd.DataFrame(summary_rows),
    use_container_width=True,
    hide_index=True,
)

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.markdown("---")
summary_df = pd.DataFrame(
    {
        "Method": [
            "Historical",
            "Parametric",
            "Monte Carlo",
            "EWMA",
        ],
        "VaR 95% (%)": [
            f"{h_var_95*100:.4f}%",
            f"{p_var_95*100:.4f}%",
            f"{mc_var_95*100:.4f}%",
            f"{ewma_95*100:.4f}%",
        ],
        "VaR 99% (%)": [
            f"{h_var_99*100:.4f}%",
            f"{p_var_99*100:.4f}%",
            f"{mc_var_99*100:.4f}%",
            f"{ewma_99*100:.4f}%",
        ],
        "VaR 95% ($)": [
            f"${h_var_95*pv:,.0f}",
            f"${p_var_95*pv:,.0f}",
            f"${mc_var_95*pv:,.0f}",
            f"${ewma_95*pv:,.0f}",
        ],
        "VaR 99% ($)": [
            f"${h_var_99*pv:,.0f}",
            f"${p_var_99*pv:,.0f}",
            f"${mc_var_99*pv:,.0f}",
            f"${ewma_99*pv:,.0f}",
        ],
    }
)
st.download_button(
    "⬇ Download VaR Summary (CSV)", data=summary_df.to_csv(index=False),
    file_name="var_summary.csv", mime="text/csv"
)