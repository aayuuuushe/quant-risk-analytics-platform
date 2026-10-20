"""Quant Risk Analytics Platform — Home / Overview page."""
import streamlit as st
import numpy as np
import pandas as pd

from ui_helpers import inject_theme, render_sidebar, cached_backtests
from src.risk.var_models import (
    historical_var, parametric_var, monte_carlo_var, ewma_var_current,
)
from src.visualization.charts import cumulative_performance_chart, var_comparison_chart
from src.reporting.report_generator import (
    generate_report_data, render_pdf_report, render_json_report,
)

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

h_var_95, h_var_99 = historical_var(port_ret, 0.95), historical_var(port_ret, 0.99)
p_var_95, p_var_99 = parametric_var(log_returns, w, 0.95), parametric_var(log_returns, w, 0.99)
mc_var_95 = monte_carlo_var(log_returns, w, 0.95, n_sim)
mc_var_99 = monte_carlo_var(log_returns, w, 0.99, n_sim)
_, ewma_95, _ = ewma_var_current(port_ret, 0.95)
_, ewma_99, _ = ewma_var_current(port_ret, 0.99)

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

st.info(
    f"**In plain words:** on a normal bad day (about 1 in 20) this portfolio is expected to lose "
    f"roughly **${h_var_95*pv:,.0f}** ({h_var_95*100:.2f}%) based on its history; on a really bad "
    f"day (about 1 in 100), roughly **${h_var_99*pv:,.0f}** ({h_var_99*100:.2f}%). "
    f"These limits are called Value at Risk (VaR). The pages on the left show four ways of "
    f"calculating them, check how well each one worked in the past, and replay real crises."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Portfolio Value", f"${pv:,.0f}")
    st.metric("Daily Volatility", f"{port_ret.std()*100:.2f}%")
with col2:
    st.metric("Historical VaR (95% / 99%)", f"${h_var_95*pv:,.0f}", f"{h_var_95*100:.2f}%")
    st.caption(f"99%: ${h_var_99*pv:,.0f} ({h_var_99*100:.2f}%)")
    st.metric("Parametric VaR (95% / 99%)", f"${p_var_95*pv:,.0f}", f"{p_var_95*100:.2f}%")
    st.caption(f"99%: ${p_var_99*pv:,.0f} ({p_var_99*100:.2f}%)")
with col3:
    st.metric("Monte Carlo VaR (95% / 99%)", f"${mc_var_95*pv:,.0f}", f"{mc_var_95*100:.2f}%")
    st.caption(f"99%: ${mc_var_99*pv:,.0f} ({mc_var_99*100:.2f}%)")
    st.metric("EWMA VaR (95% / 99%)", f"${ewma_95*pv:,.0f}", f"{ewma_95*100:.2f}%")
    st.caption(f"99%: ${ewma_99*pv:,.0f} ({ewma_99*100:.2f}%)")
st.caption("Historical, Parametric and Monte Carlo use the whole data period. EWMA uses recent "
           "volatility, so it reflects today's conditions rather than the long-run average.")

st.markdown("---")
st.markdown("#### Cumulative Performance")
st.plotly_chart(cumulative_performance_chart(log_returns), width="stretch")

st.markdown("---")
st.markdown("#### VaR Comparison — All Methods")
st.plotly_chart(
    var_comparison_chart(
        ["Historical", "Parametric", "Monte Carlo", "EWMA"],
        [h_var_95 * 100, p_var_95 * 100, mc_var_95 * 100, ewma_95 * 100],
        [h_var_99 * 100, p_var_99 * 100, mc_var_99 * 100, ewma_99 * 100],
    ),
    width="stretch",
)

st.markdown("---")
st.markdown("#### Model Validation — Live Backtest Results")
st.caption("Computed live for the current asset/date selection, 95% and 99%, all methods on the same dates.")

rows = []
for cl_val in (0.95, 0.99):
    for method, result in cached_backtests(port_ret, cl_val).items():
        k, c = result["kupiec"], result["christoffersen"]
        rows.append({
            "Method": method, "CL": f"{cl_val*100:.0f}%",
            "Expected bad days": round(k["expected"], 1), "Actual": k["n_breaches"],
            "Kupiec p": round(k["p_value"], 4), "Christoffersen p": round(c["p_value"], 4),
            "Verdict": result["verdict"],
        })
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.caption("PASS = right number of bad days and not bunched; CLUSTERED = right number but bunched; "
           "FAIL = wrong number.")

st.markdown("---")


@st.cache_data(show_spinner="Building report...")
def _build_reports(log_returns, port_ret, w, tickers, weights, pv, n_sim, start, end, dq):
    data = generate_report_data(log_returns, port_ret, w, tickers, weights, pv,
                                n_sim, start, end, data_quality=dq)
    return render_pdf_report(data), render_json_report(data)


pdf_bytes, json_report = _build_reports(
    log_returns, port_ret, w, cfg["tickers"], cfg["weights"], pv, n_sim,
    cfg["start_date"], cfg["end_date"], cfg["data_quality"])

col_a, col_b = st.columns(2)
col_a.download_button("⬇ Download Full Report (PDF)", data=pdf_bytes,
                      file_name="risk_report.pdf", mime="application/pdf")
col_b.download_button("⬇ Download Full Report (JSON)", data=json_report,
                      file_name="risk_report.json", mime="application/json")
