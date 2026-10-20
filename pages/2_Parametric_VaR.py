import streamlit as st
from ui_helpers import inject_theme, render_sidebar
from src.risk.var_models import (
    historical_var, parametric_var, parametric_es, portfolio_mu, portfolio_sigma
)
from src.visualization.charts import parametric_overlay_chart

st.set_page_config(page_title="Parametric VaR", page_icon="📊", layout="wide")
inject_theme()
cfg = render_sidebar()
log_returns, port_ret, w, pv = (
    cfg["log_returns"], cfg["portfolio_returns"], cfg["w"], cfg["portfolio_value"],
)

st.markdown("## Parametric (Delta-Normal) VaR")
st.caption("Assumes portfolio returns are normally distributed; VaR is computed analytically from mean and volatility.")

mu_p, sigma_p = portfolio_mu(log_returns, w), portfolio_sigma(log_returns, w)

p_var_95 = parametric_var(log_returns, w, 0.95)
p_var_99 = parametric_var(log_returns, w, 0.99)
p_es_95 = parametric_es(log_returns, w, 0.95)
p_es_99 = parametric_es(log_returns, w, 0.99)
h_var_95 = historical_var(port_ret, 0.95)
h_var_99 = historical_var(port_ret, 0.99)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Parametric VaR (95%)", f"${p_var_95*pv:,.0f}", f"{p_var_95*100:.2f}%")
col2.metric("ES (95%)", f"${p_es_95*pv:,.0f}", f"{p_es_95*100:.2f}%")
col3.metric("Parametric VaR (99%)", f"${p_var_99*pv:,.0f}", f"{p_var_99*100:.2f}%")
col4.metric("ES (99%)", f"${p_es_99*pv:,.0f}", f"{p_es_99*100:.2f}%")

st.plotly_chart(
    parametric_overlay_chart(port_ret, mu_p, sigma_p, p_var_95, p_var_99, h_var_95, h_var_99),
    width="stretch"
)