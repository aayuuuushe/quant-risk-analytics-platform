import streamlit as st
from ui_helpers import inject_theme, render_sidebar
from src.risk.var_models import ewma_var_current, ewma_rolling_var
from src.visualization.charts import ewma_volatility_chart, ewma_var_chart

st.set_page_config(page_title="EWMA VaR", page_icon="📊", layout="wide")
inject_theme()
cfg = render_sidebar()
port_ret, pv = cfg["portfolio_returns"], cfg["portfolio_value"]

st.markdown("## EWMA (RiskMetrics) VaR")
st.caption("Exponentially weighted volatility (λ=0.94) — recent moves carry more weight, so the VaR estimate tracks current market conditions rather than a fixed historical average.")

vol, var_95, es_95 = ewma_var_current(port_ret, 0.95)
_, var_99, es_99 = ewma_var_current(port_ret, 0.99)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current EWMA Volatility", f"{vol*100:.2f}%")
col2.metric("EWMA VaR (95%)", f"${var_95*pv:,.0f}", f"{var_95*100:.2f}%")
col3.metric("EWMA VaR (99%)", f"${var_99*pv:,.0f}", f"{var_99*100:.2f}%")
col4.metric("ES (99%)", f"${es_99*pv:,.0f}", f"{es_99*100:.2f}%")

ewma_vol_series, rolling_var_95 = ewma_rolling_var(port_ret, 0.95)
_, rolling_var_99 = ewma_rolling_var(port_ret, 0.99)

st.plotly_chart(ewma_volatility_chart(port_ret.index, ewma_vol_series), width="stretch")
st.plotly_chart(ewma_var_chart(port_ret.index, rolling_var_95, rolling_var_99), width="stretch")