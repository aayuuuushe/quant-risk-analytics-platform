import streamlit as st
import plotly.graph_objects as go
from ui_helpers import inject_theme, render_sidebar
from src.risk.var_models import monte_carlo_full, parametric_var

st.set_page_config(page_title="Monte Carlo VaR", page_icon="📊", layout="wide")
inject_theme()
cfg = render_sidebar()
log_returns, w, pv, n_sim = (
    cfg["log_returns"], cfg["w"], cfg["portfolio_value"], cfg["n_simulations"],
)

st.markdown("## Monte Carlo VaR")
st.caption(
    f"{n_sim:,} multivariate-normal simulations using the historical covariance "
    f"structure. Under this normality assumption it converges to the Parametric "
    f"result — the value of the Monte Carlo framework is that it extends "
    f"unchanged to non-normal return models."
)

result_95 = monte_carlo_full(log_returns, w, 0.95, n_sim)
result_99 = monte_carlo_full(log_returns, w, 0.99, n_sim)
p_var_95 = parametric_var(log_returns, w, 0.95)
p_var_99 = parametric_var(log_returns, w, 0.99)

col1, col2, col3, col4 = st.columns(4)
col1.metric("MC VaR (95%)", f"${result_95['var']*pv:,.0f}", f"{result_95['var']*100:.2f}%")
col2.metric("ES (95%)", f"${result_95['es']*pv:,.0f}", f"{result_95['es']*100:.2f}%")
col3.metric("MC VaR (99%)", f"${result_99['var']*pv:,.0f}", f"{result_99['var']*100:.2f}%")
col4.metric("ES (99%)", f"${result_99['es']*pv:,.0f}", f"{result_99['es']*100:.2f}%")

fig = go.Figure()
fig.add_trace(go.Histogram(x=result_95['simulated']*100, nbinsx=100, marker_color="#00A3FF", opacity=0.7))
fig.add_vline(x=-result_95['var']*100, line_color="#FFB347", line_dash="dash", annotation_text="95% VaR")
fig.add_vline(x=-result_99['var']*100, line_color="#FF4444", line_dash="dash", annotation_text="99% VaR")
fig.update_layout(paper_bgcolor="#1E2130", plot_bgcolor="#1E2130",
                   font=dict(color="#8B9DC3"), title="Simulated Portfolio Return Distribution")
st.plotly_chart(fig, width="stretch")