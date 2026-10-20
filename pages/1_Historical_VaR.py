import streamlit as st
from ui_helpers import inject_theme, render_sidebar
from src.risk.var_models import historical_var, historical_es
from src.risk.metrics import diversification_benefit, individual_historical_vars
from src.visualization.charts import return_distribution_chart, diversification_bar_chart, correlation_heatmap

st.set_page_config(page_title="Historical VaR", page_icon="📊", layout="wide")
inject_theme()
cfg = render_sidebar()
log_returns, port_ret, w, pv = (
    cfg["log_returns"],
    cfg["portfolio_returns"],
    cfg["w"],
    cfg["portfolio_value"],
)

st.markdown("## Historical Simulation VaR")
st.caption("No distributional assumption — VaR is read directly off the empirical return distribution.")

var_95, var_99 = historical_var(port_ret, 0.95), historical_var(port_ret, 0.99)
es_95, es_99 = historical_es(port_ret, 0.95), historical_es(port_ret, 0.99)

col1, col2, col3, col4 = st.columns(4)
col1.metric("VaR (95%)", f"${var_95*pv:,.0f}", f"{var_95*100:.2f}%")
col2.metric("ES (95%)", f"${es_95*pv:,.0f}", f"{es_95*100:.2f}%")
col3.metric("VaR (99%)", f"${var_99*pv:,.0f}", f"{var_99*100:.2f}%")
col4.metric("ES (99%)", f"${es_99*pv:,.0f}", f"{es_99*100:.2f}%")

st.plotly_chart(return_distribution_chart(port_ret, var_95, var_99), width="stretch")


st.markdown("### Diversification Benefit")
div_benefit = diversification_benefit(log_returns, w, 0.95)
tickers = log_returns.columns.tolist()
individual_vars = individual_historical_vars(log_returns, 0.95)
weighted_avg = sum(wi * v for wi, v in zip(w, individual_vars))

st.metric("Diversification Benefit", f"${div_benefit*pv:,.0f}/day",
          f"{div_benefit*100:.2f}% of portfolio value")
st.plotly_chart(
    diversification_bar_chart(
        tickers,
        individual_vars,
        weighted_avg,
        var_95,
    ),
    width="stretch",
)
st.plotly_chart(correlation_heatmap(log_returns), width="stretch")