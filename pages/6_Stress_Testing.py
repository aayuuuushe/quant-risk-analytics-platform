import pandas as pd
import streamlit as st
from ui_helpers import inject_theme, render_sidebar
from src.scenarios.stress_testing import stress_scenarios, worst_drawdown_windows
from src.visualization.charts import stress_cumulative_chart, shock_bar_chart

st.set_page_config(page_title="Stress Testing", page_icon="📊", layout="wide")
inject_theme()
cfg = render_sidebar()
port_ret, pv = cfg["portfolio_returns"], cfg["portfolio_value"]

st.markdown("## Stress Testing")
st.caption(
    "VaR is calibrated to typical days. Stress testing replays real crisis "
    "periods and asks what this portfolio actually lost, benchmarked against "
    "each scenario's OWN pre-crisis VaR (not a full-sample figure, which would "
    "be circular since it includes the crisis itself)."
)

results = stress_scenarios(port_ret)
rows = []
for name, data in results.items():
    worst = abs(data["worst_day"]) / 100
    ratio = worst / data["pre_var_99"] if data["pre_var_99"] else float("nan")
    rows.append({
        "Scenario": name,
        "Total Return": f"{data['total']:.2f}%",
        "Worst Day": f"{data['worst_day']:.2f}%",
        "Pre-Crisis VaR99": f"{data['pre_var_99']*100:.2f}%",
        "Worst / VaR99": f"{ratio:.1f}x",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.plotly_chart(stress_cumulative_chart(results), use_container_width=True)

shocks = {'Market Correction  −10%': -0.10, 'Severe Drawdown    −20%': -0.20,
          'Crisis Scenario    −30%': -0.30, 'Extreme Tail Event −40%': -0.40}
shock_losses = {name: abs(pct) * pv for name, pct in shocks.items()}
st.plotly_chart(shock_bar_chart(shock_losses, pv), use_container_width=True)

with st.expander("How were these scenario windows chosen?"):
    st.write(worst_drawdown_windows(port_ret))