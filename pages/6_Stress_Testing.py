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
    pre = data["pre_var_99"]
    ratio = abs(data["worst_day"]) / 100 / pre if pre == pre else float("nan")   # nan-safe
    rows.append({
        "Scenario": name,
        "Total Return": f"{data['total']:.2f}%",
        "Worst Day": f"{data['worst_day']:.2f}%",
        "Pre-Crisis VaR99": f"{pre*100:.2f}%" if pre == pre else "n/a",
        "Worst / VaR99": f"{ratio:.1f}x" if ratio == ratio else "n/a",
        "Days of history used": data["pre_obs"],
        "VaR breaches (expected)": (f"{data['breaches']} ({data['expected_breaches']:.1f})"
                                    if data["breaches"] is not None else "n/a"),
    })
    if data["pre_obs"] < 250:
        st.warning(f"{name}: the pre-crisis benchmark uses only {data['pre_obs']} days of "
                   f"history instead of 250, so it is less reliable.")
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.plotly_chart(stress_cumulative_chart(results), width="stretch")

shocks = {'Market Correction  −10%': -0.10, 'Severe Drawdown    −20%': -0.20,
          'Crisis Scenario    −30%': -0.30, 'Extreme Tail Event −40%': -0.40}
shock_losses = {name: abs(pct) * pv for name, pct in shocks.items()}
st.plotly_chart(shock_bar_chart(shock_losses, pv), width="stretch")

with st.expander("Cross-check: the portfolio's own worst drawdown episodes"):
    st.caption("The four scenario windows are well-known stress periods chosen by date. "
               "This table shows the worst peak-to-trough episodes in the portfolio's own "
               "history, for comparison.")
    st.dataframe(worst_drawdown_windows(port_ret), hide_index=True)