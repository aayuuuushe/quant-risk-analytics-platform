import streamlit as st
import pandas as pd
from ui_helpers import inject_theme, render_sidebar
from src.validation.backtesting import backtest_all_methods
from src.visualization.charts import backtest_chart

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")
inject_theme()
cfg = render_sidebar()
port_ret = cfg["portfolio_returns"]

st.markdown("## Model Validation — Backtesting")
st.caption(
    "Rolling 250-day VaR, re-estimated daily using only data available up to "
    "that point (no look-ahead). Kupiec tests breach FREQUENCY; Christoffersen "
    "tests whether breaches cluster (independence)."
)

results_95 = backtest_all_methods(port_ret, 0.95)
results_99 = backtest_all_methods(port_ret, 0.99)

method = st.selectbox("Method to inspect", list(results_95.keys()))
level = st.radio("Confidence level to inspect", [0.95, 0.99],
                  horizontal=True, format_func=lambda x: f"{x*100:.0f}%")
r = (results_95 if level == 0.95 else results_99)[method]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Kupiec Result", "PASS" if r["kupiec"]["passed"] else "FAIL")
col1.caption(f"p-value: {r['kupiec']['p_value']:.3f}" if pd.notna(r['kupiec']['p_value']) else "p-value: N/A")
col2.metric("Christoffersen Result", "PASS" if r["christoffersen"]["passed"] else "FAIL")
col3.metric("Breaches", f"{len(r['breach_dates'])} / {len(r['var_series'])}")
col4.metric("Verdict", r["verdict"])

st.plotly_chart(
    backtest_chart(r["var_series"].index, r["returns"], r["var_series"],
                    r["breach_dates"], r["returns"].loc[r["breach_dates"]], level),
    width="stretch"
)

st.markdown("### All Methods — Both Confidence Levels")
rows = []
for cl_val, results_set in [(0.95, results_95), (0.99, results_99)]:
    for m, res in results_set.items():
        rows.append({
            "Method": m, "CL": f"{cl_val*100:.0f}%",
            "Kupiec": "PASS" if res["kupiec"]["passed"] else "FAIL",
            "Christoffersen": "PASS" if res["christoffersen"]["passed"] else "FAIL",
            "Breaches": len(res["breach_dates"]),
            "Verdict": res["verdict"],
        })
st.dataframe(pd.DataFrame(rows).set_index(["Method", "CL"]),
             width="stretch")