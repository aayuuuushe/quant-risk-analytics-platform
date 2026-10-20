import streamlit as st
import pandas as pd
from ui_helpers import inject_theme, render_sidebar, cached_backtests
from src.visualization.charts import backtest_chart

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")
inject_theme()
cfg = render_sidebar()
port_ret = cfg["portfolio_returns"]

st.markdown("## Model Validation — Backtesting")
st.caption(
    "Each day the VaR is forecast using only data available the day before (no look-ahead), "
    "then compared with what really happened. Kupiec tests the NUMBER of bad days; "
    "Christoffersen tests whether bad days are spread out or bunched together; "
    "all methods are tested on exactly the same dates."
)

results_95 = cached_backtests(port_ret, 0.95)
results_99 = cached_backtests(port_ret, 0.99)

method = st.selectbox("Method to inspect", list(results_95.keys()))
level = st.radio("Confidence level to inspect", [0.95, 0.99],
                 horizontal=True, format_func=lambda x: f"{x*100:.0f}%")
r = (results_95 if level == 0.95 else results_99)[method]
k, c = r["kupiec"], r["christoffersen"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Kupiec (number of bad days)", "PASS" if k["passed"] else "FAIL")
col1.caption(f"p-value {k['p_value']:.3f}")
col2.metric("Christoffersen (bunching)", "PASS" if c["passed"] else "FAIL")
col2.caption(f"p-value {c['p_value']:.3f}")
col3.metric("Bad days: actual vs expected", f"{k['n_breaches']} vs {k['expected']:.1f}")
col3.caption(f"out of {k['n_obs']} forecasts")
col4.metric("Verdict", r["verdict"])
if "es_check" in r and r["es_check"]["n_breaches"] > 0:
    col4.caption(f"On bad days, actual loss was {r['es_check']['ratio']:.2f}x the forecast ES")

st.plotly_chart(
    backtest_chart(r["var_series"].index, r["returns"], r["var_series"],
                   r["breach_dates"], r["returns"].loc[r["breach_dates"]], level),
    width="stretch"
)

st.markdown("### All Methods — Both Confidence Levels")
rows = []
for cl_val, results_set in [(0.95, results_95), (0.99, results_99)]:
    for m, res in results_set.items():
        kk, cc, cv = res["kupiec"], res["christoffersen"], res["conditional_coverage"]
        rows.append({
            "Method": m, "CL": f"{cl_val*100:.0f}%",
            "Expected bad days": round(kk["expected"], 1), "Actual": kk["n_breaches"],
            "Kupiec p": round(kk["p_value"], 4),
            "Christoffersen p": round(cc["p_value"], 4),
            "Combined p": round(cv["p_value"], 4),
            "Verdict": res["verdict"],
        })
table = pd.DataFrame(rows)
st.dataframe(table.set_index(["Method", "CL"]), width="stretch")
st.caption("PASS = right number of bad days and not bunched; CLUSTERED = right number but bunched; "
           "FAIL = wrong number. Monte Carlo is not listed: under a bell-curve model it is the same "
           "as Parametric.")
