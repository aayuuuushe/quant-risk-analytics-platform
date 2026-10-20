"""Risk report: every sentence is COMPUTED from the results (nothing hard-coded
about what the numbers 'should' say), then rendered to PDF (reportlab) and JSON.

Fonts: reportlab's built-in fonts only cover basic Latin characters, so the
text below deliberately avoids symbols such as the Greek letter lambda."""
from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape
import hashlib
import json

import numpy as np
import pandas as pd
import scipy
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src.config import (
    BACKTEST_WINDOW, EWMA_BURN_IN, EWMA_LAMBDA, MC_SEED, PRE_CRISIS_WINDOW,
)
from src.data.quality import check_data_quality
from src.risk.metrics import diversification_benefit
from src.risk.var_models import (
    ewma_var_current, historical_es, historical_var, monte_carlo_full,
    parametric_es, parametric_var,
)
from src.scenarios.stress_testing import stress_scenarios
from src.validation.backtesting import backtest_all_methods

MUTED = colors.HexColor("#555555")
METHODS = ["Historical", "Parametric", "Monte Carlo", "EWMA"]


def _usd(x): return f"${x:,.0f}"
def _pct(x, d=2): return f"{x * 100:.{d}f}%"


# =====================================================================
# 1. Data assembly
# =====================================================================
def _fingerprint(log_returns: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(log_returns, index=True).values.tobytes()
    return hashlib.sha256(h).hexdigest()[:12]


def generate_report_data(log_returns, port_ret, w, tickers, weights,
                         portfolio_value, n_simulations, start_date, end_date,
                         data_quality=None) -> dict:
    results = {}
    for cl in (0.95, 0.99):
        mc = monte_carlo_full(log_returns, w, cl, n_simulations)
        _, e_var, e_es = ewma_var_current(port_ret, cl)
        results[cl] = {
            "historical_var": historical_var(port_ret, cl),
            "historical_es": historical_es(port_ret, cl),
            "parametric_var": parametric_var(log_returns, w, cl),
            "parametric_es": parametric_es(log_returns, w, cl),
            "monte_carlo_var": mc["var"], "monte_carlo_es": mc["es"],
            "ewma_var": e_var, "ewma_es": e_es,
        }
    backtests = {cl: backtest_all_methods(port_ret, cl) for cl in (0.95, 0.99)}
    stress = stress_scenarios(port_ret)
    data = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_source": "Daily adjusted closing prices (Yahoo Finance), converted to returns",
        "data_period": f"{start_date} to {end_date}",
        "n_observations": len(port_ret),
        "portfolio": {"tickers": list(tickers), "weights": dict(weights),
                      "value": portfolio_value},
        "portfolio_stats": {
            "daily_mean": float(port_ret.mean()), "daily_vol": float(port_ret.std()),
            "skew": float(port_ret.skew()), "kurtosis": float(port_ret.kurtosis()),
        },
        "results": results, "backtests": backtests, "stress": stress,
        "diversification_benefit": diversification_benefit(log_returns, w, 0.95),
        "data_quality": data_quality or check_data_quality(log_returns),
        "reproducibility": {
            "mc_seed": MC_SEED, "mc_simulations": n_simulations,
            "backtest_window_days": BACKTEST_WINDOW, "ewma_lambda": EWMA_LAMBDA,
            "ewma_burn_in_days": EWMA_BURN_IN, "pre_crisis_window_days": PRE_CRISIS_WINDOW,
            "data_fingerprint": _fingerprint(log_returns),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        },
    }
    data["narrative"] = build_narrative(data)
    return data


# =====================================================================
# 2. Narrative: sentences computed from the numbers
# =====================================================================
def _bt_sentence(method, cl, res):
    k, c = res["kupiec"], res["christoffersen"]
    obs, exp = k["n_breaches"], k["expected"]
    if k["passed"]:
        freq = f"it had about as many bad days as promised ({obs} against {exp:.0f} expected)"
    elif obs > exp:
        freq = (f"it had too many bad days ({obs} against {exp:.0f} expected), "
                f"so it underestimated the risk")
    else:
        freq = (f"it had too few bad days ({obs} against {exp:.0f} expected), "
                f"so it was more cautious than needed")
    clus = ("the bad days were spread out over time" if c["passed"]
            else "the bad days came in bunches instead of being spread out")
    return f"{method} at {cl:.0%}: {freq}, and {clus}."


def _scenario_sentence(name, s):
    parts = []
    if s["total"] < 0:
        parts.append(f"{name}: the portfolio lost {abs(s['total']):.1f}% over the period "
                     f"and its worst single day was {s['worst_day']:.1f}%.")
    else:
        parts.append(f"{name}: the portfolio actually ended {s['total']:.1f}% higher over "
                     f"the period (so this was not a loss event for this portfolio), "
                     f"although its worst single day was {s['worst_day']:.1f}%.")
    if np.isfinite(s["pre_var_99"]):
        ratio = abs(s["worst_day"]) / 100 / s["pre_var_99"]
        parts.append(f"The worst day was {ratio:.1f} times the normal-day 99% risk limit "
                     f"({_pct(s['pre_var_99'])}) that could have been calculated just before.")
        if s["breaches"] is not None:
            parts.append(f"That limit was broken on {s['breaches']} of {s['n_days']} days "
                         f"(about {s['expected_breaches']:.1f} would be expected in calm times).")
    else:
        parts.append("There was too little history before this event to calculate a "
                     "reliable normal-day risk limit.")
    if np.isfinite(s["pre_var_99"]) and s["pre_obs"] < PRE_CRISIS_WINDOW:
        parts.append(f"Caution: that limit uses only {s['pre_obs']} days of history "
                     f"instead of {PRE_CRISIS_WINDOW}.")
    return " ".join(parts)


def build_narrative(d: dict) -> dict:
    pv = d["portfolio"]["value"]
    r95, r99 = d["results"][0.95], d["results"][0.99]
    ps = d["portfolio_stats"]
    bt, stress = d["backtests"], d["stress"]
    n_fc = next(iter(bt[0.95].values()))["kupiec"]["n_obs"]

    v95 = {"Historical": r95["historical_var"], "Parametric": r95["parametric_var"],
           "Monte Carlo": r95["monte_carlo_var"], "EWMA": r95["ewma_var"]}
    v99 = {"Historical": r99["historical_var"], "Parametric": r99["parametric_var"],
           "Monte Carlo": r99["monte_carlo_var"], "EWMA": r99["ewma_var"]}
    lo95, hi95 = min(v95, key=v95.get), max(v95, key=v95.get)
    lo99, hi99 = min(v99, key=v99.get), max(v99, key=v99.get)

    plain = [
        f"The portfolio is worth {_usd(pv)}. On a normal bad day (the kind that happens about "
        f"1 day in 20), historical records say the loss would be about "
        f"{_usd(r95['historical_var'] * pv)}, or {_pct(r95['historical_var'])} of the portfolio. "
        f"On a really bad day (about 1 day in 100) the loss would be about "
        f"{_usd(r99['historical_var'] * pv)} ({_pct(r99['historical_var'])}).",
        f"These 'limits' are called Value at Risk (VaR). They do not say how bad the worst days "
        f"can get. Looking at the days that go beyond the 1-in-100 limit, the historical "
        f"average loss on those days was {_usd(r99['historical_es'] * pv)} "
        f"({_pct(r99['historical_es'])}). This average is called Expected Shortfall (ES).",
        f"Four different calculation methods give different answers. For the 1-in-20 day the "
        f"estimates run from {_usd(v95[lo95] * pv)} ({lo95}) to {_usd(v95[hi95] * pv)} ({hi95}); "
        f"for the 1-in-100 day from {_usd(v99[lo99] * pv)} ({lo99}) to {_usd(v99[hi99] * pv)} ({hi99}). "
        f"Differences come from the assumptions each method makes, not from errors.",
    ]
    if r99["historical_var"] > r99["parametric_var"] and r95["historical_var"] < r95["parametric_var"]:
        plain.append(
            "The bell-curve (Parametric) method gives a higher number than history for the "
            "1-in-20 day but a lower number for the 1-in-100 day. This is the typical sign that "
            "real market moves have 'fatter tails' than a bell curve: extreme days happen more "
            "often than the bell curve expects.")

    # --- did the models work?
    bt_lines, passed_summary = [], []
    for cl in (0.95, 0.99):
        passing = [m for m, r in bt[cl].items() if r["verdict"] == "PASS"]
        n_m = len(bt[cl])
        if passing:
            passed_summary.append(f"At {cl:.0%}, {len(passing)} of {n_m} methods passed both tests "
                                  f"({', '.join(passing)}).")
        else:
            passed_summary.append(f"At {cl:.0%}, none of the {n_m} methods passed both tests.")
        too_many = [m for m, r in bt[cl].items()
                    if not r["kupiec"]["passed"] and r["kupiec"]["n_breaches"] > r["kupiec"]["expected"]]
        if too_many:
            passed_summary.append(f"At {cl:.0%}, {', '.join(too_many)} had significantly more bad "
                                  f"days than promised, meaning the risk limit was set too low.")
        for m, r in bt[cl].items():
            bt_lines.append(_bt_sentence(m, cl, r))
    if (ps["kurtosis"] > 1 and any(not r["kupiec"]["passed"] for r in bt[0.99].values())):
        passed_summary.append(
            f"A likely reason is that daily returns have fat tails (excess kurtosis "
            f"{ps['kurtosis']:.1f}; a bell curve would give 0), so extreme days are more common "
            f"than the simpler models allow for.")

    es_lines = []
    for cl in (0.95, 0.99):
        for m, r in bt[cl].items():
            e = r["es_check"]
            if e["n_breaches"] > 0 and np.isfinite(e["ratio"]):
                es_lines.append((m, cl, e))

    stress_lines = [_scenario_sentence(n, s) for n, s in stress.items()]

    warnings = []
    if d["n_observations"] < 1000:
        warnings.append(f"Only {d['n_observations']} days of data: the 1-in-100 results are unreliable.")
    exp99 = next(iter(bt[0.99].values()))["kupiec"]["expected"]
    if exp99 < 10:
        warnings.append(f"Only about {exp99:.0f} bad days are expected at 99% in the test period, "
                        f"so the 99% backtest has little statistical power.")
    for n, s in stress.items():
        if np.isfinite(s["pre_var_99"]) and s["pre_obs"] < PRE_CRISIS_WINDOW:
            warnings.append(f"{n}: the pre-crisis benchmark uses only {s['pre_obs']} days (not {PRE_CRISIS_WINDOW}).")
    for i in d["data_quality"]["issues"]:
        warnings.append(f"Data check: {i['message']}")
    return {"plain": plain, "backtest_lines": bt_lines, "backtest_summary": passed_summary,
            "es_rows": es_lines, "stress_lines": stress_lines, "warnings": warnings,
            "n_forecasts": n_fc}


# =====================================================================
# 3. JSON
# =====================================================================
def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, (pd.Series, pd.DataFrame, pd.Index, np.ndarray)):
        return None
    return o


def render_json_report(data: dict) -> str:
    def bt(res):
        return {"kupiec": res["kupiec"], "christoffersen": res["christoffersen"],
                "conditional_coverage": res["conditional_coverage"],
                "es_check": res["es_check"], "verdict": res["verdict"],
                "n_breaches": len(res["breach_dates"])}

    def st(s):
        return {k: s[k] for k in ("total", "worst_day", "pre_var_99", "pre_obs",
                                  "n_days", "breaches", "expected_breaches")}
    export = {k: data[k] for k in ("generated_at", "data_source", "data_period",
                                   "n_observations", "portfolio", "portfolio_stats",
                                   "results", "diversification_benefit",
                                   "data_quality", "reproducibility")}
    export["backtests"] = {str(cl): {m: bt(r) for m, r in ms.items()}
                           for cl, ms in data["backtests"].items()}
    export["stress"] = {n: st(s) for n, s in data["stress"].items()}
    nar = dict(data["narrative"])
    nar["es_rows"] = [{"method": m, "confidence": cl, **e} for m, cl, e in nar["es_rows"]]
    export["narrative"] = nar
    return json.dumps(_json_safe(export), indent=2, allow_nan=False)


# =====================================================================
# 4. PDF
# =====================================================================
def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("ReportTitle", parent=s["Title"], fontSize=20, spaceAfter=4))
    s.add(ParagraphStyle("Meta", parent=s["Normal"], fontSize=9, textColor=MUTED, spaceAfter=10))
    s.add(ParagraphStyle("H2", parent=s["Heading2"], fontSize=14, spaceBefore=16, spaceAfter=6))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=10, leading=14, spaceAfter=6))
    s.add(ParagraphStyle("Cell", parent=s["Normal"], fontSize=8.5, leading=11))
    return s


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def _table(rows, header, col_widths=None, styles=None, wrap=False):
    cell = styles["Cell"] if styles else None
    body = [[(_p(c, cell) if wrap else c) for c in r] for r in rows]
    t = Table([header] + body, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A1D27")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F6F8")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


GLOSSARY = [
    ("Value at Risk (VaR)", "A loss limit for one day. At 95% it is the loss the portfolio is expected to exceed on only about 1 day in 20; at 99%, about 1 day in 100. It says nothing about how bad the days beyond the limit are."),
    ("Expected Shortfall (ES)", "The average loss on the days that go beyond the VaR limit. It answers: when things are worse than the limit, how bad are they on average?"),
    ("Backtesting", "Checking an old forecast against what really happened. Each day we pretend not to know the future, make the forecast, then compare it with the actual result."),
    ("Bad day / breach", "A day on which the loss was bigger than the forecast limit."),
    ("Kupiec test", "Asks: did we get about the right NUMBER of bad days? (95% should give about 5 per 100 days.)"),
    ("Christoffersen test", "Asks: were the bad days spread out, or did they come in bunches (as in a crisis)? A model can pass the Kupiec test and still fail this one."),
    ("Stress test", "Replaying a real past crisis on today's portfolio to see what would have happened, without relying on a statistical model."),
    ("Fat tails", "Extreme days happen more often than a bell curve says they should."),
]


def render_pdf_report(data: dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            leftMargin=0.6 * inch, rightMargin=0.6 * inch)
    S = _styles()
    story = []
    nar, pv = data["narrative"], data["portfolio"]["value"]
    r95, r99 = data["results"][0.95], data["results"][0.99]
    ps, rep = data["portfolio_stats"], data["reproducibility"]

    story.append(Paragraph("Quant Risk Analytics - Model Report", S["ReportTitle"]))
    story.append(_p(f"Generated {data['generated_at']}  |  Data: {data['data_period']} "
                    f"({data['n_observations']} trading days)  |  Portfolio value {_usd(pv)}", S["Meta"]))

    story.append(Paragraph("What this report says, in plain words", S["H2"]))
    for t in nar["plain"]:
        story.append(_p(t, S["Body"]))

    story.append(Paragraph("The portfolio", S["H2"]))
    story.append(_table([[t, f"{w * 100:.1f}%"] for t, w in data["portfolio"]["weights"].items()],
                        ["Ticker", "Weight"], col_widths=[2 * inch, 1.5 * inch]))
    story.append(Spacer(1, 6))
    story.append(_p(f"Average daily return {_pct(ps['daily_mean'], 3)}; typical daily swing "
                    f"(volatility) {_pct(ps['daily_vol'], 3)}; excess kurtosis {ps['kurtosis']:.2f} "
                    f"(0 would be a bell curve); skew {ps['skew']:.2f}.", S["Body"]))

    story.append(Paragraph("Value at Risk and Expected Shortfall", S["H2"]))
    rows = []
    for name, key in [("Historical", "historical"), ("Parametric", "parametric"),
                      ("Monte Carlo", "monte_carlo"), ("EWMA", "ewma")]:
        rows.append([name, _usd(r95[key + "_var"] * pv), _usd(r95[key + "_es"] * pv),
                     _usd(r99[key + "_var"] * pv), _usd(r99[key + "_es"] * pv)])
    story.append(_table(rows, ["Method", "VaR 95%", "ES 95%", "VaR 99%", "ES 99%"]))
    story.append(Spacer(1, 6))
    story.append(_p(f"Diversification: holding these assets together lowers the 95% historical "
                    f"VaR by about {_usd(data['diversification_benefit'] * pv)} per day compared "
                    f"with the weighted average of holding each one alone.", S["Body"]))
    story.append(_p("Historical, Parametric and Monte Carlo use the whole data period; EWMA uses "
                    "recent volatility, so it describes today's conditions rather than the long-run "
                    "average.", S["Body"]))

    story.append(PageBreak())
    story.append(Paragraph("Did the models work? (backtesting)", S["H2"]))
    story.append(_p(f"Each day, each method forecast a loss limit using only the past "
                    f"{BACKTEST_WINDOW} days of data (EWMA: all data so far). Results below cover "
                    f"{nar['n_forecasts']} daily forecasts per method, all on the same dates.", S["Meta"]))
    for t in nar["backtest_summary"]:
        story.append(_p(t, S["Body"]))
    for cl in (0.95, 0.99):
        story.append(Paragraph(f"{int(cl * 100)}% confidence", S["Heading3"]))
        rows = []
        for m, r in data["backtests"][cl].items():
            k, c, cc = r["kupiec"], r["christoffersen"], r["conditional_coverage"]
            rows.append([m, f"{k['expected']:.1f}", str(k["n_breaches"]), f"{k['p_value']:.4f}",
                         f"{c['p_value']:.4f}", f"{cc['p_value']:.4f}", r["verdict"]])
        story.append(_table(rows, ["Method", "Expected bad days", "Actual", "Kupiec p",
                                   "Christoffersen p", "Combined p", "Verdict"]))
        story.append(Spacer(1, 6))
    story.append(_p("p-values below 0.05 mean the result is unlikely to be luck. PASS = right number "
                    "of bad days and not bunched; CLUSTERED = right number but bunched together; "
                    "FAIL = wrong number of bad days. Monte Carlo is not listed because under a "
                    "bell-curve model it is the same as Parametric.", S["Meta"]))
    for t in nar["backtest_lines"]:
        story.append(_p("- " + t, S["Body"]))

    story.append(Paragraph("When the limit was broken, how bad was it?", S["H2"]))
    story.append(_p("On bad days we compare the actual loss with the Expected Shortfall the model "
                    "forecast for that day. 1.0x would be perfect; above 1.0x means real losses "
                    "were bigger than the model said.", S["Body"]))
    story.append(_table([[m, f"{cl:.0%}", str(e["n_breaches"]), _pct(e["avg_loss_on_breach"]),
                          _pct(e["avg_forecast_es_on_breach"]), f"{e['ratio']:.2f}x"]
                         for m, cl, e in nar["es_rows"]],
                        ["Method", "Level", "Bad days", "Avg actual loss", "Avg forecast ES", "Actual / forecast"]))

    story.append(PageBreak())
    story.append(Paragraph("Replaying real crises (stress testing)", S["H2"]))
    rows = []
    for n, s in data["stress"].items():
        ratio = (abs(s["worst_day"]) / 100 / s["pre_var_99"]) if np.isfinite(s["pre_var_99"]) else float("nan")
        rows.append([n, f"{s['total']:.2f}%", f"{s['worst_day']:.2f}%",
                     _pct(s["pre_var_99"]) if np.isfinite(s["pre_var_99"]) else "n/a",
                     f"{ratio:.1f}x" if np.isfinite(ratio) else "n/a", str(s["pre_obs"])])
    story.append(_table(rows, ["Scenario", "Total return", "Worst day", "Normal-day 99% VaR",
                               "Worst day / VaR", "Days of history used"]))
    story.append(Spacer(1, 6))
    for t in nar["stress_lines"]:
        story.append(_p(t, S["Body"]))

    story.append(Paragraph("Warnings and data checks", S["H2"]))
    if nar["warnings"]:
        for w in nar["warnings"]:
            story.append(_p("- " + w, S["Body"]))
    else:
        story.append(_p("None.", S["Body"]))

    story.append(Paragraph("What the models assume", S["H2"]))
    for a in [
        "All figures are for a one-day horizon. The portfolio is rebalanced to its weights every day.",
        "Portfolio return = sum of (weight x each asset's simple daily return); loss = minus the return.",
        "Historical: the future looks like the past data. No bell-curve assumption.",
        "Parametric and Monte Carlo: returns follow a bell curve with the historical average and covariance.",
        f"EWMA: bell curve, zero average return, volatility that gives recent days more weight (decay factor {rep['ewma_lambda']}).",
        "Stress scenarios replay actual past returns; they are not forecasts.",
        "Educational use only. Not investment advice and not a regulatory capital model.",
    ]:
        story.append(_p("- " + a, S["Body"]))

    story.append(Paragraph("Glossary", S["H2"]))
    story.append(_table([[t, d] for t, d in GLOSSARY], ["Term", "Meaning"],
                        col_widths=[1.6 * inch, 5.3 * inch], styles=S, wrap=True))

    story.append(Paragraph("How to reproduce these numbers", S["H2"]))
    story.append(_p(f"Data fingerprint {rep['data_fingerprint']}; Monte Carlo seed {rep['mc_seed']} with "
                    f"{rep['mc_simulations']:,} simulations; backtest window {rep['backtest_window_days']} days; "
                    f"EWMA decay {rep['ewma_lambda']} with {rep['ewma_burn_in_days']}-day warm-up; "
                    f"pre-crisis window {rep['pre_crisis_window_days']} days; numpy {rep['numpy']}, "
                    f"pandas {rep['pandas']}, scipy {rep['scipy']}.", S["Meta"]))
    doc.build(story)
    buf.seek(0)
    return buf.read()
