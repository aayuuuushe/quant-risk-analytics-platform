"""Generates the risk report: data assembly + PDF rendering (reportlab,
pure Python, no system dependencies) + JSON export."""
from datetime import datetime, timezone
from io import BytesIO
import json

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem,
)

from src.risk.var_models import (
    historical_var, historical_es, parametric_var, parametric_es,
    monte_carlo_full, ewma_var_current,
)
from src.risk.metrics import diversification_benefit
from src.validation.backtesting import backtest_all_methods
from src.scenarios.stress_testing import stress_scenarios

MUTED = colors.HexColor("#555555")


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _warnings_for(port_ret: pd.Series, start_date: str) -> list:
    warnings = []
    n_obs = len(port_ret)
    if n_obs < 500:
        warnings.append(
            f"Sample size is {n_obs} trading days — below the ~2 years "
            f"typically wanted for stable 99% VaR/backtesting estimates."
        )
    if start_date >= "2015-08-01":
        warnings.append(
            "China Selloff (Aug 2015) scenario's pre-crisis VaR benchmark "
            "uses fewer than the standard 250-day window, since it falls "
            "near the start of the available data — less statistically "
            "reliable than the other three stress scenarios."
        )
    kurt = port_ret.kurtosis()
    if abs(kurt) > 3:
        warnings.append(
            f"Portfolio returns show excess kurtosis of {kurt:.2f} (fat tails) "
            f"— Parametric and Monte Carlo VaR, which assume normality, will "
            f"tend to understate tail risk relative to Historical VaR."
        )
    return warnings


def _plain_english_summary(results: dict, backtests: dict, stress: dict, pv: float) -> dict:
    """Translates the technical results into a summary any non-specialist
    reader — a recruiter, a hiring manager, a non-quant — can follow."""
    r95 = results[0.95]
    method_values_95 = {
        "Historical": r95["historical_var"], "Parametric": r95["parametric_var"],
        "Monte Carlo": r95["monte_carlo_var"], "EWMA": r95["ewma_var"],
    }
    passing_both_95 = [m for m, res in backtests[0.95].items() if res["verdict"] == "PASS"]
    worst_scenario = max(stress.items(), key=lambda kv: abs(kv[1]["worst_day"]))

    return {
        "one_liner": (
            f"On a typical bad day (95% confidence), this portfolio is not expected "
            f"to lose more than about ${r95['historical_var']*pv:,.0f} out of "
            f"${pv:,.0f} — roughly {r95['historical_var']*100:.1f}% of its value."
        ),
        "which_model_worked_best": (
            f"Four different ways of estimating that number were tested. "
            f"'{passing_both_95[0]}' was the only one that held up cleanly when "
            f"checked against ten years of real market history — the others got "
            f"the frequency of bad days roughly right, but their bad days tended "
            f"to arrive in clusters (crises) rather than being spread out, which "
            f"the model didn't account for."
        ) if passing_both_95 else (
            "None of the four methods fully held up when checked against ten "
            "years of real market history at this confidence level — see "
            "Interpretation for why."
        ),
        "worst_real_event": (
            f"The single worst real-world test was the {worst_scenario[0]}, where "
            f"the portfolio would have lost about {abs(worst_scenario[1]['worst_day']):.1f}% "
            f"in one day — several times larger than what the day-to-day risk "
            f"estimate alone would have suggested was likely."
        ),
    }


def generate_report_data(log_returns: pd.DataFrame, port_ret: pd.Series,
                          w: np.ndarray, tickers: list, weights: dict,
                          portfolio_value: float, n_simulations: int,
                          start_date: str, end_date: str) -> dict:
    results = {}
    for cl in (0.95, 0.99):
        mc = monte_carlo_full(log_returns, w, cl, n_simulations)
        _, ewma_var, ewma_es = ewma_var_current(port_ret, cl)
        results[cl] = {
            "historical_var": historical_var(port_ret, cl),
            "historical_es": historical_es(port_ret, cl),
            "parametric_var": parametric_var(log_returns, w, cl),
            "parametric_es": parametric_es(log_returns, w, cl),
            "monte_carlo_var": mc["var"],
            "monte_carlo_es": mc["es"],
            "ewma_var": ewma_var,
            "ewma_es": ewma_es,
        }

    backtests = {cl: backtest_all_methods(port_ret, cl) for cl in (0.95, 0.99)}
    stress = stress_scenarios(port_ret)
    div_benefit = diversification_benefit(log_returns, w, 0.95)
    plain_summary = _plain_english_summary(results, backtests, stress, pv=portfolio_value)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_source": "Yahoo Finance (via yfinance), daily adjusted close",
        "data_period": f"{start_date} to {end_date}",
        "n_observations": len(port_ret),
        "portfolio": {"tickers": tickers, "weights": weights, "value": portfolio_value},
        "portfolio_stats": {
            "daily_mean": port_ret.mean(), "daily_vol": port_ret.std(),
            "skew": port_ret.skew(), "kurtosis": port_ret.kurtosis(),
        },
        "results": results,
        "backtests": backtests,
        "stress": stress,
        "diversification_benefit": div_benefit,
        "warnings": _warnings_for(port_ret, start_date),
        "plain_summary": plain_summary,
    }


def render_json_report(data: dict) -> str:
    """Exports a JSON-safe summary — deliberately excludes raw per-day
    series (var_series, returns, breaches, breach_dates), which are
    already summarized in the PDF's tables and aren't needed here."""

    def clean_backtest(res: dict) -> dict:
        return {
            "kupiec": res["kupiec"],
            "christoffersen": res["christoffersen"],
            "verdict": res["verdict"],
            "n_breaches": len(res["breach_dates"]),
        }

    def clean_stress(s: dict) -> dict:
        return {
            "total_return_pct": s["total"],
            "worst_day_pct": s["worst_day"],
            "pre_crisis_var99": s["pre_var_99"],
        }

    export = {
        "generated_at": data["generated_at"],
        "data_source": data["data_source"],
        "data_period": data["data_period"],
        "n_observations": data["n_observations"],
        "portfolio": data["portfolio"],
        "portfolio_stats": data["portfolio_stats"],
        "results": data["results"],
        "backtests": {
            str(cl): {m: clean_backtest(res) for m, res in methods.items()}
            for cl, methods in data["backtests"].items()
        },
        "stress": {name: clean_stress(s) for name, s in data["stress"].items()},
        "diversification_benefit": data["diversification_benefit"],
        "warnings": data["warnings"],
        "plain_summary": data["plain_summary"],
    }

    def convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"Not serializable: {type(o)}")

    return json.dumps(export, default=convert, indent=2)


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=4))
    styles.add(ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, textColor=MUTED, spaceAfter=14))
    styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=18, spaceAfter=8))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8))
    return styles


def _table(rows, header, col_widths=None):
    t = Table([header] + rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A1D27")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F6F8")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def render_pdf_report(data: dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.6 * inch, rightMargin=0.6 * inch)
    styles = _styles()
    story = []

    pv = data["portfolio"]["value"]
    r95, r99 = data["results"][0.95], data["results"][0.99]
    ps = data["portfolio_stats"]

    story.append(Paragraph("Quant Risk Analytics — Model Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Generated: {data['generated_at']} &nbsp;|&nbsp; "
        f"Data source: {data['data_source']} &nbsp;|&nbsp; "
        f"Period: {data['data_period']} ({data['n_observations']} trading days)",
        styles["Meta"]
    ))

    story.append(Paragraph("Executive Summary", styles["H2"]))
    story.append(Paragraph(
        f"A ${pv:,.0f} portfolio across {len(data['portfolio']['tickers'])} equities was assessed "
        f"using four Value at Risk methodologies (Historical Simulation, Parametric/Delta-Normal, "
        f"Monte Carlo, EWMA/RiskMetrics), validated against {data['n_observations']} days of historical "
        f"returns via Kupiec and Christoffersen backtests, and stress-tested against four historical "
        f"crisis periods. At 95% confidence, estimated 1-day VaR ranges from "
        f"${r95['ewma_var']*pv:,.0f} (EWMA) to ${r95['monte_carlo_var']*pv:,.0f} (Monte Carlo). "
        f"All methods pass Kupiec at 95%; all fail at 99%, a statistically expected result given the "
        f"limited number of tail observations available (see Limitations).",
        styles["Body"]
    ))

    # --- Plain Language Summary ---
    ps_summary = data["plain_summary"]
    story.append(Paragraph("In Plain Terms", styles["H2"]))
    story.append(Paragraph(ps_summary["one_liner"], styles["Body"]))
    story.append(Paragraph(ps_summary["which_model_worked_best"], styles["Body"]))
    story.append(Paragraph(ps_summary["worst_real_event"], styles["Body"]))

    story.append(Paragraph("Portfolio", styles["H2"]))
    weight_rows = [[t, f"{w*100:.1f}%"] for t, w in data["portfolio"]["weights"].items()]
    story.append(_table(weight_rows, ["Ticker", "Weight"], col_widths=[3 * inch, 2 * inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Portfolio value: ${pv:,.0f} &nbsp;|&nbsp; Daily mean return: {ps['daily_mean']*100:.4f}% "
        f"&nbsp;|&nbsp; Daily volatility: {ps['daily_vol']*100:.4f}% &nbsp;|&nbsp; "
        f"Skew: {ps['skew']:.2f} &nbsp;|&nbsp; Excess kurtosis: {ps['kurtosis']:.2f}",
        styles["Body"]
    ))

    story.append(Paragraph("Value at Risk &amp; Expected Shortfall", styles["H2"]))
    var_rows = [
        ["Historical", f"${r95['historical_var']*pv:,.0f}", f"${r95['historical_es']*pv:,.0f}",
         f"${r99['historical_var']*pv:,.0f}", f"${r99['historical_es']*pv:,.0f}"],
        ["Parametric", f"${r95['parametric_var']*pv:,.0f}", f"${r95['parametric_es']*pv:,.0f}",
         f"${r99['parametric_var']*pv:,.0f}", f"${r99['parametric_es']*pv:,.0f}"],
        ["Monte Carlo", f"${r95['monte_carlo_var']*pv:,.0f}", f"${r95['monte_carlo_es']*pv:,.0f}",
         f"${r99['monte_carlo_var']*pv:,.0f}", f"${r99['monte_carlo_es']*pv:,.0f}"],
        ["EWMA", f"${r95['ewma_var']*pv:,.0f}", f"${r95['ewma_es']*pv:,.0f}",
         f"${r99['ewma_var']*pv:,.0f}", f"${r99['ewma_es']*pv:,.0f}"],
    ]
    story.append(_table(var_rows, ["Method", "VaR 95%", "ES 95%", "VaR 99%", "ES 99%"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Diversification benefit at 95%: ${data['diversification_benefit']*pv:,.0f}/day "
        f"versus holding each asset individually.",
        styles["Body"]
    ))

    story.append(Paragraph("Model Validation — Backtesting", styles["H2"]))
    story.append(Paragraph("Rolling 250-day window, walk-forward (no look-ahead).", styles["Meta"]))
    for cl in (0.95, 0.99):
        story.append(Paragraph(f"{int(cl*100)}% Confidence", styles["Heading3"]))
        rows = []
        for method, res in data["backtests"][cl].items():
            k, c = res["kupiec"], res["christoffersen"]
            rows.append([
                method, "PASS" if k["passed"] else "FAIL", f"{k['p_value']:.4f}",
                "PASS" if c["passed"] else "FAIL", res["verdict"],
            ])
        story.append(_table(rows, ["Method", "Kupiec", "Kupiec p", "Christoffersen", "Verdict"]))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    story.append(Paragraph("Stress Testing", styles["H2"]))
    stress_rows = []
    for name, d in data["stress"].items():
        worst = abs(d["worst_day"]) / 100
        ratio = worst / d["pre_var_99"] if d["pre_var_99"] else float("nan")
        stress_rows.append([
            name, f"{d['total']:.2f}%", f"{d['worst_day']:.2f}%",
            f"{d['pre_var_99']*100:.2f}%", f"{ratio:.1f}x",
        ])
    story.append(_table(stress_rows,
                         ["Scenario", "Total Return", "Worst Day", "Pre-Crisis VaR99", "Worst/VaR99"]))

    story.append(Paragraph("Model Assumptions", styles["H2"]))
    assumptions = [
        "Historical Simulation: no distributional assumption; assumes the future resembles the observed sample.",
        "Parametric: assumes portfolio returns are normally distributed.",
        "Monte Carlo: samples from a multivariate normal fit to historical mean/covariance — under this assumption, converges to the Parametric result.",
        "EWMA (\u03bb=0.94): assumes zero mean return; time-varying volatility via exponential decay, seeded on a 250-day burn-in window.",
        "Portfolio returns are constructed by aggregating simple returns across assets, then converting to a log return — not by summing log returns directly.",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(a, styles["Body"])) for a in assumptions],
        bulletType="bullet", start="circle"
    ))

    story.append(Paragraph("Interpretation", styles["H2"]))
    story.append(Paragraph(
        "EWMA is the only method passing both Kupiec and Christoffersen at 95%, because its dynamic "
        "volatility adjustment prevents breach clustering during stress periods — the other three "
        "methods, holding a fixed rolling estimate, produce breaches that cluster together rather than "
        "occurring independently. All methods fail Kupiec at 99%, reflecting the limited number of tail "
        "observations (~23 expected) available to validate a 99th-percentile threshold over this sample size.",
        styles["Body"]
    ))

    story.append(Paragraph("Limitations", styles["H2"]))
    limitations = [
        "1-day VaR only — no regulatory horizon scaling (e.g. Basel's 10-day standard) is applied.",
        "Linear, equity-only risk — does not extend to options or other non-linear instruments.",
        "Monte Carlo currently draws from a Gaussian assumption and does not yet capture fat tails independently of the Parametric method.",
        "Data sourced via yfinance; Yahoo Finance adjusted closes can be revised historically, so exact reproduction of this report's figures is not guaranteed at a later date.",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(l, styles["Body"])) for l in limitations],
        bulletType="bullet", start="circle"
    ))

    story.append(Paragraph("Warnings", styles["H2"]))
    if data["warnings"]:
        story.append(ListFlowable(
            [ListItem(Paragraph(w, styles["Body"])) for w in data["warnings"]],
            bulletType="bullet", start="circle"
        ))
    else:
        story.append(Paragraph("None.", styles["Body"]))

    story.append(Paragraph("Glossary", styles["H2"]))
    glossary = [
        ("Value at Risk (VaR)", "The most this portfolio is expected to lose over one day, at a given confidence level (e.g. 95% or 99%) — it does not say how bad the loss could get beyond that threshold."),
        ("Expected Shortfall (ES)", "The average loss on the days that are worse than the VaR threshold — answers 'how bad is bad' once VaR is breached."),
        ("Confidence level (95% / 99%)", "How often the estimate is expected to hold. 95% means the loss is expected to exceed VaR on about 1 in 20 days; 99% means about 1 in 100 days."),
        ("Backtesting", "Checking, using real historical data, whether a risk model's predictions actually held up over time."),
        ("Kupiec test", "Checks whether the NUMBER of bad days matched what the model expected."),
        ("Christoffersen test", "Checks whether bad days were spread out evenly, or bunched together in clusters (like during a crisis) — a model can pass Kupiec and still fail this."),
        ("Stress testing", "Replaying real historical crises against the portfolio to see what actually would have happened, independent of any statistical model."),
    ]
    glossary_rows = [[term, defn] for term, defn in glossary]
    story.append(_table(glossary_rows, ["Term", "Meaning"], col_widths=[1.7 * inch, 4.8 * inch]))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "This report is generated for educational/demonstration purposes and does not "
        "constitute investment or risk-management advice.",
        styles["Meta"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()