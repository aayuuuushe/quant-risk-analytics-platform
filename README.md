# Quant Risk Analytics Platform

A Value-at-Risk engine and interactive dashboard implementing four risk
models, statistical model validation, and historical stress testing on
a 5-stock equity portfolio — built to demonstrate applied quantitative
risk methodology and clean, reproducible analytical Python, not
software engineering.

**Live demo:** _add your deployed Streamlit URL here once live_
**Portfolio:** AAPL (tech), JPM (banking), XOM (energy), JNJ (healthcare),
GS (financials) — equally weighted, daily prices 2015–2024 (2,514 trading days)

---

## Table of Contents

- [Motivation](#motivation)
- [Risk Models Implemented](#risk-models-implemented)
- [Mathematical Background](#mathematical-background)
- [Key Result](#key-result)
- [Two Corrections Made During Development](#two-corrections-made-during-development)
- [The Numbers](#the-numbers)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [References](#references)

---

## Motivation

Value at Risk is the standard first question in market risk management:
how much could a portfolio lose, and how much should that number be
trusted? This project builds four established approaches to answering
that question, checks each one against ten years of real market data
using formal statistical tests, and stress-tests the results against
genuine historical crises. Two methodological errors were found and
corrected during development — both are documented below, since finding
and fixing them was as instructive as building the models themselves.

The goal was not to build the most feature-rich dashboard possible, but
to build a small number of models correctly, prove they're correct with
real tests, and be honest about where each one breaks down.

---

## Risk Models Implemented

| Model | Assumption | Reference |
|---|---|---|
| Historical Simulation | None — uses the empirical return distribution directly | Hull, Ch. 22 |
| Parametric (Delta-Normal) | Portfolio returns are normally distributed | Hull, Ch. 22 |
| Monte Carlo | Multivariate normal, fit to historical mean/covariance | Hull, Ch. 22 |
| EWMA (RiskMetrics) | Time-varying volatility, exponential decay (λ = 0.94) | J.P. Morgan RiskMetrics, 1996 |

Each model estimates both **Value at Risk (VaR)** and **Expected
Shortfall (ES)** at 95% and 99% confidence.

---

## Mathematical Background

**Portfolio return construction.** Portfolio return is built by
aggregating *simple* returns across assets — the mathematically correct
approach, since simple returns are additive across a portfolio, while
log returns are only additive through time for a single asset — and
then converted to a log return where one is needed downstream (e.g. for
the EWMA recursion).

**Parametric VaR and ES**, for portfolio mean μ_p and volatility σ_p:

VaR_α = −(μ_p + z_α · σ_p)
ES_α = −μ_p + σ_p · φ(z_α) / (1 − α)

**EWMA volatility**, RiskMetrics recursion:

σ²_t = λ · σ²_(t−1) + (1 − λ) · r²_(t−1)

seeded on a 250-day burn-in window rather than the full sample, to
avoid look-ahead bias in the initial estimate.

**Model validation** uses two tests on every method, at both confidence
levels:
- **Kupiec Proportion-of-Failures test** — does the *number* of breach
  days match what the model's stated confidence level predicts?
- **Christoffersen Independence test** — are breach days spread
  independently through time, or do they cluster together (e.g. during
  a single crisis)?

A model can pass Kupiec and still fail Christoffersen — which is
exactly what happens with three of the four models here (see below).

---

## Key Result

**EWMA is the only model that passes both backtests at 95% confidence.**
The other three (Historical, Parametric, Monte Carlo) get the *number*
of breach days statistically right, but their breaches cluster together
during real crises — COVID, the 2018 selloff, the 2022 rate shock —
rather than occurring independently through time. That clustering is a
failure mode the Kupiec test alone cannot detect, which is exactly why
the Christoffersen test is run alongside it: EWMA's daily volatility
re-estimation lets its risk threshold widen quickly once a crisis
begins, spreading its breaches out instead of letting them accumulate
in one bad month.

All four models fail Kupiec at 99% confidence — a statistically
expected result given only ~23 tail observations are available across
the full ten-year sample.

---

## Two Corrections Made During Development

**1. Portfolio return construction.** Portfolio returns were originally
computed as a weighted sum of individual assets' log returns. Log
returns are only additive through time for a single asset, not across
assets in a portfolio — the correct construction weights *simple*
returns, then converts the result to a log return if needed downstream.
Corrected across every notebook; the effect on VaR figures is small
under normal conditions and grows on large-move days, since the
approximation error scales with the size of the return.

**2. Stress-test benchmarking.** The original "pre-crisis VaR" benchmark
was computed on the full 2015–2024 sample, which already contains the
crisis days being measured against it — a circular comparison. Each
scenario is now benchmarked against a VaR estimate computed only from
the 250 trading days immediately preceding that scenario's start
date — the risk figure a desk would actually have been holding on day
one of the event.

Both fixes shifted every downstream VaR, backtest, and stress-test
figure; the numbers below reflect the corrected methodology.

---

## The Numbers

**VaR on a $1,000,000 portfolio:**

| Method | VaR 95% | VaR 99% |
|--------|---------|---------|
| Historical | $18,045 | $34,161 |
| Parametric | $20,046 | $28,562 |
| Monte Carlo | $20,190 | $27,904 |
| EWMA (current) | $14,640 | $20,706 |

**Backtesting** (rolling 250-day window, walk-forward, no look-ahead):

| Method | 95% Verdict | 99% Verdict |
|---|---|---|
| Historical | CONDITIONAL | FAIL |
| Parametric | CONDITIONAL | FAIL |
| Monte Carlo | CONDITIONAL | FAIL |
| **EWMA** | **PASS** | FAIL |

**Stress testing** (each scenario benchmarked against its own
pre-crisis 99% VaR):

| Scenario | Worst Day | Pre-Crisis VaR99 | Worst / VaR99 |
|---|---|---|---|
| COVID Crash | −11.74% | 2.98% | **3.9x** |
| 2018 Selloff | −3.60% | 3.09% | 1.2x |
| 2022 Rate Shock | −3.75% | 2.46% | 1.5x |
| China Selloff | −4.09% | 1.99% | 2.1x |

Diversification benefit at 95%: **$6,331/day** versus holding each
asset individually.

Full detail, interpretation, assumptions, and limitations are in the
downloadable report generated by the app (see [Usage](#usage)).

---

## Features

- Four VaR/ES methodologies, each with its own dashboard page
- Statistical backtesting (Kupiec + Christoffersen) at both 95% and 99%
- Historical stress testing against four real market crises, benchmarked
  against scenario-specific pre-crisis VaR (not a circular full-sample figure)
- Interactive portfolio controls — assets, weights, portfolio value,
  date range — with every page recalculating live from the same shared
  `src/` logic
- One-click downloadable report (PDF and JSON) with an executive
  summary, plain-language interpretation for non-specialist readers, full
  model detail, assumptions, limitations, warnings, and a glossary
- Known-answer test suite pinned to the notebooks' own output

---

## Project Structure

```
├── notebooks/              # 01-07 — the mathematical derivations (source of truth)
│   ├── 01_market_data_pipeline.ipynb
│   ├── 02_historical_var.ipynb
│   ├── 03_parametric_var.ipynb
│   ├── 04_Monte_carlo_var.ipynb
│   ├── 05_EWMA_var.ipynb
│   ├── 06_Backtesting.ipynb
│   └── 07_Stress_testing.ipynb
├── src/
│   ├── config.py            # shared portfolio/model defaults
│   ├── data/loader.py
│   ├── risk/                # var_models.py, metrics.py
│   ├── validation/          # backtesting.py — Kupiec, Christoffersen
│   ├── scenarios/           # stress_testing.py
│   ├── visualization/       # charts.py
│   └── reporting/           # report_generator.py — PDF/JSON reports
├── app.py                   # Streamlit home page
├── pages/                   # one page per method, mirrors notebooks 1:1
├── ui_helpers.py            # shared sidebar/theme — presentation layer only
├── tests/                   # known-answer tests, pinned to notebook output
├── reports/                 # generated charts
├── docs/model_validation_memo.md
├── requirements.txt
└── README.md
```

`src/` contains every calculation exactly once; notebooks derive the
math, and the Streamlit app calls into `src/` for everything it
displays — nothing is computed twice, and nothing in the app can drift
from what the notebooks derived.

---

## Technologies

Python · pandas · NumPy · SciPy · Plotly · Streamlit · reportlab ·
pytest · yfinance

---

## Installation

```bash
git clone <https://github.com/aayuuuushe/quant-risk-analytics-platform>
cd quant-risk-analytics-platform
pip install -r requirements.txt
```

---

## Usage

**Notebooks** — run `01` through `07` in order to regenerate the
underlying data and figures. Notebook 01 pulls live data from Yahoo
Finance (internet connection required, no API key needed) and must run
first.


**Interactive dashboard:**

```bash
streamlit run app.py
```

The sidebar lets you explore different assets, weights, portfolio
value, and date ranges — useful for demonstration and comparing how
the models behave under different inputs. The baseline results
reported in this README (VaR figures, backtest verdicts, stress test
ratios) come from the fixed 5-asset, equal-weight portfolio validated
in the notebooks; changing the sidebar inputs recalculates every
number live using the same underlying models, but those specific
combinations haven't been separately backtested the way the notebook's
baseline case was.

**Full risk report** — downloadable from the home page as PDF or JSON,
containing an executive summary, plain-language interpretation, full
VaR/ES tables, backtest verdicts, stress test results, model
assumptions, limitations, warnings, and a glossary.

---

## Testing

```bash
pytest tests/ -v
```

Tests pin known-answer values derived directly from the notebooks (e.g.
Historical VaR at 95% = $18,045), so any future change to `src/` that
breaks the underlying math is caught automatically rather than silently
reaching the dashboard.

---

## Roadmap

Planned extensions, in priority order:

- **Portfolio risk contribution** — decompose total VaR by asset
  (component/marginal VaR) instead of showing only the aggregate
  figure, matching how risk attribution is used on professional risk
  desks.
- **Realistic macro stress scenarios** — sector-specific shocks (rate
  hikes, oil price, banking-sector decline) applied through the actual
  weight vector, replacing simple uniform percentage shocks.
- **GARCH(1,1) with Student-t innovations** — models volatility
  clustering directly and captures fat tails, directly addressing the
  limitation the current backtests already reveal (all four models
  fail Kupiec at 99%, consistent with fat-tailed returns the Gaussian
  and EWMA models don't fully capture).
- **Regulatory horizon scaling** — extending from 1-day VaR to the
  10-day horizon used in Basel capital calculations.

These are sequenced deliberately after the current baseline models are
fully tested and validated — correctness of what exists takes priority
over adding new, less-validated methods.

---

## References

- Hull, *Options, Futures and Other Derivatives*, Ch. 22
- J.P. Morgan, *RiskMetrics Technical Document*, 1996
- Kupiec, P. H. *Techniques for Verifying the Accuracy of Risk
  Measurement Models*, Journal of Derivatives, 1995
- Christoffersen, P. F. *Evaluating Interval Forecasts*, International
  Economic Review, 1998
- Basel Committee on Banking Supervision, *Supervisory Framework for
  the Use of Backtesting*, 1996

---

*This project is for educational and demonstration purposes and does
not constitute investment or risk-management advice.*