# Quant Risk Analytics Platform

Ten years of daily market data. Five stocks across five sectors: tech,
banking, energy, healthcare, and financials.

Risk management, at its core, comes down to one question: how much
could this portfolio lose, and how confident can we be in that number?
Value at Risk (VaR) answers the first part. It estimates the maximum
expected loss at a given confidence level, over a given time horizon.
What it does not answer is how bad the loss could get once that
threshold is breached. That is the domain of Expected Shortfall (ES),
which averages the losses in the tail beyond VaR instead of stopping
at a single cutoff. This project focuses on VaR. It builds four
different versions of the model, tests whether each one predicts
losses accurately using ten years of real data, and then places the
results against genuine market crises: the COVID crash, the 2018
selloff, the 2022 rate shock, and the 2015 China selloff, to see where
each method holds up and where it breaks down.

**The portfolio:** AAPL (tech), JPM (banking), XOM (energy), JNJ
(healthcare), GS (financials), equally weighted, using daily prices
from 2015-01-02 to 2024-12-30 (2,515 trading days, 2,514 daily returns).

---

## Table of contents

- [What's inside](#whats-inside)
- [The numbers](#the-numbers)
- [Methodology notes](#methodology-notes--changes-from-the-original-implementation)
- [How to run it](#how-to-run-it)
- [Repo structure](#repo-structure)
- [Built with](#built-with)
- [References](#references)

---

## What's inside

Seven notebooks, each building on the last:

**01 — Data Pipeline**
Downloads ten years of daily prices from Yahoo Finance and converts
them into log returns. Every subsequent notebook reads from this
output.

**02 — Historical Simulation VaR**
The most direct approach: sort the actual historical portfolio returns
from worst to best and read off the loss at the confidence level
required. No distributional assumptions, only what actually occurred.
Also computes the diversification benefit of holding five stocks
instead of any single one.

**03 — Parametric VaR**
Assumes portfolio returns follow a normal distribution and computes
VaR analytically from the portfolio's mean and volatility. Faster to
compute than historical simulation, but less accurate in the tails,
where it matters most. That limitation is exactly what motivates the
Monte Carlo and EWMA approaches that follow.

**04 — Monte Carlo VaR**
Generates 10,000 simulated scenarios using the historical covariance
structure of the five stocks and estimates VaR from the resulting
distribution. Under normal conditions this converges with the
parametric result, but the same framework extends to far more complex
models, including non-normal return distributions.

**05 — EWMA Volatility**
Applies the RiskMetrics exponential weighting method (λ = 0.94), so
recent market movements carry more weight than older ones. The
resulting VaR estimate tracks current market conditions instead of a
fixed historical average. Seeded on a 250-day burn-in window rather
than the full sample, to avoid look-ahead bias in the initial estimate.

**06 — Backtesting**
Validates whether the model performs as expected. Recomputes VaR
daily using a rolling one-year window for all four methods, shifted
forward one day so the estimate for date *t* uses only information
available through *t*-1. Applies two tests to each method: the Kupiec
Proportion of Failures test (is the breach frequency consistent with
the stated confidence level?) and the Christoffersen Independence test
(are breaches clustered in time rather than independent?).

**07 — Stress Testing**
Evaluates how the portfolio performed during the most severe market
periods of the last decade: the COVID crash, the 2018 selloff, the
2022 rate shock, and the 2015 China selloff. Each scenario is
benchmarked against its own pre-crisis VaR — estimated only from the
250 trading days immediately before that scenario began — rather than
a full-sample figure, which would be circular since it includes the
crisis itself. Also applies simple instantaneous shocks to quantify
direct dollar impact. This is the kind of scenario analysis VaR cannot
capture on its own.

---

## The numbers

**VaR on a $1,000,000 portfolio** *(portfolio daily return: mean 0.06%,
std 1.25%; recalculated after fixing portfolio return construction —
simple returns are now aggregated across assets, then converted to a
log return, rather than summing log returns directly; see
[Methodology Notes](#methodology-notes--changes-from-the-original-implementation)
below):*

| Method | 95% | 99% |
|--------|-----|-----|
| Historical | $18,045 | $34,161 |
| Parametric | $20,046 | $28,562 |
| Monte Carlo | $20,190 | $27,904 |
| EWMA (current) | $14,640 | $20,706 |

Current EWMA volatility is 0.89%/day, versus a full-sample average
daily volatility of 1.25% — EWMA sits below the other three methods
because recent market conditions have been calmer than the ten-year
average. That is the intended behavior of the model: it reflects
present conditions rather than a fixed historical average.

Expected Shortfall alongside each VaR:

| Method | ES 95% | ES 99% |
|--------|--------|--------|
| Historical | $29,459 (1.63x VaR) | $51,214 (1.50x VaR) |
| Parametric | $25,267 | $32,796 |
| Monte Carlo | $25,329 | $32,427 |
| EWMA | $18,359 | $23,722 |

Diversification benefit at 95%: **$6,331 per day**, versus the
weighted average of holding each of the five stocks individually.

**Backtesting** *(rolling 250-day window, walk-forward, no look-ahead
— the estimate for date t uses only returns through t-1):*

| Method | CL | Expected | Observed | Kupiec p | Kupiec | Chris p | Christoffersen | Verdict |
|---|---|---|---|---|---|---|---|---|
| Historical | 95% | 113.2 | 109 | 0.6837 | PASS | 0.0002 | FAIL | CONDITIONAL |
| Historical | 99% | 22.6 | 36 | 0.0094 | FAIL | 0.0199 | FAIL | FAIL |
| Parametric | 95% | 113.2 | 108 | 0.6135 | PASS | 0.0002 | FAIL | CONDITIONAL |
| Parametric | 99% | 22.6 | 59 | 0.0000 | FAIL | 0.0000 | FAIL | FAIL |
| **EWMA** | **95%** | 113.2 | 114 | 0.9347 | PASS | 0.0874 | PASS | **PASS** |
| EWMA | 99% | 22.6 | 54 | 0.0000 | FAIL | 0.1847 | PASS | FAIL |
| Monte Carlo | 95% | 113.2 | 108 | 0.6135 | PASS | 0.0002 | FAIL | CONDITIONAL |
| Monte Carlo | 99% | 22.6 | 59 | 0.0000 | FAIL | 0.0000 | FAIL | FAIL |

All methods pass Kupiec at 95% — breach counts are statistically
consistent with the stated confidence level under normal market
conditions. All methods fail Kupiec at 99% — a systematic result. With
only ~23 expected tail observations across 2,514 days, finite-sample
estimation cannot reliably anchor a 99th-percentile threshold, and
fat-tailed return distributions compound this further.

EWMA is the only method passing **both** tests at 95%. Its dynamic
volatility adjustment prevents breach clustering during stress
periods; the other three methods, holding a fixed rolling estimate,
produce breaches that cluster together during volatile periods —
a failure mode Kupiec's breach *count* alone cannot detect, which is
exactly why the Christoffersen test is run alongside it.

**Stress testing** *(each scenario benchmarked against its own
pre-crisis 99% VaR — the 250 trading days immediately before that
scenario began, not a full-sample figure; see
[Methodology Notes](#methodology-notes--changes-from-the-original-implementation)):*

| Scenario | Total Return | Worst Day | Pre-Crisis VaR99 | Worst/VaR99 |
|---|---|---|---|---|
| COVID Crash | −37.86% | −11.74% | 2.98% | 3.9x |
| 2018 Selloff | −18.66% | −3.60% | 3.09% | 1.2x |
| 2022 Rate Shock | +6.93%* | −3.75% | 2.46% | 1.5x |
| China Selloff | −9.13% | −4.09% | 1.99%† | 2.1x |

*The 2022 rate shock ended positive over the full period, despite a
single-day loss of 3.75% at its worst point.

†China Selloff's pre-crisis window has only ~150 trading days of
history available (the sample starts 2015-01-02), versus the full 250
days used for the other three scenarios — this benchmark rests on a
thinner, less statistically reliable sample than the other three.

The COVID crash's single worst day lost 11.74% — 3.9 times larger than
the 99% VaR estimated from the 250 days immediately preceding it. That
gap is the practical argument for why stress testing complements VaR
rather than replacing it: VaR is calibrated to typical days, not to
the specific historical scenario about to unfold.

**Scenario selection** was verified against the data rather than
asserted: the six worst rolling drawdown points in the portfolio's own
return series were located directly (the worst is −49.15% on
2020-03-23, matching the COVID trough) and each of the four scenario
windows above was chosen to bracket a genuine drawdown found this way,
not picked from an external list.

![Backtesting chart](reports/06_backtesting.png)

---

## Methodology Notes — changes from the original implementation

Two corrections were made after the initial version of this project,
both caught during a self-review before treating the notebooks as
production logic:

1. **Portfolio return construction.** Portfolio returns were originally
   computed as a weighted sum of individual assets' *log* returns. Log
   returns are only additive through time for a single asset, not
   across assets in a portfolio — the correct construction is to
   weight *simple* returns (which are additive cross-sectionally),
   then convert the result to a log return if one is needed
   downstream. This has been corrected in all notebooks (02–07). The
   effect on VaR figures is small under normal conditions and grows on
   large-move days, since the approximation error scales with the
   size of the return.

2. **Stress-test benchmarking.** The original "pre-crisis VaR"
   benchmark was computed on the full 2015–2024 sample, which already
   contains the crisis days being measured against it — a circular
   comparison, since the benchmark would be partly built from the very
   event it's supposed to be measuring. Each scenario in Notebook 07
   is now benchmarked against a VaR estimate computed only from the
   250 trading days immediately preceding that scenario's start date —
   the risk figure a desk would actually have been holding on day one
   of the event.

Both fixes shifted every downstream VaR, backtest, and stress-test
figure slightly; the numbers in this README reflect the corrected
methodology, not the original implementation.

---

## How to run it

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

Run the notebooks in sequence, `01` through `07`, to regenerate `data/`
and `reports/`. Notebook 01 must run first, since it generates the
data file every other notebook depends on and requires an internet
connection (no API key needed).

To run the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard's sidebar controls the portfolio (assets, weights,
portfolio value, date range) and the confidence level (95%/99%, the
only two levels the underlying models are validated against). All
calculations shown in the app are read from `src/`, the same logic
used in the notebooks — nothing is recalculated independently in the
app.

---

## Repo structure

```
├── notebooks/            # 01-07, run in order — the authoritative derivations
├── src/
│   ├── config.py          # shared portfolio/model defaults
│   ├── data/
│   │   └── loader.py
│   ├── risk/
│   │   ├── var_models.py  # historical, parametric, Monte Carlo, EWMA
│   │   └── metrics.py     # diversification benefit
│   ├── validation/
│   │   └── backtesting.py # rolling VaR, Kupiec, Christoffersen
│   ├── scenarios/
│   │   └── stress_testing.py
│   └── visualization/
│       └── charts.py
├── app.py                 # Streamlit home page
├── pages/                 # one page per method, mirrors notebooks 1:1
├── ui_helpers.py           # shared sidebar/theme — presentation layer only
├── tests/                 # known-answer tests, pinned to notebook output
├── reports/                # generated charts and output data
├── docs/
│   └── model_validation_memo.md
├── requirements.txt
└── README.md
```

---

## Built with

Python, pandas, NumPy, SciPy, matplotlib, Plotly, Streamlit, yfinance

---

## References

- Hull, *Options, Futures and Other Derivatives*, Ch. 22
- J.P. Morgan, *RiskMetrics Technical Document*, 1996
- Hilpisch, *Python for Finance*, O'Reilly
- Basel Committee, *Supervisory Framework for Backtesting*, 1996