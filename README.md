# Quant Risk Analytics Platform

I built this to get a proper understanding of how market risk is 
measured in practice. The project covers the four main VaR methods 
used in bank risk departments, backtests the model against ten years 
of real data, and stress tests the portfolio against some of the worst 
market periods of the last decade.

The portfolio is five US equities — AAPL, JPM, XOM, JNJ, GS — equally 
weighted, with daily data from 2015 to 2024.

---

## What is in here

Seven notebooks that build on each other:

**01 — Data Pipeline**
Downloads ten years of daily prices from Yahoo Finance and converts 
them into log returns. Everything else reads from this output.

**02 — Historical Simulation VaR**
The simplest approach — sort the actual historical returns from worst 
to best and read off the loss at the confidence level you care about. 
No assumptions about the distribution. Just what actually happened.

**03 — Parametric VaR**
Assumes returns follow a normal distribution and computes VaR 
analytically from the portfolio mean and volatility. Faster than 
historical simulation but less accurate in the tails.

**04 — Monte Carlo VaR**
Generates 10,000 random scenarios using the historical covariance 
structure of the five stocks and estimates VaR from the simulated 
distribution. Under normality this converges to the parametric result, 
but the framework extends naturally to more complex models.

**05 — EWMA Volatility**
Uses the RiskMetrics exponential weighting approach with λ = 0.94 so 
that recent market moves carry more weight than older ones. The VaR 
estimate rises and falls with current market conditions rather than 
being anchored to a fixed historical window.

**06 — Backtesting**
Checks whether the model actually works. Recomputes VaR every day 
using a rolling one-year window and counts how many times the actual 
loss exceeded the estimate. Uses the Kupiec test to decide whether 
that number is statistically acceptable.

**07 — Stress Testing**
Looks at what the portfolio actually lost during the COVID crash, the 
2018 selloff, the 2022 rate shock, and the 2015 China correction. 
Also applies simple instantaneous shocks to see the dollar impact.

---

## Numbers

**VaR on a $1,000,000 portfolio:**

| Method | 95% | 99% |
|--------|-----|-----|
| Historical | $18,059 | $34,243 |
| Parametric | $20,046 | $28,562 |
| Monte Carlo | $20,190 | $27,904 |
| EWMA (current) | $14,559 | $20,591 |

The EWMA number is lower than the others because current volatility 
(0.89% per day) is below the long-run average. That is the point of 
the model — it reflects what is happening now, not just the last 
ten years on average.

**Backtesting:**

| Level | Expected breaches | Observed | p-value | |
|-------|------------------|----------|---------|---|
| 95% | 113 | 106 | 0.48 | PASS |
| 99% | 23 | 32 | 0.06 | PASS |

Both pass the Kupiec test. The 99% result is closer to the boundary 
which makes sense — the model is being asked to predict a rarer event 
and there is less data to validate it against.

**Stress testing:**

| Scenario | Total loss | Worst single day |
|----------|-----------|-----------------|
| COVID crash | −38.32% | −11.81% |
| 2018 selloff | −19.03% | −3.66% |
| 2022 rate shock | +4.81% | −3.76% |
| China selloff | −9.28% | −4.10% |

The COVID worst day was −11.81%. The 99% VaR over the same period 
was around 2.8%. That gap is exactly why stress testing exists — 
VaR tells you about normal bad days, not crises.

Diversification saves $6,317 per day at 95% confidence compared to 
holding each stock individually.

![Backtesting chart](reports/06_backtesting.png)

---

## How to run

```bash
pip install -r requirements.txt
```

Run the notebooks in order from 01 to 07. Notebook 01 has to go 
first because it generates the data files everything else depends on.

---

## Stack

Python · pandas · NumPy · SciPy · matplotlib · yfinance

---

## What I would add next

A Streamlit dashboard so the portfolio weights and confidence levels 
can be changed interactively, with VaR recalculating in real time. 
After that, a Black-Scholes notebook to extend this from equity risk 
into options pricing.

---

## References

- Hull — Options, Futures and Other Derivatives, Ch. 22
- J.P. Morgan — RiskMetrics Technical Document, 1996
- Hilpisch — Python for Finance, O'Reilly
- Basel Committee — Supervisory Framework for Backtesting, 1996
