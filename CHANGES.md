# Patch notes - what changed and why

Copy every file in this zip over the file with the same path in your repo
(keep the folder structure). Then:

1. Delete `tests/test_var_models.py`, `tests/test_stress_testing.py`, `tests/test_backtesting.py`
   (they are pinned to the OLD numbers; `tests/test_core.py` replaces them and needs no data file).
2. Copy `data/log_returns.csv` to `sample_data/log_returns.csv` and COMMIT it
   (make sure `.gitignore` does not ignore `sample_data/`). The app now reads this first,
   so the public app reproduces your numbers instead of downloading from Yahoo.
3. `pip install pytest` then run `python -m pytest -q` from the repo root. Expect 24 passed.
4. Update the notebooks (list below), re-run them, then refresh the README tables.

## Added after final checks
* `requirements.txt` (replaces yours): everything the dashboard, notebooks 02-07 and the tests need.
* Monte Carlo now draws its random numbers in alphabetical asset order, so the same seed gives the same result
  whatever order the assets are listed in (before this, the dashboard showed $20,136 instead of the notebook's $20,037).
  A test covers this. 25 tests now pass.

## The one convention (src/risk/var_models.py)
Portfolio return R = sum of (weight x each asset's SIMPLE daily return). Loss = -R.
Historical, Parametric, Monte Carlo, EWMA, backtests and stress tests all use this same series.
Before, Parametric/Monte Carlo used weighted LOG returns and the others used log(1+R).
Parametric sigma now equals the sample std of the portfolio series exactly.

## Files
| File | Change |
|---|---|
| src/config.py | new constants: pre-crisis window, minimum observations |
| src/risk/var_models.py | one return definition; Monte Carlo uses a local random generator (no global seed); EWMA "current" volatility now forecasts the NEXT day (includes the last return) |
| src/risk/metrics.py | individual-asset VaRs use simple returns; new helper used by page 1 |
| src/validation/backtesting.py | EWMA no longer lagged twice; Monte Carlo removed from the backtest (it equals Parametric); exact Kupiec/Christoffersen with zero-count cases handled; conditional-coverage test; ES tail check; all methods on identical dates; verdict label CONDITIONAL renamed CLUSTERED |
| src/scenarios/stress_testing.py | no crash when history is missing (benchmark = NaN below 60 days); reports days of history used and breaches of the pre-crisis VaR; compounded total return; drawdown table now lists distinct episodes |
| src/data/loader.py | columns always in the requested ticker order; missing tickers raise a clear error; reads sample_data/ first |
| src/data/quality.py | NEW: data-quality checks (dates, missing/inf values, too little history, gaps, huge moves, stale prices, duplicate series) |
| src/reporting/report_generator.py | every sentence computed from results; plain-language summary; ES tail table; reproducibility block (seed, window, data fingerprint, versions); strict JSON (no NaN); no unsupported characters in the PDF |
| src/visualization/charts.py | one line: stress chart compounds simple returns |
| ui_helpers.py | data-quality warnings/stop, weights matched by column name, minimum 500 days, cached backtests, widget keys |
| app.py | plain-words box, duplicate table removed, cached backtests and report |
| pages/1,5,6 | page 1 helper; page 5 shows p-values, expected vs actual, ES check; page 6 shows history used, breaches, honest drawdown cross-check |
| pages/2,3,4 | only `use_container_width=True` -> `width="stretch"` (old argument is being removed from Streamlit) |

## Notebook edits (same logic as src; ideally import from src so there is one copy)
* 02: `portfolio_returns = portfolio_simple_returns` (drop `np.log1p`); individual-asset VaRs on `simple_returns[ticker]`.
* 03: compute mu and covariance from `simple_returns = np.exp(log_returns) - 1`, not from `log_returns`.
* 04: simulate from the mean/covariance of `simple_returns`; use `rng = np.random.default_rng(42)` and `rng.multivariate_normal(...)` so it matches the app.
* 05: portfolio return = simple; "current" vol = `sqrt(0.94*ewma_var[-1] + 0.06*r[-1]**2)`.
* 06: portfolio return = simple; delete `.shift(1)` on the EWMA series; drop the Monte Carlo rolling row; replace the test functions with `from src.validation.backtesting import kupiec_test, christoffersen_test`; delete the hard-coded conclusion print statements.
* 07: total return = `((1 + period).prod() - 1) * 100`; guard pre-crisis VaR when fewer than 60 days exist.

## Expect these numbers to move slightly
VaR/ES (about 1-2 basis points from the return definition), EWMA backtest rows, Parametric/Monte Carlo
figures, stress totals (compounding). Nothing was pinned to new values yet: run on your real CSV first,
then pin them as regression tests.

## Not done (later, with your engineer friend)
Notebooks importing from src; CSV/API upload; Student-t/GARCH models; ES significance tests;
CI. See the roadmap discussion.
