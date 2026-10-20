"""Tests that need NO data file. Each uses a hand calculation, an independent
re-implementation, or a basic property as the 'oracle' - not the notebook output."""
import math
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

from src.data import loader
from src.data.quality import check_data_quality
from src.risk.var_models import (
    portfolio_returns_series, portfolio_mu, portfolio_sigma, historical_var,
    historical_es, parametric_var, parametric_es, monte_carlo_var, monte_carlo_full,
    ewma_variance_series, ewma_var_current,
)
from src.scenarios.stress_testing import stress_scenarios, worst_drawdown_windows
from src.validation.backtesting import (
    kupiec_test, christoffersen_test, verdict, rolling_historical_var,
    rolling_parametric_var, rolling_ewma_var, backtest_all_methods,
)


def make_log_returns(n=1500, seed=3, k=5, start="2016-01-04"):
    rng = np.random.default_rng(seed)
    data = 0.0004 + 0.01 * (0.6 * rng.standard_normal((n, 1)) + 0.8 * rng.standard_normal((n, k)))
    return pd.DataFrame(data, index=pd.bdate_range(start, periods=n),
                        columns=["AAPL", "JPM", "XOM", "JNJ", "GS"][:k])


# ---------------------------------------------------------------- VaR / ES
def test_historical_var_es_hand_example():
    r = pd.Series([-0.04, -0.02, 0.01, 0.02, 0.03])
    # 20th percentile sits at position 0.2*(5-1)=0.8 -> -0.04 + 0.8*0.02 = -0.024
    assert historical_var(r, 0.80) == pytest.approx(0.024)
    assert historical_es(r, 0.80) == pytest.approx(0.04)   # only -0.04 is <= -0.024


def test_parametric_matches_normal_formulas():
    lr = make_log_returns()
    w = np.full(5, 0.2)
    mu, sg = portfolio_mu(lr, w), portfolio_sigma(lr, w)
    z = norm.ppf(0.01)
    assert parametric_var(lr, w, 0.99) == pytest.approx(-(mu + z * sg))
    assert parametric_es(lr, w, 0.99) == pytest.approx(-mu + sg * norm.pdf(z) / 0.01)


def test_portfolio_moments_equal_series_moments():
    lr = make_log_returns()
    w = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    pr = portfolio_returns_series(lr, w)
    assert portfolio_mu(lr, w) == pytest.approx(pr.mean())
    assert portfolio_sigma(lr, w) == pytest.approx(pr.std())      # same series, same definition


def test_monte_carlo_converges_to_parametric_and_is_reproducible():
    lr, w = make_log_returns(), np.full(5, 0.2)
    big = monte_carlo_var(lr, w, 0.95, 400_000, seed=1)
    assert big == pytest.approx(parametric_var(lr, w, 0.95), rel=0.02)
    assert monte_carlo_var(lr, w, 0.95, 10_000, seed=7) == monte_carlo_var(lr, w, 0.95, 10_000, seed=7)


def test_monte_carlo_does_not_depend_on_column_order():
    lr = make_log_returns()
    w = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    a = monte_carlo_full(lr, w, 0.99, 10_000)
    cols = list(lr.columns[::-1])
    b = monte_carlo_full(lr[cols], w[::-1], 0.99, 10_000)
    assert a['var'] == pytest.approx(b['var']) and a['es'] == pytest.approx(b['es'])


def test_es_not_smaller_than_var_and_higher_confidence_not_smaller():
    lr, w = make_log_returns(), np.full(5, 0.2)
    pr = portfolio_returns_series(lr, w)
    for cl in (0.95, 0.99):
        assert historical_es(pr, cl) >= historical_var(pr, cl)
        assert parametric_es(lr, w, cl) >= parametric_var(lr, w, cl)
        assert monte_carlo_full(lr, w, cl, 20_000)['es'] >= monte_carlo_full(lr, w, cl, 20_000)['var']
    assert historical_var(pr, 0.99) >= historical_var(pr, 0.95)


def test_weight_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        portfolio_returns_series(make_log_returns(), np.array([0.5, 0.5]))


# ---------------------------------------------------------------- EWMA
def test_ewma_matches_explicit_loop_and_is_not_double_lagged():
    pr = portfolio_returns_series(make_log_returns(), np.full(5, 0.2))
    r, lam, burn = pr.values, 0.94, 250
    s2 = {burn: np.var(r[:burn])}                    # independent re-implementation
    for t in range(burn + 1, 400):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    f = rolling_ewma_var(pr, 250, 0.95)
    for t in (250, 251, 300, 399):
        assert f.iloc[t] == pytest.approx(-norm.ppf(0.05) * math.sqrt(s2[t]))


def test_ewma_current_includes_last_return():
    pr = portfolio_returns_series(make_log_returns(), np.full(5, 0.2))
    last_forecast = ewma_variance_series(pr)[-1]
    vol, _, _ = ewma_var_current(pr, 0.95)
    assert vol == pytest.approx(math.sqrt(0.94 * last_forecast + 0.06 * pr.iloc[-1] ** 2))


# ---------------------------------------------------------------- no look-ahead
@pytest.mark.parametrize("fn", [rolling_historical_var, rolling_parametric_var, rolling_ewma_var])
def test_forecast_at_t_ignores_data_from_t_onwards(fn):
    pr = portfolio_returns_series(make_log_returns(), np.full(5, 0.2))
    t0 = 700
    changed = pr.copy()
    changed.iloc[t0:] *= 5                           # rewrite the 'future'
    a, b = fn(pr, 250, 0.99), fn(changed, 250, 0.99)
    pd.testing.assert_series_equal(a.iloc[: t0 + 1], b.iloc[: t0 + 1])   # forecasts up to day t0 unchanged
    assert not a.iloc[t0 + 1:].equals(b.iloc[t0 + 1:])                    # later forecasts do change (sanity check)


def test_all_methods_are_evaluated_on_identical_dates():
    pr = portfolio_returns_series(make_log_returns(), np.full(5, 0.2))
    res = backtest_all_methods(pr, 0.95)
    idx = [r['var_series'].index for r in res.values()]
    assert all(i.equals(idx[0]) for i in idx)
    assert "Monte Carlo" not in res


# ---------------------------------------------------------------- Kupiec / Christoffersen
def test_kupiec_hand_value():
    # 250 days, 10 breaches, 95% VaR: LR = -2[10 ln(.05/.04) + 240 ln(.95/.96)]
    lr = -2 * (10 * math.log(0.05 / 0.04) + 240 * math.log(0.95 / 0.96))
    res = kupiec_test(250, 10, 0.95)
    assert res['lr'] == pytest.approx(lr) and lr == pytest.approx(0.5633, abs=1e-3)
    assert res['p_value'] == pytest.approx(chi2.sf(lr, 1)) and res['passed'] is True


def test_kupiec_zero_breaches_is_defined_and_fails_for_being_too_cautious():
    res = kupiec_test(250, 0, 0.95)
    assert res['lr'] == pytest.approx(-2 * 250 * math.log(0.95))
    assert res['passed'] is False


def test_christoffersen_clustered_fails_for_a_real_reason():
    res = christoffersen_test(pd.Series([0] * 90 + [1] * 10))
    assert np.isfinite(res['p_value']) and res['p_value'] < 0.05 and res['passed'] is False


def test_christoffersen_isolated_breaches_not_treated_as_failure():
    b = np.zeros(300, dtype=int); b[::15] = 1          # never two breaches in a row
    res = christoffersen_test(pd.Series(b))
    assert np.isfinite(res['p_value']) and res['passed'] is True and res['n11'] == 0


def test_christoffersen_hand_value():
    # transitions in this sequence: n00=5, n01=2, n10=1, n11=1
    seq = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    res = christoffersen_test(pd.Series(seq))
    n00, n01, n10, n11 = res['n00'], res['n01'], res['n10'], res['n11']
    assert (n00, n01, n10, n11) == (5, 2, 1, 1)
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 9
    ll1 = n00*math.log(1-pi01) + n01*math.log(pi01) + n10*math.log(1-pi11) + n11*math.log(pi11)
    ll0 = (n00+n10)*math.log(1-pi) + (n01+n11)*math.log(pi)
    assert res['lr'] == pytest.approx(-2 * (ll0 - ll1))


def test_verdict_labels():
    assert verdict(True, True) == "PASS"
    assert verdict(True, False) == "CLUSTERED"
    assert verdict(False, True) == "FAIL" and verdict(False, False) == "FAIL"


# ---------------------------------------------------------------- stress testing
def test_stress_does_not_crash_when_history_is_missing():
    pr = portfolio_returns_series(make_log_returns(n=2500, start="2015-01-05"), np.full(5, 0.2))
    out = stress_scenarios(pr.loc["2018-10-15":])       # data starts inside the 2018 window
    assert "2018 Selloff" in out and np.isnan(out["2018 Selloff"]["pre_var_99"])
    assert out["2018 Selloff"]["breaches"] is None


def test_stress_total_is_compounded():
    idx = pd.bdate_range("2020-02-20", periods=6)
    pr = pd.Series([-0.1, -0.1, 0.0, 0.0, 0.0, 0.0], index=idx)
    out = stress_scenarios(pr, {"x": ("2020-02-20", "2020-02-27")})
    assert out["x"]["total"] == pytest.approx((0.9 * 0.9 - 1) * 100)


def test_drawdown_episodes_are_distinct_and_correct():
    idx = pd.bdate_range("2020-01-01", periods=12)
    r = pd.Series([0, -0.1, -0.1, 0.3, 0.1, 0, -0.05, -0.05, 0.2, 0.2, 0, 0], index=idx)
    t = worst_drawdown_windows(r, 5)
    assert len(t) == 2 and t.loc[0, 'max_drawdown_pct'] == pytest.approx(-19.0)
    assert t.loc[0, 'trough_date'] != t.loc[1, 'trough_date']


# ---------------------------------------------------------------- data loading / quality
def test_loader_returns_columns_in_requested_order_and_rejects_missing(tmp_path, monkeypatch):
    lr = make_log_returns(n=50)
    f = tmp_path / "log_returns.csv"
    lr[["GS", "AAPL", "XOM", "JPM", "JNJ"]].to_csv(f)          # file in a different order
    monkeypatch.setattr(loader, "CSV_CANDIDATES", [str(f)])
    out = loader.load_data(["AAPL", "JPM", "XOM", "JNJ", "GS"], "2016-01-01", "2030-01-01")
    assert list(out.columns) == ["AAPL", "JPM", "XOM", "JNJ", "GS"]
    pd.testing.assert_series_equal(out["GS"], lr["GS"], check_freq=False)
    with pytest.raises(ValueError):
        loader.load_data(["AAPL", "TSLA"], "2016-01-01", "2030-01-01")


def test_data_quality_catches_problems():
    lr = make_log_returns(n=1200)
    assert check_data_quality(lr)['ok'] and not check_data_quality(lr)['issues']
    dup = pd.concat([lr, lr.iloc[:1]])
    assert not check_data_quality(dup)['ok']
    bad = lr.copy(); bad.iloc[100, 0] = -0.9                    # looks like an unadjusted split
    assert any("Very large" in i['message'] for i in check_data_quality(bad)['issues'])
    assert not check_data_quality(lr.iloc[:100])['ok']          # too short


# ---------------------------------------------------------------- report is computed, not hard-coded
def test_report_text_is_computed_from_results():
    from src.reporting.report_generator import (
        generate_report_data, render_pdf_report, render_json_report)
    import json
    outs = []
    for seed, scale in ((1, 1.0), (2, 2.5)):
        lr = make_log_returns(n=1600, seed=seed, start="2015-01-05") * scale
        w = np.full(5, 0.2)
        pr = portfolio_returns_series(lr, w)
        d = generate_report_data(lr, pr, w, list(lr.columns), {t: .2 for t in lr.columns},
                                 1_000_000, 5_000, "a", "b")
        v95 = historical_var(pr, 0.95) * 1_000_000
        assert f"${v95:,.0f}" in d['narrative']['plain'][0]        # number comes from the data
        assert render_pdf_report(d)[:5] == b"%PDF-"
        json.loads(render_json_report(d))                           # strict JSON, no NaN
        outs.append(d['narrative']['plain'][0])
    assert outs[0] != outs[1]
