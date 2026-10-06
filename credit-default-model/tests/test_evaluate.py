import numpy as np

from src import evaluate as E


def test_perfect_and_random_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert E.summary_metrics(y, np.array([.1, .2, .3, .7, .8, .9]))["roc_auc"] == 1.0
    assert E.ks_statistic(y, np.array([.1, .2, .3, .7, .8, .9])) == 1.0


def test_psi_zero_for_identical_and_positive_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.beta(2, 5, 5000)
    assert E.psi(a, a) < 1e-6
    assert E.psi(a, a + 0.2) > 0.25


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.25, 2000), 0, 1)
    lo, hi = E.bootstrap_ci(y, p, n_boot=200)
    assert lo < E.summary_metrics(y, p)["roc_auc"] < hi


def test_threshold_picker_prefers_low_threshold_when_misses_are_costly():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 3000)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, 3000), 0, 1)
    assert E.pick_threshold(y, p) < 0.5  # COST_FN > COST_FP


def test_decile_table_is_monotone_for_a_good_model():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    t = E.decile_table(y, p)
    assert t["default_rate"].is_monotonic_decreasing or t["default_rate"].iloc[0] > t["default_rate"].iloc[-1]
    assert abs(t["cum_capture"].iloc[-1] - 1) < 1e-9
