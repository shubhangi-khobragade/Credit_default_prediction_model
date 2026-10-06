"""Validation utilities: metrics, uncertainty, calibration, stability, fairness audit and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)

from src import config as C

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, ORANGE, GREY = "#1f5fa8", "#d9822b", "#8a8a8a"


# --------------------------------------------------------------------------- metrics
def ks_statistic(y, p) -> float:
    """Max separation between the score CDFs of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summary_metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


def bootstrap_ci(y, p, metric=roc_auc_score, n_boot=1000, seed=C.SEED, alpha=0.05):
    """Percentile bootstrap confidence interval for a metric on the test set."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def psi(expected, actual, bins=10) -> float:
    """Population Stability Index of a score between two samples (rule of thumb: <0.1 stable)."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def expected_calibration_error(y, p, bins=10) -> float:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    # quantile bins hold ~equal counts, so a simple mean is a fair ECE estimate
    return float(np.mean(np.abs(frac_pos - mean_pred)))


# --------------------------------------------------------------------------- thresholds
def confusion_at(y, p, thr) -> dict:
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "flag_rate": (tp + fp) / len(y),
        "cost_per_client": (C.COST_FN * fn + C.COST_FP * fp) / len(y),
    }


def pick_threshold(y, p) -> float:
    """Threshold that minimises the assumed misclassification cost (choose on OUT-OF-FOLD train scores)."""
    grid = np.linspace(0.05, 0.95, 91)
    costs = [confusion_at(y, p, t)["cost_per_client"] for t in grid]
    return float(grid[int(np.argmin(costs))])


# --------------------------------------------------------------------------- tables
def decile_table(y, p) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = d.groupby("decile", observed=True).agg(n=("y", "size"), defaults=("y", "sum"),
                                              avg_pred=("p", "mean"), default_rate=("y", "mean"))
    t["lift"] = t["default_rate"] / d["y"].mean()
    t["cum_capture"] = t["defaults"].cumsum() / d["y"].sum()
    return t.round(4).reset_index()


def subgroup_table(groups: pd.DataFrame, y, p, thr) -> pd.DataFrame:
    """Performance and flag rates by demographic group (audit only - these are not model inputs)."""
    rows = []
    y, p = np.asarray(y), np.asarray(p)
    for col in groups.columns:
        for level, idx in groups.groupby(col, observed=True).indices.items():
            if len(idx) < 100 or y[idx].min() == y[idx].max():
                continue
            c = confusion_at(y[idx], p[idx], thr)
            rows.append({"attribute": col, "group": level, "n": len(idx),
                         "default_rate": y[idx].mean(), "auc": roc_auc_score(y[idx], p[idx]),
                         "flag_rate": c["flag_rate"], "recall(TPR)": c["recall"],
                         "FPR": c["fp"] / max(c["fp"] + c["tn"], 1)})
    return pd.DataFrame(rows).round(4)


# --------------------------------------------------------------------------- plots
def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name)
    plt.close(fig)


def plot_roc(curves: dict):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for (label, (y, p)), col in zip(curves.items(), [BLUE, ORANGE, GREY]):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, label=f"{label} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="lightgrey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve - hold-out test set")
    ax.legend(loc="lower right")
    _save(fig, "roc.png")


def plot_calibration(y, p):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="lightgrey", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=BLUE, label="Model (deciles)")
    ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate", title="Calibration - hold-out test set")
    ax.legend()
    _save(fig, "calibration.png")


def plot_gains(y, p):
    order = np.argsort(-np.asarray(p))
    cum = np.cumsum(np.asarray(y)[order]) / np.sum(y)
    x = np.arange(1, len(y) + 1) / len(y)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(x, cum, color=BLUE, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="lightgrey", label="Random")
    ax.set(xlabel="Share of clients reviewed (highest risk first)", ylabel="Share of defaults captured",
           title="Cumulative gains - hold-out test set")
    ax.legend(loc="lower right")
    _save(fig, "gains.png")


def plot_score_distribution(y, p):
    y, p = np.asarray(y), np.asarray(p)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins, alpha=0.65, color=BLUE, density=True, label="Did not default")
    ax.hist(p[y == 1], bins, alpha=0.65, color=ORANGE, density=True, label="Defaulted")
    ax.set(xlabel="Predicted probability of default", ylabel="Density",
           title=f"Score distribution (KS = {ks_statistic(y, p):.3f})")
    ax.legend()
    _save(fig, "score_distribution.png")


def plot_importance(imp: pd.DataFrame, top=15):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(t["feature"], t["auc_drop_mean"], xerr=t["auc_drop_std"], color=BLUE, ecolor=GREY)
    ax.set(xlabel="Drop in test ROC-AUC when feature is shuffled", title="Permutation importance (top 15)")
    _save(fig, "feature_importance.png")


def plot_threshold_cost(y, p, thr):
    grid = np.linspace(0.05, 0.95, 91)
    costs = [confusion_at(y, p, t)["cost_per_client"] for t in grid]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(grid, costs, color=BLUE)
    ax.axvline(thr, color=ORANGE, ls="--", label=f"Chosen threshold = {thr:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Expected cost per client",
           title=f"Threshold vs cost (FN:FP = {C.COST_FN:g}:{C.COST_FP:g}), out-of-fold train")
    ax.legend()
    _save(fig, "threshold_cost.png")
