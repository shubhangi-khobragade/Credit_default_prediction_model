"""Train, validate and report the credit-default model.

"""
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config as C
from src import evaluate as E
from src.data import load_clean
from src.features import build_features


def make_pipeline(model, exclude=C.PROTECTED) -> Pipeline:
    feats = FunctionTransformer(build_features, kw_args={"exclude": exclude})
    return Pipeline([("features", feats), ("model", model)])


def main():
    t0 = time.time()
    for d in (C.FIG_DIR, C.TABLE_DIR, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data + split
    df = load_clean()
    y = df[C.TARGET]
    X = df.drop(columns=[C.TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)
    print(f"Train {X_tr.shape[0]:,} rows (default rate {y_tr.mean():.3%}) | "
          f"Test {X_te.shape[0]:,} rows (default rate {y_te.mean():.3%})")
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)

    # ------------------------------------------------------------------ candidate models (CV on train only)
    lr = GridSearchCV(
        make_pipeline(Pipeline([("scale", StandardScaler()),
                                ("clf", LogisticRegression(max_iter=2000))])),
        {"model__clf__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc", cv=cv, n_jobs=-1)
    gbm = RandomizedSearchCV(
        make_pipeline(HistGradientBoostingClassifier(early_stopping=True, n_iter_no_change=20,
                                                     max_iter=500, random_state=C.SEED)),
        {"model__learning_rate": [0.02, 0.05, 0.1],
         "model__max_leaf_nodes": [8, 15, 31],
         "model__max_depth": [None, 3, 4, 6],
         "model__min_samples_leaf": [20, 50, 100, 200],
         "model__l2_regularization": [0.0, 0.1, 1.0, 10.0]},
        n_iter=20, scoring="roc_auc", cv=cv, n_jobs=-1, random_state=C.SEED)

    results = {}
    for name, search in [("logistic_regression", lr), ("gradient_boosting", gbm)]:
        search.fit(X_tr, y_tr)
        i = search.best_index_
        results[name] = {"cv_auc_mean": float(search.cv_results_["mean_test_score"][i]),
                         "cv_auc_std": float(search.cv_results_["std_test_score"][i]),
                         "best_params": {k.replace("model__", ""): (v if not isinstance(v, np.generic) else v.item())
                                         for k, v in search.best_params_.items()}}
        results[name]["_search"] = search
        print(f"{name:22s} CV AUC {results[name]['cv_auc_mean']:.4f} +/- {results[name]['cv_auc_std']:.4f}  "
              f"{results[name]['best_params']}")

    champion_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
    champion = results[champion_name]["_search"].best_estimator_
    challenger_name = [k for k in results if k != champion_name][0]
    challenger = results[challenger_name]["_search"].best_estimator_
    print(f"Champion (selected on CV AUC): {champion_name}")

    # ------------------------------------------------------------------ out-of-fold predictions on train
    oof = cross_val_predict(champion, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = E.pick_threshold(y_tr, oof)                       # chosen WITHOUT looking at test
    E.plot_threshold_cost(y_tr, oof, thr)

    # ------------------------------------------------------------------ one-time test evaluation
    p_te = champion.predict_proba(X_te)[:, 1]
    p_tr = champion.predict_proba(X_tr)[:, 1]
    p_ch = challenger.predict_proba(X_te)[:, 1]
    naive = X_te["PAY_1"].to_numpy().astype(float)          # benchmark: latest repayment status alone

    test_metrics = E.summary_metrics(y_te, p_te)
    auc_lo, auc_hi = E.bootstrap_ci(y_te, p_te)
    test_metrics.update({"roc_auc_ci95": [auc_lo, auc_hi], "ece": E.expected_calibration_error(y_te, p_te)})
    train_auc = float(roc_auc_score(y_tr, p_tr))

    # Paired bootstrap on the AUC difference between champion and challenger.
    rng = np.random.default_rng(C.SEED)
    yv, pa, pb = y_te.to_numpy(), p_te, p_ch
    diffs = []
    for _ in range(1000):
        idx = rng.integers(0, len(yv), len(yv))
        diffs.append(roc_auc_score(yv[idx], pa[idx]) - roc_auc_score(yv[idx], pb[idx]))
    diff_ci = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]

    ops = {"chosen": E.confusion_at(y_te, p_te, thr), "default_0.5": E.confusion_at(y_te, p_te, 0.5),
           "flag_nobody": E.confusion_at(y_te, np.zeros(len(y_te)), 0.5),
           "flag_everybody": E.confusion_at(y_te, np.ones(len(y_te)), 0.5)}

    # ------------------------------------------------------------------ protected-attribute ablation
    with_prot = make_pipeline(clone(champion.named_steps["model"]), exclude=[]).fit(X_tr, y_tr)
    auc_with_prot = float(roc_auc_score(y_te, with_prot.predict_proba(X_te)[:, 1]))

    # ------------------------------------------------------------------ tables
    E.decile_table(y_te, p_te).to_csv(C.TABLE_DIR / "decile_table.csv", index=False)
    groups = pd.DataFrame({
        "SEX": X_te["SEX"].map({1: "male", 2: "female"}),
        "MARRIAGE": X_te["MARRIAGE"].map({1: "married", 2: "single", 3: "other"}),
        "EDUCATION": X_te["EDUCATION"].map({1: "graduate school", 2: "university", 3: "high school", 4: "other"}),
        "AGE_BAND": pd.cut(X_te["AGE"], [0, 29, 39, 49, 120], labels=["<30", "30-39", "40-49", "50+"]),
    })
    sub = E.subgroup_table(groups, y_te.to_numpy(), p_te, thr)
    sub.to_csv(C.TABLE_DIR / "subgroup_performance.csv", index=False)

    # Permutation importance on the engineered feature matrix (test set, AUC drop).
    Xt = champion[:-1].transform(X_te)
    pi = permutation_importance(champion[-1], Xt, y_te, scoring="roc_auc", n_repeats=10,
                                random_state=C.SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": Xt.columns, "auc_drop_mean": pi.importances_mean,
                         "auc_drop_std": pi.importances_std})
           .sort_values("auc_drop_mean", ascending=False).reset_index(drop=True))
    imp.round(5).to_csv(C.TABLE_DIR / "permutation_importance.csv", index=False)

    # ------------------------------------------------------------------ figures
    E.plot_roc({f"{champion_name.replace('_', ' ')}": (y_te, p_te),
                f"{challenger_name.replace('_', ' ')}": (y_te, p_ch),
                "PAY_1 only (naive)": (y_te, naive)})
    E.plot_calibration(y_te, p_te)
    E.plot_gains(y_te, p_te)
    E.plot_score_distribution(y_te, p_te)
    E.plot_importance(imp)

    # ------------------------------------------------------------------ metrics.json
    metrics = {
        "data": {"n_rows": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                 "default_rate": float(y.mean()), "n_model_features": int(Xt.shape[1])},
        "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "_search"} for k, v in results.items()},
        "champion": champion_name,
        "train_auc": train_auc,
        "test": test_metrics,
        "test_naive_pay1_auc": float(roc_auc_score(y_te, naive)),
        "test_challenger_auc": float(roc_auc_score(y_te, p_ch)),
        "champion_minus_challenger_auc_ci95": diff_ci,
        "score_psi_train_vs_test": E.psi(p_tr, p_te),
        "threshold": {"chosen": thr, "cost_fn": C.COST_FN, "cost_fp": C.COST_FP, "operating_points_test": ops},
        "ablation_auc_with_protected_attributes": auc_with_prot,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__, "seed": C.SEED},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"pipeline": champion, "threshold": thr, "metrics": metrics}, C.MODEL_DIR / "model.joblib")

    print(f"\nTRAIN AUC {train_auc:.4f} | TEST AUC {test_metrics['roc_auc']:.4f} "
          f"(95% CI {auc_lo:.4f}-{auc_hi:.4f}) | KS {test_metrics['ks']:.3f} | PR-AUC {test_metrics['pr_auc']:.3f}")
    print(f"Naive PAY_1-only AUC {metrics['test_naive_pay1_auc']:.4f} | PSI train->test {metrics['score_psi_train_vs_test']:.4f}")
    print(f"Done in {metrics['runtime_seconds']}s. Artefacts in reports/ and models/.")


if __name__ == "__main__":
    main()
