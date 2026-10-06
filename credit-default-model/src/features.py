"""Stateless feature engineering.

Every feature uses only information available at the end of September 2005
(statement and repayment history for Apr-Sep 2005); the target is default in October 2005,
so there is no look-ahead. Because the transformation is stateless (no fitted statistics),
it can safely sit inside an sklearn Pipeline without leaking information across CV folds.
"""
import numpy as np
import pandas as pd

from src import config as C


def _safe_ratio(num: pd.Series, den: pd.Series, fill: float) -> pd.Series:
    out = num / den.where(den > 0)
    return out.fillna(fill)


def build_features(df: pd.DataFrame, exclude=C.PROTECTED) -> pd.DataFrame:
    """Return the model matrix from a cleaned frame (drops ID, target and excluded columns)."""
    X = df.drop(columns=[c for c in [C.ID_COL, C.TARGET] if c in df.columns]).copy()
    limit = X["LIMIT_BAL"]

    # Credit utilisation per month (statement balance / credit limit).
    for i in range(1, 7):
        X[f"UTIL_{i}"] = X[f"BILL_AMT{i}"] / limit
    util = X[[f"UTIL_{i}" for i in range(1, 7)]]
    X["UTIL_MEAN"] = util.mean(axis=1)
    X["UTIL_MAX"] = util.max(axis=1)

    # Share of the previous month's statement that was repaid (PAY_AMT{i} pays BILL_AMT{i+1}).
    # If nothing was owed the share is set to 1 (fully paid); capped at 1 for over-payments.
    for i in range(1, 6):
        X[f"PAID_RATIO_{i}"] = _safe_ratio(X[f"PAY_AMT{i}"], X[f"BILL_AMT{i+1}"], fill=1.0).clip(0, 1)
    X["PAID_RATIO_MEAN"] = X[[f"PAID_RATIO_{i}" for i in range(1, 6)]].mean(axis=1)

    # Repayment-delay behaviour (values > 0 mean payment was late by that many months).
    status = X[C.PAY_STATUS]
    X["DELAY_MAX"] = status.max(axis=1)
    X["DELAY_MEAN"] = status.mean(axis=1)
    X["N_MONTHS_LATE"] = (status > 0).sum(axis=1)
    X["DELAY_TREND"] = X["PAY_1"] - X["PAY_6"]  # >0: getting worse recently

    # Trend in balances, scaled by limit so it is comparable across clients.
    X["BILL_TREND"] = (X["BILL_AMT1"] - X["BILL_AMT6"]) / limit
    X["PAY_TO_BILL_TOTAL"] = _safe_ratio(X[C.PAY_AMT].sum(axis=1), X[C.BILL_AMT].clip(lower=0).sum(axis=1), 1.0).clip(0, 5)

    X = X.drop(columns=[c for c in exclude if c in X.columns])
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
