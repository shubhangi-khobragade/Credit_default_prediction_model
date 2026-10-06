import numpy as np
import pandas as pd

from src import config as C
from src.data import load_clean
from src.features import build_features


def test_no_target_or_id_or_protected_in_features():
    X = build_features(load_clean())
    for col in [C.TARGET, C.ID_COL, *C.PROTECTED]:
        assert col not in X.columns


def test_features_are_finite_and_row_count_preserved():
    df = load_clean()
    X = build_features(df)
    assert len(X) == len(df)
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_features_do_not_depend_on_target():
    """Shuffling the target must not change a single feature value (no leakage)."""
    df = load_clean()
    shuffled = df.assign(**{C.TARGET: np.random.default_rng(0).permutation(df[C.TARGET].to_numpy())})
    pd.testing.assert_frame_equal(build_features(df), build_features(shuffled))


def test_features_are_row_independent():
    """Stateless: a row's features must be identical whether computed alone or in a batch."""
    df = load_clean().head(200)
    full = build_features(df)
    single = pd.concat([build_features(df.iloc[[i]]) for i in (0, 57, 199)])
    pd.testing.assert_frame_equal(full.iloc[[0, 57, 199]], single)


def test_utilisation_and_paid_ratio_hand_computed():
    row = {c: 0 for c in ["LIMIT_BAL", "EDUCATION", "AGE", *C.PAY_STATUS, *C.BILL_AMT, *C.PAY_AMT]}
    row.update({"LIMIT_BAL": 1000, "BILL_AMT1": 500, "BILL_AMT2": 400, "PAY_AMT1": 200, "PAY_1": 2,
                "SEX": 1, "MARRIAGE": 1, "EDUCATION": 2, "AGE": 30})
    X = build_features(pd.DataFrame([row]))
    assert X.loc[0, "UTIL_1"] == 0.5
    assert X.loc[0, "PAID_RATIO_1"] == 0.5          # 200 paid against previous bill of 400
    assert X.loc[0, "PAID_RATIO_2"] == 1.0          # nothing owed -> treated as fully paid
    assert X.loc[0, "DELAY_MAX"] == 2 and X.loc[0, "N_MONTHS_LATE"] == 1
