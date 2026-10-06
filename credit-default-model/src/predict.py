"""Score new clients with the trained model.
"""
import sys

import joblib
import pandas as pd

from src import config as C
from src.data import RAW_TARGET, clean


def score(df_raw: pd.DataFrame, model_path=C.MODEL_DIR / "model.joblib") -> pd.DataFrame:
    bundle = joblib.load(model_path)
    df = clean(df_raw.assign(**{RAW_TARGET: df_raw.get(RAW_TARGET, 0)}))
    pd_hat = bundle["pipeline"].predict_proba(df.drop(columns=[C.TARGET]))[:, 1]
    out = pd.DataFrame({C.ID_COL: df[C.ID_COL], "pd": pd_hat})
    out["flag"] = (out["pd"] >= bundle["threshold"]).astype(int)
    return out


if __name__ == "__main__":
    src = sys.argv[1]
    res = score(pd.read_csv(src))
    if len(sys.argv) > 2:
        res.to_csv(sys.argv[2], index=False)
    else:
        print(res.head(20).to_string(index=False))
