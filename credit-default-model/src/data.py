"""Loading, cleaning and validating the UCI 'Default of Credit Card Clients' data."""
import pandas as pd

from src import config as C

RAW_TARGET = "default payment next month"

EXPECTED_COLUMNS = (
    [C.ID_COL, "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]
    + C.BILL_AMT + C.PAY_AMT + [RAW_TARGET]
)


class DataValidationError(ValueError):
    """Raised when the input data violates an expectation."""


def load_raw(path=C.RAW_DATA) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_raw(df: pd.DataFrame) -> None:
    """Fail loudly if the raw file is not what the pipeline expects."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")
    if df[EXPECTED_COLUMNS].isna().any().any():
        raise DataValidationError("Unexpected missing values")
    if df[C.ID_COL].duplicated().any():
        raise DataValidationError("Duplicate client IDs")
    if not set(df[RAW_TARGET].unique()) <= {0, 1}:
        raise DataValidationError("Target must be binary 0/1")
    if (df["LIMIT_BAL"] <= 0).any():
        raise DataValidationError("LIMIT_BAL must be positive")
    if not df["AGE"].between(18, 100).all():
        raise DataValidationError("AGE outside plausible range")
    if not set(df["SEX"].unique()) <= {1, 2}:
        raise DataValidationError("SEX has undocumented codes")
    pay_cols = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]
    if not df[pay_cols].stack().between(-2, 9).all():
        raise DataValidationError("Repayment status outside [-2, 9]")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented fixes. Returns a new frame; raw data is never modified.

    * PAY_0 -> PAY_1 so that PAY_1..PAY_6 are consistently numbered (1 = most recent month).
    * Undocumented EDUCATION codes (0, 5, 6) are merged into 4 ("others").
    * Undocumented MARRIAGE code 0 is merged into 3 ("others").
    * Target renamed to `default`.
    Exact duplicate rows (35 of them, ignoring ID) are kept: they are plausible
    distinct clients with identical profiles, and removing them has no effect on results.
    """
    out = df.rename(columns={"PAY_0": "PAY_1", RAW_TARGET: C.TARGET}).copy()
    out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    out["MARRIAGE"] = out["MARRIAGE"].replace({0: 3})
    return out


def load_clean() -> pd.DataFrame:
    raw = load_raw()
    validate_raw(raw)
    return clean(raw)
