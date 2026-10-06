import pandas as pd
import pytest

from src import config as C
from src.data import DataValidationError, clean, load_raw, validate_raw


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_raw_file_passes_validation(raw):
    validate_raw(raw)
    assert raw.shape == (30000, 25)


def test_validation_catches_bad_target(raw):
    bad = raw.copy()
    bad.iloc[0, bad.columns.get_loc("default payment next month")] = 2
    with pytest.raises(DataValidationError):
        validate_raw(bad)


def test_validation_catches_missing_column(raw):
    with pytest.raises(DataValidationError):
        validate_raw(raw.drop(columns=["LIMIT_BAL"]))


def test_validation_catches_duplicate_ids(raw):
    bad = pd.concat([raw, raw.iloc[[0]]])
    with pytest.raises(DataValidationError):
        validate_raw(bad)


def test_clean_recodes_categories_and_renames(raw):
    c = clean(raw)
    assert set(c["EDUCATION"].unique()) == {1, 2, 3, 4}
    assert set(c["MARRIAGE"].unique()) == {1, 2, 3}
    assert "PAY_1" in c and "PAY_0" not in c and C.TARGET in c
    assert len(c) == len(raw) and raw["EDUCATION"].isin([0, 5, 6]).any()  # raw untouched
