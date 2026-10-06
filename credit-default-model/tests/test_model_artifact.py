"""Smoke test on the trained model. Skipped if `make train` has not been run."""
import pytest

from src import config as C
from src.data import load_raw

pytestmark = pytest.mark.skipif(not (C.MODEL_DIR / "model.joblib").exists(), reason="model not trained yet")


def test_model_scores_raw_rows_and_probabilities_valid():
    from src.predict import score
    out = score(load_raw().head(50))
    assert len(out) == 50 and out["pd"].between(0, 1).all() and set(out["flag"]) <= {0, 1}
