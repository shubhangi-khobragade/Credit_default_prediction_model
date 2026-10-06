"""Central configuration: paths, constants and modelling choices."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "default_of_credit_card_clients.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "default"  # renamed from "default payment next month"
ID_COL = "ID"

PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]   # PAY_1 = Sept 2005 (most recent) ... PAY_6 = Apr 2005
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]

# Attributes that are legally sensitive in consumer-credit decisions (e.g. ECOA in the US).
# They are EXCLUDED from the model by default but still used to audit subgroup performance.
PROTECTED = ["SEX", "MARRIAGE"]

# Illustrative business assumption used ONLY to pick an operating threshold.
# A missed defaulter (false negative) is assumed to cost 5x a wrongly flagged good client.
COST_FN = 5.0
COST_FP = 1.0
