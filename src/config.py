from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ASSETS_DIR = BASE_DIR / "assets"

DATASET_PATH = RAW_DIR / "incident_severity_synthetic.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VALID_PATH = PROCESSED_DIR / "valid.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
METRICS_PATH = PROCESSED_DIR / "metrics.json"
PREDICTIONS_PATH = PROCESSED_DIR / "test_predictions.csv"
SUMMARY_PATH = PROCESSED_DIR / "summary.json"
MODEL_DIR = ARTIFACTS_DIR / "severity_llm_lora"

BASE_MODEL_NAME = "google/flan-t5-small"
LABELS = ["critical", "high", "normal"]
