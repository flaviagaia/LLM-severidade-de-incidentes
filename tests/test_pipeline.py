import json
import sys
from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METRICS_PATH, MODEL_DIR, PREDICTIONS_PATH, SUMMARY_PATH
from src.pipeline import run_pipeline


class PipelineTestCase(unittest.TestCase):
    def test_pipeline_runs(self):
        if not METRICS_PATH.exists() or not PREDICTIONS_PATH.exists() or not MODEL_DIR.exists():
            summary = run_pipeline()
        else:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            summary.update(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
        self.assertTrue(METRICS_PATH.exists())
        self.assertTrue(PREDICTIONS_PATH.exists())
        self.assertTrue(MODEL_DIR.exists())
        self.assertGreater(summary["accuracy"], 0.6)
        self.assertGreater(len(pd.read_csv(PREDICTIONS_PATH)), 50)


if __name__ == "__main__":
    unittest.main()
