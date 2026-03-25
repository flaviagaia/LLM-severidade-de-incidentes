from __future__ import annotations

import json

from .data_generation import prepare_dataset
from .config import METRICS_PATH, MODEL_DIR, PREDICTIONS_PATH, SUMMARY_PATH
from .training import train_lora_model


def _artifacts_ready() -> bool:
    return all(path.exists() for path in [SUMMARY_PATH, METRICS_PATH, PREDICTIONS_PATH, MODEL_DIR])


def load_existing_summary() -> dict:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    summary.update(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    return summary


def run_pipeline(force_retrain: bool = False) -> dict:
    if _artifacts_ready() and not force_retrain:
        return load_existing_summary()

    dataset_summary = prepare_dataset()
    artifacts = train_lora_model()
    return {**dataset_summary, **artifacts.metrics}
