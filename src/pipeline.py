from __future__ import annotations

from .data_generation import prepare_dataset
from .training import train_lora_model


def run_pipeline() -> dict:
    dataset_summary = prepare_dataset()
    artifacts = train_lora_model()
    return {**dataset_summary, **artifacts.metrics}
