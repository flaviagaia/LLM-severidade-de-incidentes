import json
from pathlib import Path

from src.pipeline import run_pipeline


if __name__ == "__main__":
    summary = run_pipeline(force_retrain=False)
    print("LLM Severidade de Incidentes")
    print("-" * 40)
    print(json.dumps(summary, indent=2))
