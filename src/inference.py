from __future__ import annotations

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import BASE_MODEL_NAME, MODEL_DIR
from .training import _format_example


def load_trained_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.to("cpu")
    return tokenizer, model


def predict_severity(tokenizer, model, incident_text: str) -> str:
    prompt = _format_example(incident_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=192)
    output = model.generate(**inputs, max_new_tokens=4)
    label = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
    return label if label in {"critical", "high", "normal"} else "normal"
