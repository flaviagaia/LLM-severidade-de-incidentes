from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .config import BASE_MODEL_NAME, METRICS_PATH, MODEL_DIR, PREDICTIONS_PATH, TEST_PATH, TRAIN_PATH, VALID_PATH


PROMPT_TEMPLATE = (
    "Classify the severity of the incident report into one of: critical, high, normal.\n"
    "Incident: {incident_text}\n"
    "Answer:"
)


def _format_example(text: str) -> str:
    return PROMPT_TEMPLATE.format(incident_text=text)


def _to_dataset(frame: pd.DataFrame) -> Dataset:
    prompts = [_format_example(text) for text in frame["incident_text"].tolist()]
    return Dataset.from_dict({"input_text": prompts, "target_text": frame["severity"].tolist()})


def _tokenize_batch(batch, tokenizer, max_source_length: int = 192, max_target_length: int = 8):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=max_source_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


@dataclass
class TrainingArtifacts:
    metrics: dict


def train_lora_model(num_train_epochs: int = 4, max_train_rows: int = 360) -> TrainingArtifacts:
    train_df = pd.read_csv(TRAIN_PATH).head(max_train_rows)
    valid_df = pd.read_csv(VALID_PATH).head(max(90, min(len(pd.read_csv(VALID_PATH)), 120)))
    test_df = pd.read_csv(TEST_PATH).head(max(90, min(len(pd.read_csv(TEST_PATH)), 120)))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, peft_config)

    train_dataset = _to_dataset(train_df)
    valid_dataset = _to_dataset(valid_df)
    test_prompts = [_format_example(text) for text in test_df["incident_text"].tolist()]

    tokenized_train = train_dataset.map(
        lambda batch: _tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_valid = valid_dataset.map(
        lambda batch: _tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=valid_dataset.column_names,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_DIR),
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        num_train_epochs=num_train_epochs,
        report_to=[],
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    trainer.model.to("cpu")
    generation_inputs = tokenizer(test_prompts, return_tensors="pt", padding=True, truncation=True, max_length=192)
    predictions = trainer.model.generate(**generation_inputs, max_new_tokens=4)
    decoded = [text.strip().lower() for text in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
    normalized = [label if label in {"critical", "high", "normal"} else "normal" for label in decoded]

    metrics = {
        "accuracy": round(accuracy_score(test_df["severity"], normalized), 4),
        "macro_f1": round(f1_score(test_df["severity"], normalized, average="macro"), 4),
        "weighted_f1": round(f1_score(test_df["severity"], normalized, average="weighted"), 4),
        "base_model": BASE_MODEL_NAME,
        "train_rows_used": int(len(train_df)),
        "test_rows_used": int(len(test_df)),
    }

    pd.DataFrame(
        {
            "incident_id": test_df["incident_id"].tolist(),
            "incident_text": test_df["incident_text"].tolist(),
            "actual_severity": test_df["severity"].tolist(),
            "predicted_severity": normalized,
        }
    ).to_csv(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return TrainingArtifacts(metrics=metrics)
