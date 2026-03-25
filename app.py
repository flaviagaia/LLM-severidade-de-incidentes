from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import METRICS_PATH, MODEL_DIR, PREDICTIONS_PATH, SUMMARY_PATH
from src.inference import load_trained_model, predict_severity
from src.pipeline import run_pipeline


st.set_page_config(page_title="LLM Severidade de Incidentes", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #07111f; color: #e5eef9; }
    .hero {
        background: rgba(10, 18, 32, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 22px;
        padding: 1.2rem 1.3rem;
    }
    .hero h1, .hero p { color: #e5eef9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>LLM para Classificação de Severidade de Incidentes</h1>
        <p>Fine-tuning leve com LoRA sobre FLAN-T5 para classificar relatórios operacionais em severidade crítica, alta ou normal.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Treinar / atualizar modelo"):
    run_pipeline(force_retrain=True)

if not METRICS_PATH.exists() or not PREDICTIONS_PATH.exists() or not MODEL_DIR.exists():
    run_pipeline(force_retrain=True)

summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
predictions = pd.read_csv(PREDICTIONS_PATH)
tokenizer, model = load_trained_model()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Amostras totais", summary["rows"])
col2.metric("Base model", metrics["base_model"])
col3.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col4.metric("Macro F1", f"{metrics['macro_f1']:.3f}")

tab_predict, tab_eval = st.tabs(["Simular Incidente", "Avaliação"])

with tab_predict:
    incident_text = st.text_area(
        "Relato do incidente",
        value="Incident report: the payment gateway in the primary datacenter shows complete outage. Impact: transactions are blocked for all customers. Additional note: executive bridge activated. Reported by the payments team during the morning shift and observed for over an hour.",
        height=180,
    )
    if st.button("Prever severidade", use_container_width=True):
        prediction = predict_severity(tokenizer, model, incident_text)
        st.success(f"Severidade prevista: {prediction}")

with tab_eval:
    st.plotly_chart(
        px.histogram(predictions, x="actual_severity", color="predicted_severity", barmode="group", title="Predições no conjunto de teste"),
        use_container_width=True,
    )
    st.dataframe(predictions.head(30), use_container_width=True, hide_index=True)
