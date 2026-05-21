"""Streamlit evaluation dashboard page for Retriva query logs."""

import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

def fetch_eval_logs():
    """Fetch evaluation logs from the backend."""

    response = requests.get(f"{BACKEND_URL}/eval_logs", timeout=30)
    response.raise_for_status()
    return response.json()


def recompute_pending_scores():
    """Ask the backend to re-run pending Ragas scores."""

    response = requests.post(f"{BACKEND_URL}/eval_logs/recompute", timeout=30)
    response.raise_for_status()
    return response.json()


def average_metric(logs, metric):
    """Return the average metric value, ignoring pending scores."""

    values = [
        float(row[metric])
        for row in logs
        if row.get(metric) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def metric_value(value):
    """Format a metric value for display."""

    return "Pending" if value is None else f"{value:.2f}"


def build_table(logs):
    """Build readable table rows for logged evaluations."""

    rows = []
    for row in logs:
        rows.append(
            {
                "id": row.get("id"),
                "timestamp": row.get("timestamp"),
                "question": row.get("question"),
                "grade_score": row.get("grade_score"),
                "faithfulness": row.get("faithfulness"),
                "answer_relevancy": row.get("answer_relevancy"),
                "context_precision": row.get("context_precision"),
                "context_recall": row.get("context_recall"),
            }
        )
    return rows


st.title("Evaluation Dashboard")
st.caption("Ragas metrics and CRAG context grades for logged Retriva queries")

refresh_col, recompute_col = st.columns([1, 3])
if refresh_col.button("Refresh"):
    st.rerun()
if recompute_col.button("Re-run pending scores"):
    try:
        result = recompute_pending_scores()
        st.success(f"Queued {result.get('rows', 0)} pending row(s) for evaluation.")
    except requests.RequestException as exc:
        st.error(f"Could not queue evaluation: {exc}")

try:
    logs = fetch_eval_logs()
except requests.RequestException as exc:
    st.error(f"Could not load evaluation logs: {exc}")
    st.stop()

averages = {metric: average_metric(logs, metric) for metric in METRICS}

st.divider()

cols = st.columns(5)
cols[0].metric("Total queries", len(logs))
cols[1].metric("Faithfulness", metric_value(averages["faithfulness"]))
cols[2].metric("Answer relevancy", metric_value(averages["answer_relevancy"]))
cols[3].metric("Context precision", metric_value(averages["context_precision"]))
cols[4].metric("Context recall", metric_value(averages["context_recall"]))

pending_count = sum(
    1
    for row in logs
    if any(row.get(metric) is None for metric in METRICS)
)
if pending_count:
    st.info(
        f"{pending_count} logged query row(s) still have at least one pending "
        "Ragas score. Re-run pending scores, then refresh after the evaluator "
        "finishes."
    )

chart_rows = [
    {"metric": metric.replace("_", " ").title(), "score": score}
    for metric, score in averages.items()
    if score is not None
]
if chart_rows:
    chart_df = pd.DataFrame(chart_rows).set_index("metric")
    st.subheader("Average Ragas Scores")
    st.bar_chart(chart_df)

st.subheader("Logged Queries")
if logs:
    st.dataframe(
        build_table(logs),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write("No queries logged yet.")
