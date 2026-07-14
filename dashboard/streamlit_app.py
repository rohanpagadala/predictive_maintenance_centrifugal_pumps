"""
Streamlit dashboard for the Predictive Maintenance project.

Design split, deliberately mirroring the FastAPI/Streamlit separation in
docker-compose: **Single Prediction** and **Batch Prediction** call the
FastAPI backend over HTTP (`PDM_API_URL`) so inference logic lives in exactly
one place (`app/predict.py`). **Model Performance** and **Feature
Importance** read the shared `models/` volume directly -- static
introspection of already-loaded artifacts doesn't need a network round trip.

Run locally (with the API already running):
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Make the project root importable (pdm_utils, model_persistence, app/) when
# Streamlit runs this file directly rather than as an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_persistence as mp  # noqa: E402
from app import config  # noqa: E402

API_URL = os.getenv("PDM_API_URL", "http://localhost:8000")
CLASS_COLORS = {
    "Normal": "#0ca30c", "Warning": "#fab219", "Critical": "#d03b3b",
    "Bearing_Failure": "#e34948", "Motor_Failure": "#e87ba4", "Seal_Failure": "#eb6834",
}

st.set_page_config(page_title="PdM Dashboard", page_icon="🛠️", layout="wide")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts() -> mp.ModelArtifacts:
    """Loaded once per Streamlit server process via st.cache_resource --
    the dashboard-side equivalent of app.model_loader's singleton cache."""
    return mp.load_models(config.MODELS_DIR)


def api_health() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def call_predict(payload: dict) -> dict:
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(resp.json().get("detail", resp.text))
    return resp.json()


def call_predict_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    resp = requests.post(
        f"{API_URL}/predict-csv",
        files={"file": (filename, file_bytes, "text/csv")},
        timeout=60,
    )
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
        raise RuntimeError(detail)
    from io import StringIO
    return pd.read_csv(StringIO(resp.text))


def state_badge(state: str) -> str:
    color = CLASS_COLORS.get(state, "#898781")
    return f"<span style='background:{color};color:white;padding:4px 12px;border-radius:12px;font-weight:600'>{state}</span>"


def log_history_entry(entry: dict) -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].insert(0, entry)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home():
    st.title("🛠️ Predictive Maintenance Dashboard")
    st.caption("Industrial centrifugal pump failure classification & remaining-useful-life prediction.")

    health = api_health()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("API status", "🟢 Online" if health else "🔴 Offline")
    if health:
        col2.metric("Classifier", health.get("classifier_name", "—"))
        col3.metric("Regressor", health.get("regressor_name", "—"))
        col4.metric("Uptime (s)", f"{health.get('uptime_seconds', 0):.0f}")
    else:
        col2.warning(f"Could not reach API at {API_URL}. Start it with:\n\nuvicorn app.api:app")

    st.divider()
    st.subheader("What this dashboard does")
    st.markdown(
        """
        - **Single Prediction** — score one pump reading (optionally with recent history) manually.
        - **Batch Prediction** — upload a CSV of pump readings and download predictions for every row.
        - **Model Performance** — the trained models' evaluation metrics and plots.
        - **Feature Importance** — which engineered features drive each model's predictions.
        - **Prediction History** — a running log of predictions made this session.
        """
    )


def page_overview():
    st.title("📋 Project Overview")
    st.markdown(
        """
        ### Business problem
        Unplanned failures of industrial centrifugal pumps cause costly downtime.
        This project turns hourly pump telemetry (vibration, temperature, pressure,
        flow rate) into two decision-support signals:

        1. **Failure classification** — Normal / Warning / Critical / Bearing_Failure
           / Motor_Failure / Seal_Failure.
        2. **Remaining Useful Life (RUL) regression** — hours until failure.

        ### Pipeline stages
        Dataset Understanding → EDA → Preprocessing → Feature Engineering →
        Feature Selection → Data Preparation → Model Training (Decision Tree,
        Random Forest, XGBoost, LightGBM) → Evaluation → Comparison → Persistence →
        **this serving layer**.

        ### Known limitation, stated plainly
        The three failure-type classes (Bearing/Motor/Seal_Failure) total only
        ~20 rows in the entire training set, concentrated in a single pump.
        Predictions for those specific classes should be treated with real
        skepticism — see **About** for the full explanation.
        """
    )
    try:
        artifacts = load_artifacts()
        st.subheader("Selected feature set")
        st.dataframe(pd.DataFrame({"Feature": artifacts.feature_list}), width="stretch", height=300)
    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")


def page_single_prediction():
    st.title("🔍 Single Prediction")
    st.caption("Enter one pump reading. Add optional history rows for more reliable rolling/lag features.")

    with st.form("single_prediction_form"):
        c1, c2 = st.columns(2)
        asset_id = c1.text_input("Asset ID", value="PUMP_DEMO")
        machine_model = c2.text_input("Machine Model", value="Nova-P")

        c1, c2, c3, c4 = st.columns(4)
        vibration = c1.number_input("Vibration (mm/s)", min_value=0.0, max_value=20.0, value=2.2, step=0.1)
        temperature = c2.number_input("Temperature (°C)", min_value=0.0, max_value=200.0, value=62.0, step=0.5)
        pressure = c3.number_input("Pressure (psi)", min_value=0.0, max_value=300.0, value=145.0, step=1.0)
        flow_rate = c4.number_input("Flow Rate (m³/h)", min_value=0.0, max_value=500.0, value=250.0, step=1.0)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        payload = {
            "readings": [{
                "asset_id": asset_id,
                "machine_model": machine_model,
                "vibration_mm_s": vibration,
                "temperature_c": temperature,
                "pressure_psi": pressure,
                "flow_rate_m3_h": flow_rate,
            }]
        }
        try:
            result = call_predict(payload)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Failure state**<br>{state_badge(result['failure_state'])}", unsafe_allow_html=True)
        col2.metric("Confidence", f"{result['confidence']:.1%}")
        col3.metric("Remaining Useful Life", f"{result['remaining_useful_life']:.1f} h")

        if result.get("warnings"):
            for w in result["warnings"]:
                st.warning(w)

        proba_df = pd.DataFrame(
            {"Class": list(result["class_probabilities"].keys()), "Probability": list(result["class_probabilities"].values())}
        )
        fig = px.bar(
            proba_df, x="Class", y="Probability", color="Class",
            color_discrete_map=CLASS_COLORS, title="Class probability distribution",
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

        log_history_entry({
            "asset_id": result["asset_id"], "timestamp": result["timestamp"],
            "failure_state": result["failure_state"], "confidence": result["confidence"],
            "remaining_useful_life": result["remaining_useful_life"],
        })


def page_batch_prediction():
    st.title("📦 Batch Prediction")
    st.caption("Upload a CSV with columns: Asset_ID (optional), Timestamp (optional), Machine_Model, "
               "Vibration_mm_s, Temperature_C, Pressure_psi, Flow_Rate_m3_h.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return

    st.dataframe(pd.read_csv(uploaded), width="stretch", height=200)
    uploaded.seek(0)

    if st.button("Run batch prediction", type="primary"):
        try:
            result_df = call_predict_csv(uploaded.read(), uploaded.name)
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")
            return

        st.success(f"Scored {len(result_df)} rows.")
        st.dataframe(result_df, width="stretch", height=350)

        col1, col2 = st.columns(2)
        with col1:
            counts = result_df["Failure_State_Predicted"].value_counts().reset_index()
            counts.columns = ["Failure_State", "Count"]
            fig = px.bar(counts, x="Failure_State", y="Count", color="Failure_State", color_discrete_map=CLASS_COLORS)
            fig.update_layout(showlegend=False, title="Predicted failure state distribution")
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig2 = px.histogram(result_df, x="RUL_Hours_Predicted", nbins=30, title="Predicted RUL distribution")
            st.plotly_chart(fig2, width="stretch")

        st.download_button(
            "Download predictions.csv", data=result_df.to_csv(index=False),
            file_name="predictions.csv", mime="text/csv",
        )
        log_history_entry({
            "asset_id": f"batch ({len(result_df)} rows)", "timestamp": pd.Timestamp.utcnow().isoformat(),
            "failure_state": "-", "confidence": float("nan"), "remaining_useful_life": float("nan"),
        })


def page_model_performance():
    st.title("📊 Model Performance")
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")
        return

    meta = artifacts.metadata
    col1, col2, col3 = st.columns(3)
    col1.metric("Classifier", artifacts.classifier_name, help="Selected by best macro Recall in Stage 8.")
    col2.metric("Test macro Recall", f"{meta.get('classifier_test_recall_macro', float('nan')):.3f}")
    col3.metric("Regressor MAE (h)", f"{meta.get('regressor_test_mae_hours', float('nan')):.1f}")

    st.caption(f"Artifacts saved: {meta.get('saved_at_utc', 'unknown')} · {meta.get('n_features')} features")

    st.divider()
    figures_dir = config.OUTPUTS_DIR / "figures"
    figure_captions = {
        "07_classification_confusion_matrices.png": "Confusion matrices — all 4 classifiers",
        "09_regression_predicted_vs_actual.png": "Predicted vs. actual RUL — all 4 regressors",
        "11_model_comparison_summary.png": "Model comparison summary",
    }
    for filename, caption in figure_captions.items():
        path = figures_dir / filename
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
        else:
            st.info(f"{caption}: figure not found at {path} (re-run the training notebook to regenerate outputs/figures).")


def page_feature_importance():
    st.title("🔬 Feature Importance")
    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")
        return

    tab1, tab2 = st.tabs(["Classifier", "Regressor"])
    for tab, model, label in [(tab1, artifacts.classifier, artifacts.classifier_name),
                               (tab2, artifacts.regressor, artifacts.regressor_name)]:
        with tab:
            if not hasattr(model, "feature_importances_"):
                st.info(f"{label} does not expose feature_importances_.")
                continue
            imp_df = pd.DataFrame({
                "Feature": artifacts.feature_list,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=True).tail(20)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title=f"Top 20 features — {label}", height=600)
            st.plotly_chart(fig, width="stretch")


def page_prediction_history():
    st.title("🕘 Prediction History")

    tab1, tab2 = st.tabs(["This session", "Persisted log (logs/predictions.log)"])

    with tab1:
        history = st.session_state.get("history", [])
        if not history:
            st.info("No predictions made yet this session. Try Single Prediction or Batch Prediction.")
        else:
            st.dataframe(pd.DataFrame(history), width="stretch")
            if st.button("Clear session history"):
                st.session_state["history"] = []
                st.rerun()

    with tab2:
        log_path = config.PREDICTION_LOG_FILE
        if not log_path.exists():
            st.info(f"No log file yet at {log_path}. Make a prediction via the API first.")
            return
        pattern = re.compile(
            r"(?P<ts>[\d\-: ,]+) asset=(?P<asset>\S+) failure_state=(?P<state>\S+) "
            r"confidence=(?P<confidence>[\d.]+) rul=(?P<rul>[\d.]+)"
        )
        rows = []
        for line in log_path.read_text().splitlines()[-200:]:
            m = pattern.search(line)
            if m:
                rows.append(m.groupdict())
        if rows:
            st.dataframe(pd.DataFrame(rows).iloc[::-1], width="stretch", height=400)
        else:
            st.info("Log file exists but no single-prediction entries found yet (batch predictions are logged as row counts only).")


def page_about():
    st.title("ℹ️ About")
    st.markdown(
        f"""
        **{config.APP_NAME}** v{config.APP_VERSION}

        End-to-end predictive maintenance system for industrial centrifugal
        pumps: EDA → feature engineering → model training/comparison
        (Decision Tree, Random Forest, XGBoost, LightGBM) → persistence →
        this FastAPI + Streamlit serving layer.

        **Stack:** pandas, scikit-learn, XGBoost, LightGBM, FastAPI, Streamlit, Docker.

        **Honest caveat:** the classifier's Normal/Warning/Critical predictions
        are reliable (non-overlapping sensor bands in training data). Its
        Bearing_Failure/Motor_Failure/Seal_Failure predictions are built on
        ~20 total training examples from a single pump and should not be
        trusted for fault-type diagnosis without collecting substantially
        more failure examples across multiple assets first.

        **API backend:** `{API_URL}` (override with the `PDM_API_URL` environment variable).
        """
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Home": page_home,
    "Project Overview": page_overview,
    "Single Prediction": page_single_prediction,
    "Batch Prediction": page_batch_prediction,
    "Model Performance": page_model_performance,
    "Feature Importance": page_feature_importance,
    "Prediction History": page_prediction_history,
    "About": page_about,
}

st.sidebar.title("🛠️ PdM Dashboard")
selection = st.sidebar.radio("Navigate", list(PAGES.keys()))
st.sidebar.divider()
st.sidebar.caption(f"API: {API_URL}")

PAGES[selection]()
