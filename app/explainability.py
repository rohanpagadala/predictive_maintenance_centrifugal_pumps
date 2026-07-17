from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

TOP_N_FEATURES = 8
BACKGROUND_SAMPLE_FILENAME = "shap_background_sample.joblib"
_XGBOOST_BASE_SCORE_PATCHED = False


def _patch_shap_xgboost_base_score_parsing() -> None:
    global _XGBOOST_BASE_SCORE_PATCHED
    if _XGBOOST_BASE_SCORE_PATCHED:
        return

    import shap.explainers._tree as shap_tree

    _orig_decode = shap_tree.decode_ubjson_buffer

    def _patched_decode(fd):
        jmodel = _orig_decode(fd)
        try:
            lmp = jmodel["learner"]["learner_model_param"]
            base_score = lmp.get("base_score")
            if isinstance(base_score, str) and base_score.strip().startswith("["):
                lmp["base_score"] = str(float(base_score.strip("[]")))
        except (KeyError, TypeError, ValueError):
            pass
        return jmodel

    shap_tree.decode_ubjson_buffer = _patched_decode
    _XGBOOST_BASE_SCORE_PATCHED = True
    logger.info("Patched SHAP's XGBoost base_score parsing for XGBoost 3.x compatibility.")


@dataclass
class Explainers:
    classifier_explainer: Any
    regressor_explainer: Any


def build_explainers(classifier: Any, regressor: Any) -> Explainers:
    _patch_shap_xgboost_base_score_parsing()
    clf_explainer = shap.TreeExplainer(classifier)
    reg_model = regressor.get_booster() if hasattr(regressor, "get_booster") else regressor
    reg_explainer = shap.TreeExplainer(reg_model)
    return Explainers(classifier_explainer=clf_explainer, regressor_explainer=reg_explainer)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _top_features(
    feature_names: list[str], values: np.ndarray, row_values: np.ndarray, n: int,
    positive_label: str, negative_label: str,
) -> list[dict]:
    order = np.argsort(-np.abs(values))[:n]
    return [
        {
            "feature": feature_names[i],
            "shap_value": round(float(values[i]), 4),
            "feature_value": round(float(row_values[i]), 4) if isinstance(row_values[i], (int, float, np.floating)) and not np.isnan(row_values[i]) else None,
            "direction": positive_label if values[i] > 0 else negative_label,
        }
        for i in order
    ]


def explain_prediction(
    X_row: pd.DataFrame, explainers: Explainers, feature_list: list[str], class_names: list[str],
    predicted_class: int,
) -> dict:
    X = X_row[feature_list]
    row_values = X.iloc[0].to_numpy()

    clf_exp_full = explainers.classifier_explainer(X)
    clf_values = clf_exp_full.values[0, :, predicted_class]
    clf_single = clf_exp_full[0, :, predicted_class]

    fig_clf = plt.figure()
    shap.plots.waterfall(clf_single, max_display=TOP_N_FEATURES, show=False)
    plt.title(f"Classifier — contribution to '{class_names[predicted_class]}'")
    classifier_waterfall_b64 = _fig_to_base64(fig_clf)

    reg_exp_full = explainers.regressor_explainer(X)
    reg_values = reg_exp_full.values[0]

    fig_reg = plt.figure()
    shap.plots.waterfall(reg_exp_full[0], max_display=TOP_N_FEATURES, show=False)
    plt.title("Regressor — contribution to predicted RUL (hours)")
    regressor_waterfall_b64 = _fig_to_base64(fig_reg)

    classifier_labels = (
        ("pushes_toward_normal", "pushes_toward_failure")
        if predicted_class == 0
        else (f"pushes_toward_{class_names[predicted_class]}", "pushes_away_from_prediction")
    )

    return {
        "predicted_class": predicted_class,
        "predicted_class_label": class_names[predicted_class],
        "classifier_top_features": _top_features(
            feature_list, clf_values, row_values, TOP_N_FEATURES, *classifier_labels
        ),
        "regressor_top_features": _top_features(
            feature_list, reg_values, row_values, TOP_N_FEATURES,
            "increases_predicted_rul", "decreases_predicted_rul",
        ),
        "classifier_base_value": round(float(clf_exp_full.base_values[0, predicted_class]), 4),
        "regressor_base_value": round(float(reg_exp_full.base_values[0]), 4),
        "classifier_waterfall_png_base64": classifier_waterfall_b64,
        "regressor_waterfall_png_base64": regressor_waterfall_b64,
    }


def generate_summary_plots(explainers: Explainers, background_df: pd.DataFrame, feature_list: list[str]) -> dict:
    X = background_df[feature_list]

    reg_exp = explainers.regressor_explainer(X)
    plt.figure()
    shap.summary_plot(reg_exp.values, X, show=False, plot_size=(9, 6))
    plt.title("Regressor — global feature impact (RUL)")
    regressor_summary_b64 = _fig_to_base64(plt.gcf())

    clf_exp = explainers.classifier_explainer(X)
    plt.figure()
    shap.summary_plot(
        [clf_exp.values[:, :, c] for c in range(clf_exp.values.shape[2])],
        X, show=False, plot_size=(9, 6), class_names=None,
    )
    plt.title("Classifier — global feature impact (mean |SHAP| across classes)")
    classifier_summary_b64 = _fig_to_base64(plt.gcf())

    return {
        "classifier_summary_png_base64": classifier_summary_b64,
        "regressor_summary_png_base64": regressor_summary_b64,
        "n_background_rows": len(background_df),
    }
