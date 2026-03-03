"""
Explanation Engine
==================
The intellectual core of ExplainAI. Implements both global and local
post-hoc explainability methods for black-box ML models.

==============================================================================
THEORETICAL FOUNDATIONS
==============================================================================

1. SHAP — SHapley Additive exPlanations
----------------------------------------
SHAP decomposes a model's prediction f(x) as:

    f(x) = φ₀ + Σᵢ φᵢ

where:
  φ₀ = E[f(x)]  (the "base value" — average model output over training data)
  φᵢ = SHAP value for feature i (signed contribution to prediction)

The φᵢ values are the unique solution satisfying the three Shapley axioms:
  • Efficiency:   Σᵢ φᵢ = f(x) - φ₀
  • Symmetry:     features contributing equally have equal φᵢ
  • Dummy:        unused features have φᵢ = 0
  • Linearity:    additive model decomposition

For tree ensembles, TreeExplainer computes exact Shapley values in
O(TLD²) time (T trees, L leaves, D max depth) — polynomial, not exponential.
Reference: Lundberg et al., 2020 (Nature Machine Intelligence)

GLOBAL interpretability: aggregate |φᵢ| across all instances
LOCAL interpretability: φᵢ for a single instance explains one prediction

2. LIME — Local Interpretable Model-agnostic Explanations
----------------------------------------------------------
LIME constructs a locally faithful linear surrogate model g around a
single instance x* by:

  Step 1: Sample N perturbed instances x' in the neighbourhood of x*
          (binary superpixels for images; feature-wise perturbations for tabular)
  Step 2: Weight samples by proximity: π(x*, x') = exp(-D(x*,x')²/σ²)
  Step 3: Train weighted ridge regression: g = argmin_g Σ π·(f(x')-g(x'))² + Ω(g)
  Step 4: Report top-K feature coefficients of g as the local explanation

Key insight: LIME is ALWAYS local — it only explains the prediction at x*.
  Different instances will yield different (potentially contradictory) explanations.
  This is NOT a bug — it reflects the non-linear nature of the black-box.

Computational Trade-offs:
  - SHAP TreeExplainer: O(TLD²) — fast for tree models; exact
  - SHAP KernelExplainer: O(N·2^M) — exponential; approximated via sampling
  - LIME: O(N·model_inference) — N~5000 perturbations; fast for any model
  - SHAP is preferred for tree/linear models; LIME for any model type

Reference: Ribeiro et al., 2016 (KDD Best Student Paper Award)

==============================================================================
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — required for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime import lime_tabular

logger = logging.getLogger("explainai.explanation_engine")


# ==============================================================================
# SHAP Explanations
# ==============================================================================


def compute_shap_explanations(
    model: Any,
    X: pd.DataFrame,
    model_info: Dict,
    background_sample_size: int = 100,
) -> Dict:
    """
    Compute SHAP values for all instances in X.

    The explainer type is chosen automatically based on the model family:
      - TreeExplainer  → RandomForest, XGBoost, GradientBoosting, DecisionTree
      - LinearExplainer → LogisticRegression, Ridge, Lasso
      - KernelExplainer → any black-box (slow; uses background distribution)

    Parameters
    ----------
    model : sklearn estimator
        Fitted model object.
    X : pd.DataFrame
        Feature matrix (all rows used for global SHAP; subset for local).
    model_info : dict
        Output of model_handler.load_model()
    background_sample_size : int
        Number of background samples for KernelExplainer.
        Larger → more accurate but slower (O(N·2^M) internally).

    Returns
    -------
    dict containing:
      - shap_values: np.ndarray, shape [n_samples, n_features]
      - base_value: float (φ₀, expected model output)
      - explainer_type: str
      - feature_names: list[str]
    """
    logger.info(
        f"Computing SHAP values | backend={model_info['explainer_backend']} | "
        f"n_samples={len(X)} | n_features={X.shape[1]}"
    )

    feature_names = list(X.columns)
    X_values = X.values  # Convert to numpy for SHAP compatibility

    backend = model_info["explainer_backend"]

    try:
    # ------------------------------------------------------------
    # Use modern SHAP API (auto-detects model type)
    # Works for tree, linear, kernel, binary, multiclass, regression
    # ------------------------------------------------------------
        explainer = shap.Explainer(model, X_values)
        shap_result = explainer(X_values)

        shap_values = shap_result.values
        base_values = shap_result.base_values

        # ------------------------------------------------------------
        # Handle regression, binary, and multiclass safely
        # ------------------------------------------------------------
        if isinstance(base_values, np.ndarray):

            # Multi-class: shape (n_samples, n_classes)
            if base_values.ndim == 2:
                # Choose positive class (1) if binary
                if base_values.shape[1] == 2:
                    shap_values = shap_values[:, :, 1]
                    base_value = float(base_values[0, 1])
                else:
                    # For multi-class, default to first class
                    shap_values = shap_values[:, :, 0]
                    base_value = float(base_values[0, 0])

            # Regression or single-output
            else:
                base_value = float(base_values[0])

        else:
            base_value = float(base_values)

    except Exception as exc:
        logger.error(f"SHAP computation failed: {exc}", exc_info=True)
        raise RuntimeError(f"SHAP computation failed: {exc}") from exc
    logger.info(
        f"SHAP values computed: shape={shap_values.shape} | base_value={base_value:.4f}"
    )

    return {
        "shap_values": shap_values,
        "base_value": float(base_value),
        "explainer_type": f"shap.{backend.capitalize()}Explainer",
        "feature_names": feature_names,
    }


def generate_shap_summary_plot(shap_values: np.ndarray, X: pd.DataFrame) -> str:
    """
    Generate the SHAP Summary Beeswarm Plot as a base64 PNG.

    The summary plot provides GLOBAL interpretability by showing:
      - Feature ranking by mean |SHAP value| (vertical axis)
      - Distribution of SHAP values per feature (horizontal spread)
      - Feature value magnitude (colour: blue=low, red=high)

    This plot answers: "Which features drive model predictions on average?"

    Returns
    -------
    str: base64-encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(X.columns) * 0.4)))
    plt.sca(ax)

    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_size=None,
        color_bar=True,
    )

    ax.set_title(
        "SHAP Summary Plot — Global Feature Importance\n"
        "(dot position = SHAP value; colour = feature value magnitude)",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout()

    return _fig_to_base64(fig)


def generate_shap_bar_plot(shap_values: np.ndarray, feature_names: List[str]) -> str:
    """
    Generate mean |SHAP| bar chart — the canonical global importance ranking.

    Mean |SHAP| = (1/N) Σᵢ |φᵢ(xⱼ)|  averaged over all instances j.
    This is the SHAP-preferred measure of global feature importance,
    superior to MDI (which is biased toward high-cardinality features).
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    top_n = min(20, len(feature_names))
    top_features = [feature_names[i] for i in sorted_idx[:top_n]]
    top_values = mean_abs_shap[sorted_idx[:top_n]]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, top_n))
    ax.barh(range(top_n), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (average impact on model output magnitude)", fontsize=10)
    ax.set_title(
        "SHAP Global Feature Importance (Mean |SHAP|)\n"
        "Theoretically grounded: satisfies Shapley axioms (efficiency, symmetry, dummy, linearity)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    return _fig_to_base64(fig)


def generate_shap_force_plot_data(
    shap_values: np.ndarray,
    base_value: float,
    X: pd.DataFrame,
    instance_index: int = 0,
) -> Dict:
    """
    Compute data for a SHAP Force Plot for a single instance.

    The force plot visualizes LOCAL interpretability:
      f(x) = φ₀ + Σᵢ φᵢ
    Red arrows push the prediction HIGHER than baseline (φ₀)
    Blue arrows push the prediction LOWER than baseline

    Returns serializable data (not an image) so the React frontend
    can render an interactive Plotly force plot.
    """
    instance_shap = shap_values[instance_index]
    instance_features = X.iloc[instance_index]
    feature_names = list(X.columns)

    # Sort by absolute SHAP value (strongest contributors first)
    sorted_idx = np.argsort(np.abs(instance_shap))[::-1]
    top_n = min(15, len(feature_names))

    contributions = []
    for i in sorted_idx[:top_n]:
        contributions.append(
            {
                "feature": feature_names[i],
                "value": float(instance_features.iloc[i]),
                "shap_value": float(instance_shap[i]),
                "direction": "positive" if instance_shap[i] > 0 else "negative",
            }
        )

    prediction = float(base_value + instance_shap.sum())

    return {
        "contributions": contributions,
        "base_value": float(base_value),
        "prediction": prediction,
        "instance_index": instance_index,
        "interpretation": (
            f"The model's base prediction is {base_value:.4f}. "
            f"Feature contributions shift this to {prediction:.4f}. "
            f"This is a LOCAL explanation — valid only for instance #{instance_index}."
        ),
    }


# ==============================================================================
# LIME Explanations
# ==============================================================================


def compute_lime_explanation(
    model: Any,
    X: pd.DataFrame,
    model_info: Dict,
    instance_index: int = 0,
    num_features: int = 10,
    num_samples: int = 5000,
) -> Dict:
    """
    Generate a LIME explanation for a single instance.

    LIME Algorithm (tabular data):
    --------------------------------
    1. Select instance x* to explain
    2. Create N=num_samples perturbed versions by randomly masking features
       (replacing masked features with values drawn from the training distribution)
    3. Get black-box predictions f(x'₁), ..., f(x'ₙ)
    4. Weight samples by: π(x*, x'ᵢ) = exp(-‖x* - x'ᵢ‖² / σ²)
       where σ = 0.75 * √(number of features) [default LIME kernel width]
    5. Fit weighted ridge regression on the perturbed data
    6. Return top-K coefficients as the local explanation

    WHY LIME IS LOCAL (and why that matters):
    The surrogate g is trained in the vicinity of x* only. Moving to a
    different instance x** would produce a completely different surrogate g**.
    This reflects the non-linearity of the underlying black-box — there is no
    single global linear approximation that is faithful everywhere.

    Computational note:
    num_samples=5000 is the standard LIME default. Increasing to 10000+
    improves explanation stability at the cost of higher latency.

    Parameters
    ----------
    num_features : int
        Number of features to include in the local explanation.
    num_samples : int
        Number of perturbation samples for the surrogate model.

    Returns
    -------
    dict: serialized LIME explanation + base64 plot image
    """
    logger.info(
        f"Computing LIME explanation | instance={instance_index} | "
        f"num_samples={num_samples} | num_features={num_features}"
    )

    feature_names = list(X.columns)
    X_values = X.values

    # -------------------------------------------------------------------------
    # Build LIME Explainer
    # training_data is used to compute per-feature statistics (mean, std)
    # which define the perturbation distribution
    # -------------------------------------------------------------------------
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_values,
        feature_names=feature_names,
        mode="classification" if model_info["is_classifier"] else "regression",
        discretize_continuous=True,  # Bin continuous features for interpretability
        random_state=42,
    )

    # Choose prediction function based on task type
    if model_info["is_classifier"] and model_info["supports_probability"]:
        predict_fn = model.predict_proba
    else:
        predict_fn = lambda x: model.predict(x).reshape(-1, 1)

    instance = X_values[instance_index]

    try:
        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=1,
        )
    except Exception as exc:
        logger.error(f"LIME failed: {exc}", exc_info=True)
        raise RuntimeError(f"LIME explanation failed: {exc}") from exc

    # -------------------------------------------------------------------------
    # Serialize LIME output
    # -------------------------------------------------------------------------
    label = explanation.available_labels()[0]
    lime_features = explanation.as_list(label=label)

    lime_data = [
        {
            "feature_condition": cond,
            "weight": float(weight),
            "direction": "positive" if weight > 0 else "negative",
        }
        for cond, weight in lime_features
    ]

    # Generate matplotlib plot
    plot_image = _generate_lime_plot(lime_data, instance_index)

    # Local prediction from LIME surrogate
    local_pred = explanation.local_pred
    intercept = explanation.intercept[label]

    return {
        "explanation": lime_data,
        "label": int(label),
        "local_prediction": float(local_pred[0]) if local_pred is not None else None,
        "intercept": float(intercept),
        "instance_index": instance_index,
        "num_samples_used": num_samples,
        "plot_image": plot_image,
        "interpretation": (
            f"LIME trained a local linear surrogate using {num_samples} perturbed instances "
            f"near instance #{instance_index}. Positive weights push toward class {label}; "
            f"negative weights push away. This explanation is LOCAL — it only describes "
            f"this specific prediction, not the model's global behaviour."
        ),
    }


def _generate_lime_plot(lime_data: List[Dict], instance_index: int) -> str:
    """Render LIME feature weights as a horizontal bar chart."""
    features = [d["feature_condition"] for d in lime_data]
    weights = [d["weight"] for d in lime_data]

    # Truncate long feature condition strings for readability
    features = [f[:45] + "…" if len(f) > 45 else f for f in features]

    fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.45)))
    colors = ["#e74c3c" if w > 0 else "#3498db" for w in weights]
    y_pos = range(len(features))

    ax.barh(y_pos, weights, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8.5)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("LIME Feature Weight (local linear surrogate coefficient)", fontsize=10)
    ax.set_title(
        f"LIME Local Explanation — Instance #{instance_index}\n"
        "Red=pushes toward positive class | Blue=pushes toward negative class",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    return _fig_to_base64(fig)


# ==============================================================================
# Utility
# ==============================================================================


def _fig_to_base64(fig: plt.Figure) -> str:
    """
    Serialize a matplotlib Figure to a base64-encoded PNG string.
    The string is directly embeddable in HTML: <img src="data:image/png;base64,..."/>
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    buf.close()
    return encoded
