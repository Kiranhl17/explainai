"""
Model Handler Service
=====================
Responsible for secure deserialization, validation, and introspection
of scikit-learn compatible model files (.pkl via joblib).

Security Note:
  Pickle/joblib deserialization is inherently unsafe with untrusted files.
  In production you would:
    1. Run deserialization in an isolated subprocess / sandbox
    2. Validate file signatures / MIME types before loading
    3. Implement an allowlist of safe model classes

  For this academic system, we implement class-based allowlisting and
  sandboxed error handling as a baseline protection layer.

Supported Model Families (v1.0):
  - sklearn.ensemble: RandomForestClassifier, GradientBoostingClassifier,
                      ExtraTreesClassifier, RandomForestRegressor
  - xgboost: XGBClassifier, XGBRegressor  (if installed)
  - sklearn.tree: DecisionTreeClassifier, DecisionTreeRegressor
  - sklearn.linear_model: LogisticRegression (SHAP via LinearExplainer)

Future Extension:
  - Neural network models (SHAP DeepExplainer / GradientExplainer)
  - Pipeline objects with preprocessing steps
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np

logger = logging.getLogger("explainai.model_handler")

# ---------------------------------------------------------------------------
# Allowlisted model types — expand as new explainer backends are added
# ---------------------------------------------------------------------------
SUPPORTED_MODEL_TYPES = {
    # Tree-based ensembles (SHAP TreeExplainer — O(TLD) exact Shapley values)
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    # Single trees
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    # Linear models (SHAP LinearExplainer — closed-form Shapley values)
    "LogisticRegression",
    "LinearRegression",
    "Ridge",
    "Lasso",
    # XGBoost (SHAP TreeExplainer — native fast implementation)
    "XGBClassifier",
    "XGBRegressor",
    # LightGBM
    "LGBMClassifier",
    "LGBMRegressor",
}

TREE_BASED_MODELS = {
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
}

LINEAR_MODELS = {
    "LogisticRegression", "LinearRegression", "Ridge", "Lasso",
}


def load_model(file_path: Path) -> Tuple[Any, Dict]:
    """
    Deserialize a joblib-serialized scikit-learn model from disk.

    Parameters
    ----------
    file_path : Path
        Absolute path to the .pkl file on the server.

    Returns
    -------
    model : sklearn estimator
        The deserialized model object.
    model_info : dict
        Metadata dictionary including:
          - model_type: class name string
          - is_classifier: bool
          - n_features: int (if available)
          - explainer_backend: "tree" | "linear" | "kernel"
          - supports_probability: bool

    Raises
    ------
    ValueError
        If the model class is not on the allowlist.
    RuntimeError
        If deserialization fails.
    """
    logger.info(f"Loading model from: {file_path}")

    try:
        model = joblib.load(file_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to deserialize model. Ensure the file is a valid "
            f"joblib/pickle artefact. Details: {exc}"
        ) from exc

    model_class = type(model).__name__

    if model_class not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Model class '{model_class}' is not supported. "
            f"Supported types: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    # --- Introspect model properties ---
    is_classifier = hasattr(model, "predict_proba") or hasattr(model, "classes_")

    # n_features_in_ is standardized in sklearn >= 1.0
    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        # Fallback for older sklearn / XGBoost models
        n_features = getattr(model, "n_features_", None)

    # Determine optimal SHAP explainer backend
    if model_class in TREE_BASED_MODELS:
        explainer_backend = "tree"
    elif model_class in LINEAR_MODELS:
        explainer_backend = "linear"
    else:
        explainer_backend = "kernel"  # Slow but universal

    # Number of output classes (classifiers only)
    n_classes = len(getattr(model, "classes_", [])) if is_classifier else None

    model_info = {
        "model_type": model_class,
        "is_classifier": is_classifier,
        "n_features": n_features,
        "n_classes": n_classes,
        "explainer_backend": explainer_backend,
        "supports_probability": hasattr(model, "predict_proba"),
        "sklearn_params": _safe_get_params(model),
    }

    logger.info(
        f"Model loaded: {model_class} | "
        f"classifier={is_classifier} | "
        f"n_features={n_features} | "
        f"backend={explainer_backend}"
    )

    return model, model_info


def _safe_get_params(model) -> Dict:
    """
    Safely extract model hyperparameters for display purposes.
    Returns an empty dict if get_params() fails (e.g. XGBoost edge cases).
    """
    try:
        params = model.get_params()
        # Filter out nested estimators to keep JSON-serializable
        return {
            k: v for k, v in params.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
    except Exception:
        return {}


def get_feature_importances(model, feature_names: list) -> Dict:
    """
    Extract native feature importances from tree-based models.

    Tree models expose feature_importances_ (Mean Decrease in Impurity / MDI).
    Note: MDI is biased towards high-cardinality features; SHAP provides a
    more theoretically sound alternative (see explanation_engine.py).

    Returns
    -------
    dict with keys:
      - importances: list of (feature_name, importance_value) sorted desc
      - method: str description of the importance calculation method
    """
    importances_attr = getattr(model, "feature_importances_", None)

    if importances_attr is None:
        # Linear models: use absolute coefficient magnitudes as proxy
        coef = getattr(model, "coef_", None)
        if coef is not None:
            importances_attr = np.abs(coef).flatten()
            method = "absolute_coefficients"
        else:
            return {"importances": [], "method": "unavailable"}
    else:
        method = "mean_decrease_impurity"

    if len(feature_names) != len(importances_attr):
        feature_names = [f"feature_{i}" for i in range(len(importances_attr))]

    paired = sorted(
        zip(feature_names, importances_attr.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "importances": [{"feature": f, "importance": round(v, 6)} for f, v in paired],
        "method": method,
        "note": (
            "MDI importances can be biased toward high-cardinality features. "
            "SHAP-based importances (mean |SHAP value|) are preferred for rigorous analysis."
        ),
    }
