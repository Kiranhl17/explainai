"""
Explanation Route
=================
POST /api/generate-explanations
Orchestrates SHAP + LIME + Feature Importance computation.

This is the computationally intensive core of ExplainAI.
Computation times (approximate, on CPU):
  - SHAP TreeExplainer on 1000 instances: ~2-5 seconds
  - LIME with 5000 samples: ~3-8 seconds
  - Total: ~5-15 seconds depending on model and dataset size
"""

import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.schemas import ExplanationRequest, ExplanationResponse, FeatureImportance
from app.services.explanation_engine import (
    compute_lime_explanation,
    compute_shap_explanations,
    generate_shap_bar_plot,
    generate_shap_force_plot_data,
    generate_shap_summary_plot,
)
from app.services.model_handler import get_feature_importances
from app.utils.session_store import session_store

logger = logging.getLogger("explainai.routes.explanation")
router = APIRouter()


@router.post("/generate-explanations", response_model=ExplanationResponse)
async def generate_explanations(request: Request, params: ExplanationRequest):
    """
    Generate comprehensive XAI explanations for the uploaded model + dataset.

    Pipeline:
      1. Validate session state (model and data must both be loaded)
      2. Compute SHAP values (global + local)
      3. Generate SHAP plots (summary beeswarm, mean-|SHAP| bar)
      4. Generate SHAP force plot data for selected instance
      5. Compute LIME explanation for selected instance
      6. Extract native feature importances

    The results are cached in the session to avoid recomputation
    if the same instance is requested again.
    """
    session_id = getattr(request.state, "session_id", str(uuid.uuid4()))
    state = session_store.get(session_id)

    # --- Pre-flight checks ---
    if state is None or state.model is None:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "No model found in session. Please upload a model first.",
                "suggestion": "POST /api/upload-model before calling this endpoint.",
            },
        )

    if state.X is None:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "No dataset found in session. Please upload data first.",
                "suggestion": "POST /api/upload-data before calling this endpoint.",
            },
        )

    instance_idx = params.instance_index
    if instance_idx >= len(state.X):
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"instance_index {instance_idx} out of range (dataset has {len(state.X)} rows).",
            },
        )

    state.instance_index = instance_idx
    model = state.model
    model_info = state.model_info
    X = state.X

    logger.info(
        f"[{session_id}] Generating explanations | "
        f"model={model_info['model_type']} | "
        f"n_samples={len(X)} | instance={instance_idx}"
    )

    # ===========================================================================
    # 1. SHAP — Global + Local
    # ===========================================================================
    X_shap = X.head(50)  # limit to 50 rows for speed on free tier
    try:
        if state.shap_values is None or state.shap_values.shape[0] != len(X_shap):
            logger.info(f"[{session_id}] Computing SHAP values…")
            shap_result = compute_shap_explanations(
                model=model,
                X=X_shap,
                model_info=model_info,
                background_sample_size=min(params.background_sample_size, 50),
            )
            state.shap_values = shap_result["shap_values"]
            state.shap_base_value = shap_result["base_value"]
            shap_explainer_type = shap_result["explainer_type"]
        else:
            logger.info(f"[{session_id}] Using cached SHAP values")
            shap_explainer_type = f"shap.{model_info['explainer_backend'].capitalize()}Explainer"

        shap_summary_plot = generate_shap_summary_plot(state.shap_values, X_shap)
        shap_bar_plot = generate_shap_bar_plot(state.shap_values, state.feature_names)
        shap_force_data = generate_shap_force_plot_data(
            shap_values=state.shap_values,
            base_value=state.shap_base_value,
            X=X_shap,
            instance_index=instance_idx,
        )

    except Exception as exc:
        logger.error(f"[{session_id}] SHAP failed: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"SHAP computation failed: {str(exc)}"},
        )

    # ===========================================================================
    # 2. LIME — Local explanation for selected instance
    # ===========================================================================
    try:
        logger.info(f"[{session_id}] Computing LIME explanation for instance {instance_idx}…")
        lime_result = compute_lime_explanation(
            model=model,
            X=X.head(50),
            model_info=model_info,
            instance_index=instance_idx,
            num_features=10,
            num_samples=1000,
        )
        state.lime_explanation = lime_result
    except Exception as exc:
        logger.error(f"[{session_id}] LIME failed: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"LIME computation failed: {str(exc)}"},
        )

    # ===========================================================================
    # 3. Native Feature Importances
    # ===========================================================================
    fi_result = get_feature_importances(model, state.feature_names)
    feature_importances = [
        FeatureImportance(**item) for item in fi_result.get("importances", [])
    ]

    logger.info(f"[{session_id}] Explanation pipeline complete.")

    return ExplanationResponse(
        session_id=session_id,
        shap_summary_plot=shap_summary_plot,
        shap_bar_plot=shap_bar_plot,
        shap_explainer_type=shap_explainer_type,
        shap_force_data=shap_force_data,
        lime_explanation={
            "explanation": lime_result["explanation"],
            "label": lime_result["label"],
            "local_prediction": lime_result["local_prediction"],
            "intercept": lime_result["intercept"],
            "instance_index": lime_result["instance_index"],
            "interpretation": lime_result["interpretation"],
        },
        lime_plot=lime_result["plot_image"],
        feature_importances=feature_importances,
        feature_importance_method=fi_result.get("method", "unknown"),
        instance_index=instance_idx,
        message="Explanations generated successfully.",
    )
