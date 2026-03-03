"""
Metrics Route
=============
GET /api/metrics
Computes and returns model performance metrics.
Requires both model and labeled dataset to be loaded.
"""

import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.schemas import MetricsResponse
from app.services.metrics_service import compute_metrics
from app.utils.session_store import session_store

logger = logging.getLogger("explainai.routes.metrics")
router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(request: Request):
    """
    Compute model performance metrics on the uploaded dataset.
    Requires a target column to have been specified during data upload.
    """
    session_id = getattr(request.state, "session_id", str(uuid.uuid4()))
    state = session_store.get(session_id)

    if state is None or state.model is None:
        return JSONResponse(
            status_code=400,
            content={"detail": "No model loaded. Upload a model first."},
        )

    if state.X is None:
        return JSONResponse(
            status_code=400,
            content={"detail": "No dataset loaded. Upload data first."},
        )

    if state.y is None:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "No target column available.",
                "suggestion": "Re-upload the dataset with target_column parameter specified.",
            },
        )

    # Use cached metrics if available
    if state.metrics:
        logger.info(f"[{session_id}] Returning cached metrics")
        m = state.metrics
    else:
        logger.info(f"[{session_id}] Computing metrics…")
        try:
            m = compute_metrics(
                model=state.model,
                X=state.X,
                y=state.y,
                model_info=state.model_info,
            )
            state.metrics = m
        except Exception as exc:
            logger.error(f"[{session_id}] Metrics computation failed: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": f"Metrics computation failed: {str(exc)}"},
            )

    return MetricsResponse(
        session_id=session_id,
        task_type=m["task_type"],
        n_samples=m["n_samples"],
        accuracy=m.get("accuracy"),
        precision_macro=m.get("precision_macro"),
        recall_macro=m.get("recall_macro"),
        f1_macro=m.get("f1_macro"),
        precision_weighted=m.get("precision_weighted"),
        recall_weighted=m.get("recall_weighted"),
        f1_weighted=m.get("f1_weighted"),
        roc_auc=m.get("roc_auc"),
        confusion_matrix=m.get("confusion_matrix"),
        mse=m.get("mse"),
        rmse=m.get("rmse"),
        mae=m.get("mae"),
        r2_score=m.get("r2_score"),
        notes=m.get("notes", {}),
        message="Metrics computed successfully.",
    )
