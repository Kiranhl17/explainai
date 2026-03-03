"""
Data Upload Route
=================
POST /api/upload-data
Accepts a CSV file, validates it against the loaded model, and stores in session.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.schemas.schemas import DataUploadResponse
from app.services.data_validator import load_and_validate_csv
from app.utils.session_store import session_store

logger = logging.getLogger("explainai.routes.data")
router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_CSV_SIZE_MB = 50


@router.post("/upload-data", response_model=DataUploadResponse)
async def upload_data(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(default=""),
):
    """
    Upload a CSV dataset.

    The target_column parameter (optional) specifies which column contains
    the ground-truth labels/targets. If provided, metrics computation
    becomes available.
    """
    session_id = getattr(request.state, "session_id", str(uuid.uuid4()))

    if not file.filename.lower().endswith(".csv"):
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Only CSV files are accepted.",
                "suggestion": "Save your dataset as a .csv file with headers.",
            },
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_CSV_SIZE_MB:
        return JSONResponse(
            status_code=413,
            content={"detail": f"CSV file too large ({size_mb:.1f} MB > {MAX_CSV_SIZE_MB} MB)."},
        )

    save_path = UPLOAD_DIR / f"{session_id}_data.csv"
    with open(save_path, "wb") as f:
        f.write(content)

    # Get current model info for compatibility check
    state = session_store.get_or_create(session_id)
    model_info = state.model_info if state.model_info else None

    try:
        X, y, data_info = load_and_validate_csv(
            str(save_path),
            model_info=model_info,
            target_column=target_column.strip() if target_column else None,
        )
    except ValueError as exc:
        save_path.unlink(missing_ok=True)
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # Persist in session
    state.X = X
    state.y = y
    state.feature_names = list(X.columns)
    # Reset explanation cache when new data is uploaded
    state.shap_values = None
    state.lime_explanation = None
    state.metrics = None

    logger.info(
        f"[{session_id}] Data loaded: {data_info['n_rows']} rows × "
        f"{data_info['n_features']} features | target={data_info['has_target']}"
    )

    return DataUploadResponse(
        session_id=session_id,
        n_rows=data_info["n_rows"],
        n_features=data_info["n_features"],
        feature_names=data_info["feature_names"],
        has_target=data_info["has_target"],
        missing_values_imputed=data_info.get("missing_values_imputed", {}),
        dropped_columns=data_info.get("dropped_columns", []),
        compatibility_issues=data_info.get("compatibility_issues", []),
        message=f"Dataset loaded: {data_info['n_rows']} rows × {data_info['n_features']} features.",
    )
