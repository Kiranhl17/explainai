"""
Model Upload Route
==================
POST /api/upload-model
Accepts a .pkl file, validates it, and stores the model in the session.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.schemas.schemas import ModelUploadResponse
from app.services.model_handler import load_model
from app.utils.session_store import session_store

logger = logging.getLogger("explainai.routes.model")
router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_MODEL_SIZE_MB = 200
ALLOWED_EXTENSIONS = {".pkl", ".joblib"}


@router.post("/upload-model", response_model=ModelUploadResponse)
async def upload_model(request: Request, file: UploadFile = File(...)):
    """
    Upload a serialized scikit-learn model (.pkl / .joblib).

    Security checks:
      - Extension allowlist
      - File size limit
      - Model class allowlist (inside model_handler)
    """
    session_id = getattr(request.state, "session_id", str(uuid.uuid4()))

    # --- Extension check ---
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"Invalid file extension '{file_ext}'. Only .pkl and .joblib are accepted.",
                "suggestion": "Re-serialize your model using joblib.dump(model, 'model.pkl')",
            },
        )

    # --- Read and size-check ---
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_MODEL_SIZE_MB:
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"Model file too large ({size_mb:.1f} MB > {MAX_MODEL_SIZE_MB} MB).",
                "suggestion": "Consider compressing or using a smaller model.",
            },
        )

    # --- Save to disk ---
    save_path = UPLOAD_DIR / f"{session_id}_model{file_ext}"
    with open(save_path, "wb") as f:
        f.write(content)

    # --- Load and validate ---
    try:
        model, model_info = load_model(save_path)
    except (ValueError, RuntimeError) as exc:
        save_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(exc),
                "suggestion": "Ensure you are uploading a supported sklearn model type.",
            },
        )

    # --- Persist in session ---
    state = session_store.get_or_create(session_id)
    state.model = model
    state.model_info = model_info
    # Reset any prior explanation cache
    state.shap_values = None
    state.lime_explanation = None
    state.metrics = None

    logger.info(f"[{session_id}] Model loaded: {model_info['model_type']}")

    return ModelUploadResponse(
        session_id=session_id,
        model_type=model_info["model_type"],
        is_classifier=model_info["is_classifier"],
        n_features=model_info.get("n_features"),
        n_classes=model_info.get("n_classes"),
        explainer_backend=model_info["explainer_backend"],
        supports_probability=model_info["supports_probability"],
        message=f"Model '{model_info['model_type']}' loaded successfully.",
        model_params=model_info.get("sklearn_params", {}),
    )
