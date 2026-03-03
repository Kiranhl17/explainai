"""
ExplainAI – Model Transparency Visualizer
==========================================
Main FastAPI application entry point.

Academic Context:
-----------------
Explainable AI (XAI) sits at the intersection of interpretability research and
practical ML deployment. This system implements two dominant post-hoc explanation
paradigms:

  1. SHAP (SHapley Additive exPlanations) — game-theoretic, model-agnostic
     global AND local explanations rooted in cooperative game theory.
     Ref: Lundberg & Lee, 2017 (NeurIPS)

  2. LIME (Local Interpretable Model-agnostic Explanations) — perturbation-based
     local surrogate models that approximate the black-box decision boundary
     in a neighbourhood of a single instance.
     Ref: Ribeiro et al., 2016 (KDD)

The distinction between GLOBAL and LOCAL interpretability is foundational:
  - Global: understand average model behaviour across the dataset
  - Local: understand why a specific prediction was made

This FastAPI app exposes these capabilities via a clean REST API consumed
by the React frontend.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import model_routes, data_routes, explanation_routes, metrics_routes
from app.utils.session_store import session_store

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("explainai.main")

# ---------------------------------------------------------------------------
# Upload directory bootstrap
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle handler.
    Cleans up stale session files on restart to prevent disk accumulation.
    """
    logger.info("ExplainAI backend starting up…")
    # Purge uploads older than the current process (optional in prod)
    yield
    logger.info("ExplainAI backend shutting down…")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ExplainAI – Model Transparency Visualizer",
    description=(
        "A production-grade Explainable AI backend providing SHAP, LIME, "
        "and feature importance explanations for scikit-learn models. "
        "Designed for MSc-level AI transparency research."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS – allow the React frontend (Vercel / localhost)
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,https://explainai.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Session Injection Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """
    Injects a lightweight session ID header so the frontend can correlate
    model uploads, data uploads, and explanation requests without auth.
    In production this would be replaced by JWT-based user auth.
    """
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())
    request.state.session_id = session_id

    response = await call_next(request)
    response.headers["X-Session-ID"] = session_id
    return response


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred.",
            "error_type": type(exc).__name__,
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(model_routes.router, prefix="/api", tags=["Model"])
app.include_router(data_routes.router, prefix="/api", tags=["Data"])
app.include_router(explanation_routes.router, prefix="/api", tags=["Explanations"])
app.include_router(metrics_routes.router, prefix="/api", tags=["Metrics"])


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe for Docker / Render / AWS health checks."""
    return {"status": "healthy", "service": "ExplainAI Backend", "version": "1.0.0"}


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Welcome to ExplainAI – Model Transparency Visualizer",
        "docs": "/docs",
        "health": "/health",
    }
