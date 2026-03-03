"""
API Schemas
===========
Pydantic v2 models for request validation and response serialization.
Using strict typing ensures API contract reliability.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Model Upload
# ---------------------------------------------------------------------------


class ModelUploadResponse(BaseModel):
    session_id: str
    model_type: str
    is_classifier: bool
    n_features: Optional[int]
    n_classes: Optional[int]
    explainer_backend: str
    supports_probability: bool
    message: str
    model_params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data Upload
# ---------------------------------------------------------------------------


class DataUploadRequest(BaseModel):
    target_column: Optional[str] = None


class DataUploadResponse(BaseModel):
    session_id: str
    n_rows: int
    n_features: int
    feature_names: List[str]
    has_target: bool
    missing_values_imputed: Dict[str, int] = Field(default_factory=dict)
    dropped_columns: List[str] = Field(default_factory=list)
    compatibility_issues: List[str] = Field(default_factory=list)
    message: str


# ---------------------------------------------------------------------------
# Explanation Request
# ---------------------------------------------------------------------------


class ExplanationRequest(BaseModel):
    instance_index: int = Field(default=0, ge=0, description="Index of instance for local explanation")
    num_lime_features: int = Field(default=10, ge=1, le=50)
    num_lime_samples: int = Field(default=5000, ge=100, le=20000)
    background_sample_size: int = Field(default=100, ge=10, le=500)

    @field_validator("instance_index")
    @classmethod
    def validate_instance_index(cls, v):
        if v < 0:
            raise ValueError("instance_index must be non-negative")
        return v


# ---------------------------------------------------------------------------
# Explanation Response
# ---------------------------------------------------------------------------


class SHAPContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str  # "positive" | "negative"


class LIMEContribution(BaseModel):
    feature_condition: str
    weight: float
    direction: str


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ExplanationResponse(BaseModel):
    session_id: str
    # Global SHAP
    shap_summary_plot: str = Field(description="Base64-encoded PNG")
    shap_bar_plot: str = Field(description="Base64-encoded PNG")
    shap_explainer_type: str
    # Local SHAP
    shap_force_data: Dict[str, Any]
    # Local LIME
    lime_explanation: Dict[str, Any]
    lime_plot: str = Field(description="Base64-encoded PNG")
    # Native feature importances
    feature_importances: List[FeatureImportance]
    feature_importance_method: str
    # Meta
    instance_index: int
    message: str


# ---------------------------------------------------------------------------
# Metrics Response
# ---------------------------------------------------------------------------


class MetricsResponse(BaseModel):
    session_id: str
    task_type: str
    n_samples: int
    # Classification
    accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_weighted: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    # Regression
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    # Common
    notes: Dict[str, str] = Field(default_factory=dict)
    message: str


# ---------------------------------------------------------------------------
# Error Response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    suggestion: Optional[str] = None
