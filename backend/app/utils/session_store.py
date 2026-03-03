"""
Session Store
=============
Thread-safe in-memory session store mapping session IDs to:
  - Loaded model objects
  - Loaded datasets
  - Cached explanation results

Design Note:
  In a horizontally-scaled deployment (multiple workers), this in-memory
  store should be replaced with a Redis-backed store (e.g., via aioredis).
  For the scope of this MSc project, a single-worker deployment is assumed.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("explainai.session_store")


@dataclass
class SessionState:
    """
    Encapsulates all per-session artefacts.

    Attributes
    ----------
    model : Any
        The deserialized scikit-learn estimator.
    model_info : dict
        Metadata about the model (type, n_features, etc.).
    X : pd.DataFrame | None
        Feature matrix uploaded by the user.
    y : pd.Series | None
        Target vector (if present in the CSV).
    feature_names : list[str]
        Column names of X.
    shap_values : np.ndarray | None
        Cached SHAP values (shape: [n_samples, n_features]).
    shap_base_value : float | None
        SHAP expected value (E[f(x)]).
    lime_explanation : dict | None
        Serialised LIME explanation for the selected instance.
    metrics : dict | None
        Cached model performance metrics.
    instance_index : int
        Index of the instance selected for local explanation.
    """

    model: Any = None
    model_info: Dict = field(default_factory=dict)
    X: Optional[pd.DataFrame] = None
    y: Optional[pd.Series] = None
    feature_names: list = field(default_factory=list)
    shap_values: Optional[np.ndarray] = None
    shap_base_value: Optional[float] = None
    lime_explanation: Optional[dict] = None
    metrics: Optional[dict] = None
    instance_index: int = 0


class SessionStore:
    """
    Thread-safe dictionary-backed session registry.
    """

    def __init__(self):
        self._store: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._store:
                logger.info(f"Creating new session: {session_id}")
                self._store[session_id] = SessionState()
            return self._store[session_id]

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._store.get(session_id)

    def delete(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)

    def __len__(self):
        return len(self._store)


# Singleton instance used across all routes/services
session_store = SessionStore()
