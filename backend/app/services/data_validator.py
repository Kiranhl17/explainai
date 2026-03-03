"""
Data Validation Service
========================
Handles CSV ingestion, validation, and preprocessing before
passing data to the explanation engine.

Design Principles:
  - Fail fast with informative error messages
  - Never silently impute missing values without user awareness
  - Validate feature compatibility against the loaded model
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("explainai.data_validator")

# Maximum dataset size to protect against memory exhaustion
MAX_ROWS = 50_000
MAX_COLS = 500
MAX_FILE_SIZE_MB = 50


def load_and_validate_csv(
    file_path: str,
    model_info: Optional[Dict] = None,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict]:
    """
    Load a CSV file and validate its compatibility with the loaded model.

    Parameters
    ----------
    file_path : str — path to the CSV file
    model_info : dict — model metadata from model_handler (optional)
    target_column : str — name of the target column (optional)

    Returns
    -------
    X : pd.DataFrame — feature matrix (excludes target column)
    y : pd.Series | None — target vector (if target_column provided)
    data_info : dict — dataset metadata and validation report
    """
    logger.info(f"Loading CSV from: {file_path}")

    # --- Load CSV ---
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file: {exc}") from exc

    # --- Size validation ---
    if df.shape[0] > MAX_ROWS:
        logger.warning(f"Dataset has {df.shape[0]} rows — truncating to {MAX_ROWS}")
        df = df.head(MAX_ROWS)

    if df.shape[1] > MAX_COLS:
        raise ValueError(
            f"Dataset has {df.shape[1]} columns (max {MAX_COLS}). "
            "Please reduce dimensionality before uploading."
        )

    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    # --- Separate target column ---
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column].copy()
        df = df.drop(columns=[target_column])
        logger.info(f"Target column '{target_column}' separated | classes: {y.nunique()}")
    elif target_column:
        logger.warning(
            f"Requested target column '{target_column}' not found in CSV. "
            f"Available: {list(df.columns)}"
        )

    # --- Drop non-numeric columns (with warning) ---
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(
            f"Dropping non-numeric columns: {non_numeric}. "
            "Consider encoding categoricals before upload."
        )
        df = df.select_dtypes(include=[np.number])

    if df.empty:
        raise ValueError(
            "No numeric columns found after preprocessing. "
            "Ensure your dataset contains numeric features."
        )

    # --- Missing value report ---
    missing_counts = df.isnull().sum()
    missing_features = missing_counts[missing_counts > 0].to_dict()

    if missing_features:
        logger.warning(f"Missing values detected: {missing_features}. Imputing with column medians.")
        df = df.fillna(df.median(numeric_only=True))

    # --- Model compatibility check ---
    compatibility_issues = []
    if model_info and model_info.get("n_features") is not None:
        expected_n = model_info["n_features"]
        actual_n = df.shape[1]
        if expected_n != actual_n:
            compatibility_issues.append(
                f"Feature count mismatch: model expects {expected_n} features, "
                f"dataset has {actual_n}. Ensure columns match training data."
            )

    # --- Data summary ---
    data_info = {
        "n_rows": df.shape[0],
        "n_features": df.shape[1],
        "feature_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values_imputed": missing_features,
        "dropped_columns": non_numeric,
        "has_target": y is not None,
        "target_distribution": (
            y.value_counts().to_dict() if y is not None and y.nunique() < 20 else None
        ),
        "compatibility_issues": compatibility_issues,
        "statistics": {
            col: {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            }
            for col in df.columns[:20]  # Limit for performance
        },
    }

    logger.info(
        f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} features | "
        f"issues={len(compatibility_issues)}"
    )

    return df, y, data_info
