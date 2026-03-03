"""
Test Model & Dataset Generator
================================
Generates a sample RandomForestClassifier and matching CSV dataset
for testing the ExplainAI system locally.

Usage:
    python generate_test_data.py

Outputs:
    test_model.pkl   — Trained RandomForestClassifier
    test_data.csv    — Feature matrix + target column (label)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---- Generate synthetic dataset ----
print("Generating synthetic dataset…")
X, y = make_classification(
    n_samples=500,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=2,
    random_state=42,
)

feature_names = [
    "age", "income", "credit_score", "debt_ratio",
    "employment_years", "loan_amount", "account_balance",
    "num_transactions", "missed_payments", "credit_utilization",
    "savings_rate", "investment_score"
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---- Train model ----
print("Training RandomForestClassifier…")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {acc:.4f}")

# ---- Save model ----
joblib.dump(model, "test_model.pkl")
print("Saved: test_model.pkl")

# ---- Save dataset (test split with target) ----
df = pd.DataFrame(X_test, columns=feature_names)
df["label"] = y_test
df.to_csv("test_data.csv", index=False)
print(f"Saved: test_data.csv ({len(df)} rows, target column = 'label')")

print("\n✓ Ready to upload to ExplainAI:")
print("  Model:  test_model.pkl")
print("  Data:   test_data.csv")
print("  Target: label")
