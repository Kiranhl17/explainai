import joblib
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_diabetes,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Choose a dataset:")
print("1 - Breast Cancer (classification, 30 features)")
print("2 - Wine Quality  (classification, 13 features)")
print("3 - Diabetes      (regression,     10 features)")
choice = input("Enter 1, 2 or 3: ").strip()

if choice == "1":
    data = load_breast_cancer()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    task = "classification"
    name = "breast_cancer"

elif choice == "2":
    data = load_wine()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    task = "classification"
    name = "wine"

elif choice == "3":
    data = load_diabetes()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    task = "regression"
    name = "diabetes"

else:
    print("Invalid choice")
    exit()

# Build DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Train/test split
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
print(f"\nTraining RandomForest on {name} dataset...")
model.fit(X_train, y_train)

if task == "classification":
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")

# Save model
model_file = f"{name}_model.pkl"
joblib.dump(model, model_file)
print(f"Saved: {model_file}")

# Save test data with target
test_df = X_test.copy()
test_df["target"] = y_test.values
csv_file = f"{name}_data.csv"
test_df.to_csv(csv_file, index=False)
print(f"Saved: {csv_file} ({len(test_df)} rows)")

print(f"\nReady to upload to ExplainAI:")
print(f"  Model:  {model_file}")
print(f"  Data:   {csv_file}")
print(f"  Target: target")