import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────
# EDIT THESE THREE LINES ONLY
CSV_FILE    = "Middle_East_Economic_Data_1990_2024_with_Oil.csv"
TARGET_COL  = "Life_expectancy_years"
DROP_COLS   = ["Country", "Country_Code", "Year"]
# ─────────────────────────────────────────

print(f"Loading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
print(f"Shape: {df.shape}")

# Drop unwanted columns
if DROP_COLS:
    df = df.drop(columns=DROP_COLS, errors="ignore")
    print(f"Dropped columns: {DROP_COLS}")

# Drop non-numeric columns
non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
if TARGET_COL in non_numeric:
    non_numeric.remove(TARGET_COL)
if non_numeric:
    print(f"Dropping non-numeric columns: {non_numeric}")
    df = df.drop(columns=non_numeric)

# Handle missing values
df = df.dropna()
print(f"Shape after cleaning: {df.shape}")

# Split features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"Features: {list(X.columns)}")
print(f"Target range: {y.min():.2f} to {y.max():.2f} years")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train RandomForestRegressor
print("\nTraining RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R² Score : {r2:.4f}  (1.0 = perfect)")
print(f"MAE      : {mae:.4f} years average error")

# Save model
model_name = "middle_east_life_expectancy_model.pkl"
data_name  = "middle_east_life_expectancy_data.csv"

joblib.dump(model, model_name)
print(f"\nSaved: {model_name}")

# Save test dataset with target
test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values
test_df.to_csv(data_name, index=False)
print(f"Saved: {data_name} ({len(test_df)} rows)")

print(f"\n✓ Ready to upload to ExplainAI:")
print(f"  Model:         {model_name}")
print(f"  Data:          {data_name}")
print(f"  Target column: {TARGET_COL}")