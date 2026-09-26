from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    create_preprocessor,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/car_data_feature_engineered.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

TARGET = "selling_price"

# ============================================================
# HELPER
# ============================================================

def format_inr(value):
    return f"₹{value:,.2f}"


# ============================================================
# VALIDATION
# ============================================================

print("=" * 80)
print("FINAL MODEL VALIDATION")
print("=" * 80)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\n[1/8] Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ------------------------------------------------------------
# 2. Validate required columns
# ------------------------------------------------------------

print("\n[2/8] Validating required columns...")

required_columns = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
    + [TARGET]
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are present.")


# ------------------------------------------------------------
# 3. Prepare X and y
# ------------------------------------------------------------

print("\n[3/8] Preparing features and target...")

X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

if y.isna().any():
    raise ValueError("Target contains missing values.")

print("Target contains no missing values.")


# ------------------------------------------------------------
# 4. Reproducible train/test split
# ------------------------------------------------------------

print("\n[4/8] Creating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

if len(X_train) != 5540:
    raise ValueError(
        f"Unexpected training size: {len(X_train)}"
    )

if len(X_test) != 1386:
    raise ValueError(
        f"Unexpected test size: {len(X_test)}"
    )

print("Train/test split matches the established project split.")


# ------------------------------------------------------------
# 5. Build final pipeline
# ------------------------------------------------------------

print("\n[5/8] Building final model pipeline...")

preprocessor = create_preprocessor()

model = RandomForestRegressor(
    **MODEL_PARAMS
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

print("Pipeline created successfully.")


# ------------------------------------------------------------
# 6. Train model
# ------------------------------------------------------------

print("\n[6/8] Training final tuned Random Forest...")

pipeline.fit(X_train, y_train)

print("Training completed successfully.")


# ------------------------------------------------------------
# 7. Generate predictions
# ------------------------------------------------------------

print("\n[7/8] Generating test predictions...")

y_pred = pipeline.predict(X_test)

if np.isnan(y_pred).any():
    raise ValueError(
        "Predictions contain NaN values."
    )

if np.isinf(y_pred).any():
    raise ValueError(
        "Predictions contain infinite values."
    )

print(f"Prediction count: {len(y_pred)}")
print("Predictions contain no NaN or infinite values.")


# ------------------------------------------------------------
# 8. Calculate final metrics
# ------------------------------------------------------------

print("\n[8/8] Calculating final metrics...")

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 80)
print("FINAL VALIDATION RESULTS")
print("=" * 80)

print(f"MAE :  {format_inr(mae)}")
print(f"RMSE:  {format_inr(rmse)}")
print(f"R²  :  {r2:.4f}")


# ============================================================
# EXPECTED RESULT CHECK
# ============================================================

print("\n" + "=" * 80)
print("REPRODUCIBILITY CHECK")
print("=" * 80)

EXPECTED_MAE = 70200.55
EXPECTED_RMSE = 120388.55
EXPECTED_R2 = 0.9339

MAE_TOLERANCE = 1.00
RMSE_TOLERANCE = 1.00
R2_TOLERANCE = 0.0001


mae_match = abs(mae - EXPECTED_MAE) <= MAE_TOLERANCE
rmse_match = abs(rmse - EXPECTED_RMSE) <= RMSE_TOLERANCE
r2_match = abs(r2 - EXPECTED_R2) <= R2_TOLERANCE


print(
    f"MAE expected:  {format_inr(EXPECTED_MAE)}"
)
print(
    f"MAE actual:    {format_inr(mae)}"
)
print(
    f"MAE match:     {'PASS' if mae_match else 'FAIL'}"
)

print()

print(
    f"RMSE expected: {format_inr(EXPECTED_RMSE)}"
)
print(
    f"RMSE actual:   {format_inr(rmse)}"
)
print(
    f"RMSE match:    {'PASS' if rmse_match else 'FAIL'}"
)

print()

print(
    f"R² expected:    {EXPECTED_R2:.4f}"
)
print(
    f"R² actual:      {r2:.4f}"
)
print(
    f"R² match:       {'PASS' if r2_match else 'FAIL'}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)

if mae_match and rmse_match and r2_match:
    print("FINAL MODEL VALIDATION: PASS")
    print("=" * 80)
    print("The final model is reproducible.")
else:
    print("FINAL MODEL VALIDATION: CHECK REQUIRED")
    print("=" * 80)
    print("Metrics differ from the established results.")


print("\nValidation complete.")