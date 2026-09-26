from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from ml_preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    create_preprocessor,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/car_data_feature_engineered.csv"
)

MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "used_car_price_model.joblib"

TARGET = "selling_price"

RANDOM_STATE = 42
TEST_SIZE = 0.20


MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("SAVE FINAL USED CAR PRICE MODEL")
print("=" * 80)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\n[1/6] Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ------------------------------------------------------------
# 2. Prepare features and target
# ------------------------------------------------------------

print("\n[2/6] Preparing features and target...")

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURES]
y = df[TARGET]

print(f"Features: {len(FEATURES)}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ------------------------------------------------------------
# 3. Create train/test split
# ------------------------------------------------------------

print("\n[3/6] Creating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"Training rows: {len(X_train)}")
print(f"Test rows:     {len(X_test)}")


# ------------------------------------------------------------
# 4. Build final pipeline
# ------------------------------------------------------------

print("\n[4/6] Building final pipeline...")

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
# 5. Train on complete training set
# ------------------------------------------------------------

print("\n[5/6] Training final model...")

pipeline.fit(X_train, y_train)

print("Model training complete.")


# ------------------------------------------------------------
# 6. Save model
# ------------------------------------------------------------

print("\n[6/6] Saving model...")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"\nModel saved successfully:")
print(MODEL_PATH)


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL INFORMATION")
print("=" * 80)

print(f"Model type       : RandomForestRegressor")
print(f"Number of trees  : {MODEL_PARAMS['n_estimators']}")
print(f"Maximum depth    : {MODEL_PARAMS['max_depth']}")
print(f"Min samples leaf : {MODEL_PARAMS['min_samples_leaf']}")
print(f"Random state     : {RANDOM_STATE}")

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")

print("\n" + "=" * 80)
print("MODEL SAVING COMPLETE")
print("=" * 80)