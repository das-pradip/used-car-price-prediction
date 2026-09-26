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

DATA_PATH = Path(
    "data/processed/car_data_feature_engineered.csv"
)

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


TECHNICAL_FEATURES = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("PRODUCTION MODEL — MISSING INPUT ANALYSIS")
print("=" * 80)

print("\n[1/6] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# PREPARE DATA
# ============================================================

print("\n[2/6] Preparing features and target...")

FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)

X = df[FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"Training rows: {len(X_train)}")
print(f"Test rows:     {len(X_test)}")


# ============================================================
# TRAIN THE EXACT PRODUCTION MODEL
# ============================================================

print("\n[3/6] Training exact production model...")

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

pipeline.fit(
    X_train,
    y_train
)

print("Production model trained.")


# ============================================================
# SCENARIO 1 — NORMAL INPUT
# ============================================================

print("\n[4/6] Evaluating normal complete input...")

X_test_full = X_test.copy()

y_pred_full = pipeline.predict(
    X_test_full
)

full_mae, full_rmse, full_r2 = calculate_metrics(
    y_test,
    y_pred_full
)

print(f"MAE :  ₹{full_mae:,.2f}")
print(f"RMSE:  ₹{full_rmse:,.2f}")
print(f"R²  :  {full_r2:.4f}")


# ============================================================
# SCENARIO 2 — POWER + TORQUE MISSING
# ============================================================

print("\n[5/6] Evaluating power + torque missing...")

X_test_power_torque_missing = X_test.copy()

X_test_power_torque_missing[
    ["max_power_bhp", "torque_nm"]
] = np.nan

y_pred_power_torque = pipeline.predict(
    X_test_power_torque_missing
)

pt_mae, pt_rmse, pt_r2 = calculate_metrics(
    y_test,
    y_pred_power_torque
)

print(f"MAE :  ₹{pt_mae:,.2f}")
print(f"RMSE:  ₹{pt_rmse:,.2f}")
print(f"R²  :  {pt_r2:.4f}")


# ============================================================
# SCENARIO 3 — ALL TECHNICAL FEATURES MISSING
# ============================================================

print("\n[6/6] Evaluating all technical features missing...")

X_test_all_missing = X_test.copy()

X_test_all_missing[
    TECHNICAL_FEATURES
] = np.nan

y_pred_all_missing = pipeline.predict(
    X_test_all_missing
)

all_mae, all_rmse, all_r2 = calculate_metrics(
    y_test,
    y_pred_all_missing
)

print(f"MAE :  ₹{all_mae:,.2f}")
print(f"RMSE:  ₹{all_rmse:,.2f}")
print(f"R²  :  {all_r2:.4f}")


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    [
        {
            "scenario": "Normal complete input",
            "MAE": full_mae,
            "RMSE": full_rmse,
            "R2": full_r2,
        },
        {
            "scenario": "Power + torque missing",
            "MAE": pt_mae,
            "RMSE": pt_rmse,
            "R2": pt_r2,
        },
        {
            "scenario": "All technical missing",
            "MAE": all_mae,
            "RMSE": all_rmse,
            "R2": all_r2,
        },
    ]
)


print("\n" + "=" * 80)
print("PRODUCTION MODEL COMPARISON")
print("=" * 80)

display_results = results.copy()

display_results["MAE"] = display_results["MAE"].map(
    lambda x: f"₹{x:,.2f}"
)

display_results["RMSE"] = display_results["RMSE"].map(
    lambda x: f"₹{x:,.2f}"
)

display_results["R2"] = display_results["R2"].map(
    lambda x: f"{x:.4f}"
)

print(
    display_results.to_string(
        index=False
    )
)


# ============================================================
# CHANGE FROM NORMAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("CHANGE FROM NORMAL COMPLETE INPUT")
print("=" * 80)


scenarios_to_compare = [
    (
        "Power + torque missing",
        pt_mae,
        pt_rmse,
        pt_r2,
    ),
    (
        "All technical missing",
        all_mae,
        all_rmse,
        all_r2,
    ),
]


for name, mae, rmse, r2 in scenarios_to_compare:

    print(f"\n{name}")

    print(
        f"MAE change : "
        f"{'+' if mae - full_mae >= 0 else ''}"
        f"₹{mae - full_mae:,.2f}"
    )

    print(
        f"RMSE change: "
        f"{'+' if rmse - full_rmse >= 0 else ''}"
        f"₹{rmse - full_rmse:,.2f}"
    )

    print(
        f"R² change  : "
        f"{'+' if r2 - full_r2 >= 0 else ''}"
        f"{r2 - full_r2:.4f}"
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print(
    """
This experiment uses the same trained production model
and removes technical information only at prediction time.

Therefore, the results measure the effect of missing user
input on the existing production model.

These results should be used to decide how much technical
information the prediction interface should request.
"""
)

print("=" * 80)
print("PRODUCTION MISSING INPUT ANALYSIS COMPLETE")
print("=" * 80)