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


ALL_TECHNICAL_FEATURES = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("MISSING INPUT SCENARIO ANALYSIS")
print("=" * 80)

print("\n[1/5] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# PREPARE DATA
# ============================================================

print("\n[2/5] Preparing train/test split...")

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

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
# SCENARIOS
# ============================================================

scenarios = {
    "Scenario 1 - All technical features": [],
    
    "Scenario 2 - Mileage + Seats only": [
        "engine_cc",
        "max_power_bhp",
        "torque_nm",
    ],
    
    "Scenario 3 - Mileage + Engine + Seats": [
        "max_power_bhp",
        "torque_nm",
    ],
    
    "Scenario 4 - Only basic information": [
        "mileage_value",
        "engine_cc",
        "max_power_bhp",
        "torque_nm",
        "seats",
    ],
    
    "Scenario 5 - All technical missing": [
        "mileage_value",
        "engine_cc",
        "max_power_bhp",
        "torque_nm",
        "seats",
    ],
}


# ============================================================
# BUILD AND EVALUATE MODELS
# ============================================================

print("\n[3/5] Training and evaluating scenarios...")

results = []


for scenario_name, missing_features in scenarios.items():

    print("\n" + "-" * 80)
    print(scenario_name)
    print("-" * 80)

    X_train_scenario = X_train.copy()
    X_test_scenario = X_test.copy()

    if scenario_name == "Scenario 4 - Only basic information":
        # Keep all technical features missing.
        # This represents a user who knows only
        # the basic car information.
        for feature in missing_features:
            X_train_scenario[feature] = np.nan
            X_test_scenario[feature] = np.nan

    elif scenario_name == "Scenario 5 - All technical missing":
        # Same technical-information condition as Scenario 4.
        # Kept as a separate explicit scenario for reporting.
        for feature in missing_features:
            X_train_scenario[feature] = np.nan
            X_test_scenario[feature] = np.nan

    else:
        for feature in missing_features:
            X_train_scenario[feature] = np.nan
            X_test_scenario[feature] = np.nan

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
        X_train_scenario,
        y_train,
    )

    y_pred = pipeline.predict(
        X_test_scenario
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append(
        {
            "scenario": scenario_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

    print(f"MAE :  ₹{mae:,.2f}")
    print(f"RMSE:  ₹{rmse:,.2f}")
    print(f"R²  :  {r2:.4f}")


# ============================================================
# RESULTS TABLE
# ============================================================

print("\n" + "=" * 80)
print("SCENARIO COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(results)

display_df = results_df.copy()

display_df["MAE"] = display_df["MAE"].map(
    lambda x: f"₹{x:,.2f}"
)

display_df["RMSE"] = display_df["RMSE"].map(
    lambda x: f"₹{x:,.2f}"
)

display_df["R2"] = display_df["R2"].map(
    lambda x: f"{x:.4f}"
)

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# COMPARISON AGAINST CURRENT MODEL
# ============================================================

baseline_mae = results_df.iloc[0]["MAE"]
baseline_rmse = results_df.iloc[0]["RMSE"]
baseline_r2 = results_df.iloc[0]["R2"]

print("\n" + "=" * 80)
print("CHANGE FROM FULL-INFORMATION SCENARIO")
print("=" * 80)

for _, row in results_df.iloc[1:].iterrows():

    mae_change = row["MAE"] - baseline_mae
    rmse_change = row["RMSE"] - baseline_rmse
    r2_change = row["R2"] - baseline_r2

    print(f"\n{row['scenario']}")

    print(
        f"MAE change : "
        f"{'+' if mae_change >= 0 else ''}"
        f"₹{mae_change:,.2f}"
    )

    print(
        f"RMSE change: "
        f"{'+' if rmse_change >= 0 else ''}"
        f"₹{rmse_change:,.2f}"
    )

    print(
        f"R² change  : "
        f"{'+' if r2_change >= 0 else ''}"
        f"{r2_change:.4f}"
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("IMPORTANT INTERPRETATION")
print("=" * 80)

print(
    """
This experiment measures how model performance changes when
technical vehicle specifications are unavailable.

The experiment does NOT prove that missing information is
acceptable for every individual prediction.

It measures aggregate test-set performance under each
simulated missing-information scenario.
"""
)

print("=" * 80)
print("MISSING INPUT ANALYSIS COMPLETE")
print("=" * 80)