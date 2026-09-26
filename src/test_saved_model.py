from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/used_car_price_model.joblib"
)

DATA_PATH = Path(
    "data/processed/car_data_feature_engineered.csv"
)


FEATURES = [
    "car_age",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("TEST SAVED USED CAR PRICE MODEL")
print("=" * 80)


# ------------------------------------------------------------
# 1. Check files
# ------------------------------------------------------------

print("\n[1/5] Checking required files...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("Model file found.")
print("Dataset file found.")


# ------------------------------------------------------------
# 2. Load model
# ------------------------------------------------------------

print("\n[2/5] Loading saved model...")

model = joblib.load(MODEL_PATH)

print("Saved model loaded successfully.")
print(f"Model type: {type(model).__name__}")


# ------------------------------------------------------------
# 3. Load dataset
# ------------------------------------------------------------

print("\n[3/5] Loading feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ------------------------------------------------------------
# 4. Select a real complete example
# ------------------------------------------------------------

print("\n[4/5] Selecting a test example...")

required_test_columns = FEATURES + [
    "name",
    "selling_price",
]

missing_columns = [
    column
    for column in required_test_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required test columns are missing: "
        f"{missing_columns}"
    )


complete_rows = df[
    required_test_columns
].dropna()


if complete_rows.empty:
    raise ValueError(
        "No complete feature row found for testing."
    )


sample = complete_rows.iloc[[0]]

car_name = sample.iloc[0]["name"]

X_sample = sample[FEATURES]

actual_price = sample.iloc[0]["selling_price"]


print("\nExample car:")
print(f"Name: {car_name}")

print("\nInput features:")

for feature in FEATURES:
    print(
        f"  {feature}: "
        f"{sample.iloc[0][feature]}"
    )


# ------------------------------------------------------------
# 5. Predict
# ------------------------------------------------------------

print("\n[5/5] Generating prediction...")

prediction = model.predict(X_sample)[0]

absolute_error = abs(
    actual_price - prediction
)


print("\n" + "=" * 80)
print("PREDICTION RESULT")
print("=" * 80)

print(f"Car:             {car_name}")
print(f"Actual price:    ₹{actual_price:,.2f}")
print(f"Predicted price: ₹{prediction:,.2f}")
print(f"Absolute error:  ₹{absolute_error:,.2f}")

print("\n" + "=" * 80)
print("SAVED MODEL TEST: PASS")
print("=" * 80)

print(
    "The saved .joblib model successfully "
    "loaded and generated a prediction."
)