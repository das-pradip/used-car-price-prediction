import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/processed/car_data_feature_engineered.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURES]

y = df["selling_price"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# BASELINE 1 — MEAN
# ============================================================

mean_prediction = y_train.mean()

y_pred_mean = [mean_prediction] * len(y_test)


# ============================================================
# BASELINE 2 — MEDIAN
# ============================================================

median_prediction = y_train.median()

y_pred_median = [median_prediction] * len(y_test)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = mean_squared_error(
        y_true,
        y_pred
    ) ** 0.5

    r2 = r2_score(y_true, y_pred)

    print()
    print(name)
    print("-" * 50)

    print(f"MAE:  ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R²:   {r2:.4f}")


# ============================================================
# DISPLAY BASELINE PREDICTIONS
# ============================================================

print("=" * 70)
print("BASELINE MODEL")
print("=" * 70)

print()

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print()

print(f"Mean training price:   ₹{mean_prediction:,.2f}")
print(f"Median training price: ₹{median_prediction:,.2f}")


# ============================================================
# EVALUATE
# ============================================================

evaluate_model(
    "MEAN BASELINE",
    y_test,
    y_pred_mean
)

evaluate_model(
    "MEDIAN BASELINE",
    y_test,
    y_pred_median
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("BASELINE MODEL COMPLETE")
print("=" * 70)