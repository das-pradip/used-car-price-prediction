import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_preprocessing import (
    create_preprocessor,
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
# CREATE PREPROCESSOR
# ============================================================

preprocessor = create_preprocessor()


# ============================================================
# CREATE RANDOM FOREST MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("=" * 70)
print("RANDOM FOREST REGRESSION MODEL")
print("=" * 70)

print()

print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)


# ============================================================
# TRAINING PERFORMANCE
# ============================================================

train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

train_rmse = mean_squared_error(
    y_train,
    y_train_pred
) ** 0.5

train_r2 = r2_score(
    y_train,
    y_train_pred
)


# ============================================================
# TESTING PERFORMANCE
# ============================================================

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_rmse = mean_squared_error(
    y_test,
    y_test_pred
) ** 0.5

test_r2 = r2_score(
    y_test,
    y_test_pred
)


# ============================================================
# TRAINING RESULTS
# ============================================================

print()

print("=" * 70)
print("TRAINING PERFORMANCE")
print("=" * 70)

print()

print(f"MAE:  ₹{train_mae:,.2f}")
print(f"RMSE: ₹{train_rmse:,.2f}")
print(f"R²:   {train_r2:.4f}")


# ============================================================
# TESTING RESULTS
# ============================================================

print()

print("=" * 70)
print("TESTING PERFORMANCE")
print("=" * 70)

print()

print(f"MAE:  ₹{test_mae:,.2f}")
print(f"RMSE: ₹{test_rmse:,.2f}")
print(f"R²:   {test_r2:.4f}")


# ============================================================
# OVERFITTING GAP
# ============================================================

print()

print("=" * 70)
print("TRAINING vs TESTING GAP")
print("=" * 70)

print()

print(f"MAE gap:  ₹{test_mae - train_mae:,.2f}")
print(f"RMSE gap: ₹{test_rmse - train_rmse:,.2f}")
print(f"R² gap:   {train_r2 - test_r2:.4f}")


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)
print("RANDOM FOREST REGRESSION COMPLETE")
print("=" * 70)
