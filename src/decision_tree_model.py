import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
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
# CREATE DECISION TREE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            DecisionTreeRegressor(
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("=" * 70)
print("DECISION TREE REGRESSION MODEL")
print("=" * 70)

print()

print("Training model...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print()

print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print()

print(f"MAE:  ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R²:   {r2:.4f}")

print()

print("=" * 70)
print("DECISION TREE REGRESSION COMPLETE")
print("=" * 70)