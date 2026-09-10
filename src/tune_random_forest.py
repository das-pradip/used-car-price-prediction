import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
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
# CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "regressor",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [None, 15, 25],
    "regressor__min_samples_leaf": [1, 2, 4]
}


# ============================================================
# GRID SEARCH
# ============================================================

print("=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

print()

print("Starting GridSearchCV...")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)


# ============================================================
# TRAIN GRID SEARCH
# ============================================================

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# BEST PARAMETERS
# ============================================================

print()

print("=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print()

print(grid_search.best_params_)

print()

print(
    f"Best CV MAE: "
    f"₹{-grid_search.best_score_:,.2f}"
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


# ============================================================
# TEST PREDICTIONS
# ============================================================

y_pred = best_model.predict(X_test)


# ============================================================
# TEST METRICS
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
# TEST RESULTS
# ============================================================

print()

print("=" * 70)
print("TUNED MODEL TEST PERFORMANCE")
print("=" * 70)

print()

print(f"MAE:  ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R²:   {r2:.4f}")


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)
print("RANDOM FOREST TUNING COMPLETE")
print("=" * 70)