import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

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
# CREATE TUNED RANDOM FOREST PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# CREATE 5-FOLD CROSS-VALIDATION
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

print("=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

print()

print("Running cross-validation...")


scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}


results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


# ============================================================
# EXTRACT RESULTS
# ============================================================

mae_scores = -results["test_mae"]

rmse_scores = -results["test_rmse"]

r2_scores = results["test_r2"]


# ============================================================
# DISPLAY INDIVIDUAL FOLD RESULTS
# ============================================================

print()

print("=" * 70)
print("FOLD RESULTS")
print("=" * 70)

print()

for i in range(5):

    print(
        f"Fold {i + 1}: "
        f"MAE = ₹{mae_scores[i]:,.2f}, "
        f"RMSE = ₹{rmse_scores[i]:,.2f}, "
        f"R² = {r2_scores[i]:.4f}"
    )


# ============================================================
# DISPLAY AVERAGE RESULTS
# ============================================================

print()

print("=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print()

print(
    f"Mean MAE:  ₹{mae_scores.mean():,.2f}"
)

print(
    f"Std MAE:   ₹{mae_scores.std():,.2f}"
)

print()

print(
    f"Mean RMSE: ₹{rmse_scores.mean():,.2f}"
)

print(
    f"Std RMSE:  ₹{rmse_scores.std():,.2f}"
)

print()

print(
    f"Mean R²:   {r2_scores.mean():.4f}"
)

print(
    f"Std R²:    {r2_scores.std():.4f}"
)


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)
print("5-FOLD CROSS-VALIDATION COMPLETE")
print("=" * 70)