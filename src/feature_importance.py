import pandas as pd

from sklearn.model_selection import train_test_split
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
# TRAIN MODEL
# ============================================================

print("=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

print()

print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# GET FEATURE NAMES
# ============================================================

preprocessor = model.named_steps["preprocessor"]

regressor = model.named_steps["regressor"]


feature_names = preprocessor.get_feature_names_out()


# ============================================================
# GET FEATURE IMPORTANCES
# ============================================================

importances = regressor.feature_importances_


# ============================================================
# CREATE FEATURE IMPORTANCE DATAFRAME
# ============================================================

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances
    }
)


# ============================================================
# SORT BY IMPORTANCE
# ============================================================

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


# ============================================================
# DISPLAY TOP 20 FEATURES
# ============================================================

print()

print("=" * 70)
print("TOP 20 FEATURES")
print("=" * 70)

print()

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# CHECK TOTAL IMPORTANCE
# ============================================================

print()

print("=" * 70)
print("IMPORTANCE VALIDATION")
print("=" * 70)

print()

print(
    f"Total importance: "
    f"{importance_df['importance'].sum():.6f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = (
    "data/processed/random_forest_feature_importance.csv"
)

importance_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print()

print("Feature importance saved to:")

print(OUTPUT_PATH)


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)