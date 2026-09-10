import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. Load feature-engineered dataset
# ============================================================

DATA_PATH = "data/processed/car_data_feature_engineered.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("PERMUTATION IMPORTANCE ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 2. Define target and features
# ============================================================

TARGET = "selling_price"

NUMERICAL_FEATURES = [
    "car_age",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]

CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 3. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain shape:")
print(X_train.shape)

print("Test shape:")
print(X_test.shape)


# ============================================================
# 4. Create preprocessing pipeline
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]),
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# 5. Preprocess data
# ============================================================

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print("\nTransformed train shape:")
print(X_train_transformed.shape)

print("Transformed test shape:")
print(X_test_transformed.shape)


# ============================================================
# 6. Train tuned Random Forest
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_transformed,
    y_train
)


# ============================================================
# 7. Baseline test performance
# ============================================================

baseline_predictions = model.predict(X_test_transformed)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("\n" + "=" * 70)
print("BASELINE MODEL PERFORMANCE")
print("=" * 70)

print(f"Test MAE: ₹{baseline_mae:,.2f}")


# ============================================================
# 8. Permutation importance
# ============================================================

print("\nCalculating permutation importance...")
print("This may take some time...")

result = permutation_importance(
    model,
    X_test_transformed,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 9. Get transformed feature names
# ============================================================

feature_names = preprocessor.get_feature_names_out()


# ============================================================
# 10. Create feature importance DataFrame
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
})

importance_df = importance_df.sort_values(
    by="importance_mean",
    ascending=False
)


# ============================================================
# 11. Display top 20 transformed features
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 PERMUTATION IMPORTANCES")
print("=" * 70)

print(
    importance_df.head(20).to_string(index=False)
)


# ============================================================
# 12. Save detailed results
# ============================================================

OUTPUT_PATH = (
    "data/processed/"
    "random_forest_permutation_importance.csv"
)

importance_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDetailed results saved to:")
print(OUTPUT_PATH)


# ============================================================
# 13. Aggregate importance by original feature
# ============================================================

# One-hot encoded categorical variables produce multiple
# transformed columns. We aggregate them back to the
# original feature level.

aggregated_importance = []

for feature in FEATURES:

    matching_rows = []

    if feature in NUMERICAL_FEATURES:

        matching_rows = importance_df[
            importance_df["feature"] == f"numerical__{feature}"
        ]

    else:

        prefix = f"categorical__{feature}_"

        matching_rows = importance_df[
            importance_df["feature"].str.startswith(prefix)
        ]

    if len(matching_rows) > 0:

        aggregated_importance.append({
            "feature": feature,
            "importance_mean": matching_rows[
                "importance_mean"
            ].sum(),
            "importance_std": (
                matching_rows["importance_std"]
                .pow(2)
                .sum()
                ** 0.5
            )
        })


aggregated_df = pd.DataFrame(
    aggregated_importance
)

aggregated_df = aggregated_df.sort_values(
    by="importance_mean",
    ascending=False
)


# ============================================================
# 14. Display original-feature importance
# ============================================================

print("\n" + "=" * 70)
print("AGGREGATED ORIGINAL FEATURE IMPORTANCE")
print("=" * 70)

print(
    aggregated_df.to_string(index=False)
)


# ============================================================
# 15. Save aggregated results
# ============================================================

AGGREGATED_OUTPUT_PATH = (
    "data/processed/"
    "random_forest_permutation_importance_aggregated.csv"
)

aggregated_df.to_csv(
    AGGREGATED_OUTPUT_PATH,
    index=False
)

print("\nAggregated results saved to:")
print(AGGREGATED_OUTPUT_PATH)


print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)