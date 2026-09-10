import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. Load feature-engineered data
# ============================================================

DATA_PATH = "data/processed/car_data_feature_engineered.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. Define target
# ============================================================

TARGET = "selling_price"

y = df[TARGET]


# ============================================================
# 3. Full feature set
# ============================================================

NUMERICAL_FEATURES_FULL = [
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

FEATURES_FULL = NUMERICAL_FEATURES_FULL + CATEGORICAL_FEATURES


# ============================================================
# 4. Ablation feature set
#    Remove max_power_bhp and torque_nm
# ============================================================

NUMERICAL_FEATURES_ABLATION = [
    "car_age",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "seats",
]

FEATURES_ABLATION = NUMERICAL_FEATURES_ABLATION + CATEGORICAL_FEATURES


# ============================================================
# 5. Train-test split
# ============================================================

X_full = df[FEATURES_FULL]
X_ablation = df[FEATURES_ABLATION]

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.20,
    random_state=42
)

X_train_ablation, X_test_ablation, _, _ = train_test_split(
    X_ablation,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 6. Create preprocessing pipelines
# ============================================================

preprocessor_full = ColumnTransformer(
    transformers=[
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            NUMERICAL_FEATURES_FULL
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

preprocessor_ablation = ColumnTransformer(
    transformers=[
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            NUMERICAL_FEATURES_ABLATION
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
# 7. Transform data
# ============================================================

X_train_full_transformed = preprocessor_full.fit_transform(X_train_full)
X_test_full_transformed = preprocessor_full.transform(X_test_full)

X_train_ablation_transformed = preprocessor_ablation.fit_transform(
    X_train_ablation
)

X_test_ablation_transformed = preprocessor_ablation.transform(
    X_test_ablation
)


print("\nTransformed shapes:")
print("Full model train:", X_train_full_transformed.shape)
print("Full model test :", X_test_full_transformed.shape)
print("Ablation train  :", X_train_ablation_transformed.shape)
print("Ablation test   :", X_test_ablation_transformed.shape)


# ============================================================
# 8. Train FULL Random Forest
# ============================================================

full_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

full_model.fit(
    X_train_full_transformed,
    y_train
)

full_predictions = full_model.predict(
    X_test_full_transformed
)


# ============================================================
# 9. Train ABLATION Random Forest
# ============================================================

ablation_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

ablation_model.fit(
    X_train_ablation_transformed,
    y_train
)

ablation_predictions = ablation_model.predict(
    X_test_ablation_transformed
)


# ============================================================
# 10. Calculate metrics
# ============================================================

full_mae = mean_absolute_error(y_test, full_predictions)
full_rmse = mean_squared_error(
    y_test,
    full_predictions
) ** 0.5
full_r2 = r2_score(y_test, full_predictions)


ablation_mae = mean_absolute_error(
    y_test,
    ablation_predictions
)

ablation_rmse = mean_squared_error(
    y_test,
    ablation_predictions
) ** 0.5

ablation_r2 = r2_score(
    y_test,
    ablation_predictions
)


# ============================================================
# 11. Display results
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ABLATION RESULTS")
print("=" * 60)

print("\nFULL MODEL")
print("-" * 30)
print(f"MAE  : ₹{full_mae:,.2f}")
print(f"RMSE : ₹{full_rmse:,.2f}")
print(f"R²   : {full_r2:.4f}")

print("\nWITHOUT POWER + TORQUE")
print("-" * 30)
print(f"MAE  : ₹{ablation_mae:,.2f}")
print(f"RMSE : ₹{ablation_rmse:,.2f}")
print(f"R²   : {ablation_r2:.4f}")


# ============================================================
# 12. Calculate performance change
# ============================================================

mae_change = ablation_mae - full_mae
rmse_change = ablation_rmse - full_rmse
r2_change = ablation_r2 - full_r2

print("\n" + "=" * 60)
print("PERFORMANCE CHANGE")
print("=" * 60)

print(f"MAE change  : ₹{mae_change:,.2f}")
print(f"RMSE change : ₹{rmse_change:,.2f}")
print(f"R² change   : {r2_change:.4f}")


# ============================================================
# 13. Interpretation
# ============================================================

print("\nINTERPRETATION")

if ablation_mae > full_mae:
    print("Removing power and torque increased MAE.")
    print("Therefore, power and torque provide useful predictive information.")
else:
    print("Removing power and torque did not increase MAE significantly.")

if ablation_r2 < full_r2:
    print("R² decreased after removing power and torque.")
    print("This indicates that these features contribute to model performance.")
else:
    print("R² did not decrease after removing power and torque.")