import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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
print("MODEL ERROR ANALYSIS")
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
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]),
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
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
# 5. Transform training and test data
# ============================================================

X_train_transformed = preprocessor.fit_transform(
    X_train
)

X_test_transformed = preprocessor.transform(
    X_test
)


print("\nTransformed shapes:")
print("Train:", X_train_transformed.shape)
print("Test :", X_test_transformed.shape)


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
# 7. Generate predictions
# ============================================================

y_pred = model.predict(
    X_test_transformed
)


# ============================================================
# 8. Evaluate model
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


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 9. Create error-analysis DataFrame
# ============================================================

error_df = X_test.copy()

error_df["actual_price"] = y_test.values
error_df["predicted_price"] = y_pred

error_df["error"] = (
    error_df["predicted_price"]
    - error_df["actual_price"]
)

error_df["absolute_error"] = (
    error_df["error"].abs()
)

# Avoid division by zero
error_df["percentage_error"] = (
    error_df["absolute_error"]
    / error_df["actual_price"]
    * 100
)


# ============================================================
# 10. Sort by largest absolute error
# ============================================================

largest_errors = error_df.sort_values(
    by="absolute_error",
    ascending=False
)


# ============================================================
# 11. Display largest prediction errors
# ============================================================

print("\n" + "=" * 70)
print("TOP 15 LARGEST PREDICTION ERRORS")
print("=" * 70)

columns_to_display = [
    "actual_price",
    "predicted_price",
    "error",
    "absolute_error",
    "percentage_error",
    "car_age",
    "km_driven",
    "brand",
    "fuel",
    "transmission",
]

print(
    largest_errors[
        columns_to_display
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 12. Save error analysis
# ============================================================

OUTPUT_PATH = (
    "data/processed/"
    "random_forest_error_analysis.csv"
)

error_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nError analysis saved to:")
print(OUTPUT_PATH)

# ============================================================
# 13. Error analysis by price range
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS BY PRICE RANGE")
print("=" * 70)


# Define price ranges
price_bins = [
    0,
    200000,
    500000,
    1000000,
    2000000,
    5000000,
    float("inf")
]

price_labels = [
    "< ₹2L",
    "₹2L–₹5L",
    "₹5L–₹10L",
    "₹10L–₹20L",
    "₹20L–₹50L",
    "> ₹50L"
]


# Assign each car to a price range
error_df["price_range"] = pd.cut(
    error_df["actual_price"],
    bins=price_bins,
    labels=price_labels,
    right=False
)


# Calculate error metrics by price range
price_range_analysis = (
    error_df
    .groupby("price_range", observed=False)
    .agg(
        car_count=("actual_price", "count"),
        mae=("absolute_error", "mean"),
        rmse=("error", lambda x: (x.pow(2).mean()) ** 0.5),
        mean_absolute_percentage_error=("percentage_error", "mean"),
        median_percentage_error=("percentage_error", "median")
    )
    .reset_index()
)


# Display results
print(
    price_range_analysis.to_string(index=False)
)


# Save results
PRICE_RANGE_OUTPUT = (
    "data/processed/"
    "random_forest_error_by_price_range.csv"
)

price_range_analysis.to_csv(
    PRICE_RANGE_OUTPUT,
    index=False
)

print("\nPrice-range analysis saved to:")
print(PRICE_RANGE_OUTPUT)





# ============================================================
# 14. Error analysis by car age
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS BY CAR AGE")
print("=" * 70)


# Define car-age ranges
age_bins = [
    -1,
    3,
    7,
    12,
    17,
    float("inf")
]

age_labels = [
    "0–3 years",
    "4–7 years",
    "8–12 years",
    "13–17 years",
    "18+ years"
]


# Assign each car to an age range
error_df["age_range"] = pd.cut(
    error_df["car_age"],
    bins=age_bins,
    labels=age_labels,
    right=True
)


# Calculate error metrics by car age
age_analysis = (
    error_df
    .groupby("age_range", observed=False)
    .agg(
        car_count=("actual_price", "count"),
        mae=("absolute_error", "mean"),
        rmse=("error", lambda x: (x.pow(2).mean()) ** 0.5),
        mean_absolute_percentage_error=("percentage_error", "mean"),
        median_absolute_percentage_error=("percentage_error", "median")
    )
    .reset_index()
)


# Display results
print(
    age_analysis.to_string(index=False)
)


# Save results
AGE_OUTPUT = (
    "data/processed/"
    "random_forest_error_by_car_age.csv"
)

age_analysis.to_csv(
    AGE_OUTPUT,
    index=False
)

print("\nCar-age analysis saved to:")
print(AGE_OUTPUT)


# ============================================================
# 15. Prediction bias analysis by price range
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION BIAS ANALYSIS BY PRICE RANGE")
print("=" * 70)


# Create signed error categories
error_df["prediction_direction"] = error_df["error"].apply(
    lambda x: "Overprediction" if x > 0 else "Underprediction"
)


# Analyze prediction direction by price range
bias_analysis = (
    error_df
    .groupby("price_range", observed=False)
    .agg(
        car_count=("actual_price", "count"),
        mean_signed_error=("error", "mean"),
        median_signed_error=("error", "median"),
        overprediction_count=(
            "prediction_direction",
            lambda x: (x == "Overprediction").sum()
        ),
        underprediction_count=(
            "prediction_direction",
            lambda x: (x == "Underprediction").sum()
        )
    )
    .reset_index()
)


# Calculate percentages
bias_analysis["overprediction_percentage"] = (
    bias_analysis["overprediction_count"]
    / bias_analysis["car_count"]
    * 100
)

bias_analysis["underprediction_percentage"] = (
    bias_analysis["underprediction_count"]
    / bias_analysis["car_count"]
    * 100
)


# Display results
print(
    bias_analysis.to_string(index=False)
)


# Save results
BIAS_OUTPUT = (
    "data/processed/"
    "random_forest_prediction_bias_by_price_range.csv"
)

bias_analysis.to_csv(
    BIAS_OUTPUT,
    index=False
)

print("\nPrediction-bias analysis saved to:")
print(BIAS_OUTPUT)

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)