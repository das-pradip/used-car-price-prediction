import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/processed/car_data_feature_engineered.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# DEFINE FEATURES
# ============================================================

numerical_features = [
    "car_age",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]

categorical_features = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


# ============================================================
# NUMERICAL PIPELINE
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# PREPARE X AND y
# ============================================================

X = df[numerical_features + categorical_features]

y = df["selling_price"]


# ============================================================
# FIT PREPROCESSOR
# ============================================================

X_transformed = preprocessor.fit_transform(X)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("PREPROCESSING PIPELINE")
print("=" * 70)

print()

print("Original X shape:")
print(X.shape)

print()

print("Transformed X shape:")
print(X_transformed.shape)

print()

print("Missing values before preprocessing:")
print(X.isnull().sum())

print()

print("Missing values after preprocessing:")

transformed_df = pd.DataFrame(X_transformed)

print(transformed_df.isnull().sum().sum())

print()

print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)