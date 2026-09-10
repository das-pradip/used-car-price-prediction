import pandas as pd

from sklearn.model_selection import train_test_split

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
# FIT ONLY ON TRAINING DATA
# ============================================================

X_train_transformed = preprocessor.fit_transform(X_train)


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

X_test_transformed = preprocessor.transform(X_test)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("TRAIN-ONLY PREPROCESSING VALIDATION")
print("=" * 70)

print()

print("X_train original shape:")
print(X_train.shape)

print()

print("X_test original shape:")
print(X_test.shape)

print()

print("X_train transformed shape:")
print(X_train_transformed.shape)

print()

print("X_test transformed shape:")
print(X_test_transformed.shape)

print()

print("Missing values in transformed X_train:")
print(pd.DataFrame(X_train_transformed).isnull().sum().sum())

print()

print("Missing values in transformed X_test:")
print(pd.DataFrame(X_test_transformed).isnull().sum().sum())

print()

print("=" * 70)
print("TRAIN-ONLY PREPROCESSING VALIDATION COMPLETE")
print("=" * 70)