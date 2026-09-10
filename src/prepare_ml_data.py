import pandas as pd


# --------------------------------------------------
# Load engineered dataset
# --------------------------------------------------

data_path = (
    "data/processed/"
    "car_data_feature_engineered.csv"
)

df = pd.read_csv(data_path)


# --------------------------------------------------
# Target
# --------------------------------------------------

target_column = "selling_price"


# --------------------------------------------------
# Numerical features
# --------------------------------------------------

numerical_features = [
    "car_age",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]


# --------------------------------------------------
# Categorical features
# --------------------------------------------------

categorical_features = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


# --------------------------------------------------
# Create X and y
# --------------------------------------------------

X = df[
    numerical_features
    + categorical_features
].copy()

y = df[target_column].copy()


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 70)
print("ML DATA PREPARATION")
print("=" * 70)


print("\nDataset shape:")
print(df.shape)


print("\nFeature matrix shape:")
print(X.shape)


print("\nTarget shape:")
print(y.shape)


# --------------------------------------------------
# Feature list
# --------------------------------------------------

print("\n" + "=" * 70)
print("FEATURES")
print("=" * 70)


print("\nNumerical features:")

for feature in numerical_features:
    print("-", feature)


print("\nCategorical features:")

for feature in categorical_features:
    print("-", feature)


print("\nTarget:")
print("-", target_column)


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES IN FEATURES")
print("=" * 70)


print(
    X.isnull()
    .sum()
)


# --------------------------------------------------
# Target validation
# --------------------------------------------------

print("\n" + "=" * 70)
print("TARGET VALIDATION")
print("=" * 70)


print("\nMissing target values:")
print(y.isnull().sum())


print("\nTarget data type:")
print(y.dtype)


print("\nTarget minimum:")
print(y.min())


print("\nTarget maximum:")
print(y.max())


# --------------------------------------------------
# Categorical cardinality
# --------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORICAL CARDINALITY")
print("=" * 70)


for feature in categorical_features:

    print(
        f"{feature}: "
        f"{X[feature].nunique()} unique values"
    )


# --------------------------------------------------
# Final confirmation
# --------------------------------------------------

print("\n" + "=" * 70)
print("ML DATA PREPARATION COMPLETE")
print("=" * 70)

print(
    "\nReady for train/test split and preprocessing pipeline."
)