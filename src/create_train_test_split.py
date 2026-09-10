import pandas as pd

from sklearn.model_selection import train_test_split


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
# Features
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

categorical_features = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


feature_columns = (
    numerical_features
    + categorical_features
)


# --------------------------------------------------
# Create X and y
# --------------------------------------------------

X = df[feature_columns].copy()

y = df[target_column].copy()


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# Display split information
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)


print("\nOriginal dataset:")
print(len(df))


print("\nTraining samples:")
print(len(X_train))


print("\nTesting samples:")
print(len(X_test))


print("\nTraining percentage:")
print(
    round(len(X_train) / len(df) * 100, 2),
    "%"
)


print("\nTesting percentage:")
print(
    round(len(X_test) / len(df) * 100, 2),
    "%"
)


# --------------------------------------------------
# Verify shapes
# --------------------------------------------------

print("\n" + "=" * 70)
print("SHAPES")
print("=" * 70)


print("\nX_train:")
print(X_train.shape)


print("\nX_test:")
print(X_test.shape)


print("\ny_train:")
print(y_train.shape)


print("\ny_test:")
print(y_test.shape)


# --------------------------------------------------
# Verify target distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)


print("\nTraining target statistics:")

print(
    y_train
    .describe()
    .round(2)
)


print("\nTesting target statistics:")

print(
    y_test
    .describe()
    .round(2)
)


# --------------------------------------------------
# Verify missing values
# --------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)


print("\nTraining feature missing values:")

print(
    X_train
    .isnull()
    .sum()
)


print("\nTesting feature missing values:")

print(
    X_test
    .isnull()
    .sum()
)


# --------------------------------------------------
# Final confirmation
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT COMPLETE")
print("=" * 70)

print(
    "\nReady for preprocessing pipeline."
)