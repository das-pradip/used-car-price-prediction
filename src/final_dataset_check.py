import pandas as pd


# --------------------------------------------------
# Load latest recovered dataset
# --------------------------------------------------

data_path = "data/processed/car_data_model_recovered.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Basic dataset information
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DATASET CHECK")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# Columns
# --------------------------------------------------

print("\nColumns:")

for column in df.columns:
    print("-", column)


# --------------------------------------------------
# Data types
# --------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# --------------------------------------------------
# Target check
# --------------------------------------------------

print("\n" + "=" * 70)
print("TARGET CHECK")
print("=" * 70)

print("\nSelling price missing values:")
print(df["selling_price"].isnull().sum())

print("\nSelling price statistics:")
print(
    df["selling_price"]
    .describe()
    .round(2)
)


# --------------------------------------------------
# Duplicate check
# --------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

print("\nExact duplicate rows:")
print(df.duplicated().sum())


# --------------------------------------------------
# Categorical missing values
# --------------------------------------------------

categorical_columns = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


print("\n" + "=" * 70)
print("CATEGORICAL MISSING VALUES")
print("=" * 70)

print(
    df[categorical_columns]
    .isnull()
    .sum()
)


# --------------------------------------------------
# Numerical missing values
# --------------------------------------------------

numerical_columns = [
    "year",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]


print("\n" + "=" * 70)
print("NUMERICAL MISSING VALUES")
print("=" * 70)

print(
    df[numerical_columns]
    .isnull()
    .sum()
)


# --------------------------------------------------
# Recovery audit
# --------------------------------------------------

if "model_recovery_applied" in df.columns:

    print("\n" + "=" * 70)
    print("MODEL RECOVERY AUDIT")
    print("=" * 70)

    print(
        df["model_recovery_applied"]
        .value_counts()
    )


# --------------------------------------------------
# Unique values
# --------------------------------------------------

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in categorical_columns:

    print(
        f"{column}: "
        f"{df[column].nunique()} unique values"
    )