import pandas as pd


# --------------------------------------------------
# Load recovered dataset
# --------------------------------------------------

data_path = "data/processed/car_data_recovered.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Technical columns
# --------------------------------------------------

technical_columns = [
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats"
]


print("\n" + "=" * 60)
print("TECHNICAL FEATURE FORMAT ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# Unique values and data types
# --------------------------------------------------

for column in technical_columns:

    print("\n" + "=" * 60)
    print(f"COLUMN: {column}")
    print("=" * 60)

    print("\nData type:")
    print(df[column].dtype)

    print("\nNumber of missing values:")
    print(df[column].isnull().sum())

    print("\nNumber of unique non-missing values:")
    print(df[column].nunique())

    print("\nFirst 20 unique non-missing values:")

    print(
        df[column]
        .dropna()
        .unique()[:20]
    )


# --------------------------------------------------
# Mileage units
# --------------------------------------------------

print("\n" + "=" * 60)
print("MILEAGE UNIT ANALYSIS")
print("=" * 60)

mileage_non_null = df["mileage"].dropna()

print(
    mileage_non_null
    .str.extract(r"([A-Za-z/]+)$")[0]
    .value_counts()
)


# --------------------------------------------------
# Engine units
# --------------------------------------------------

print("\n" + "=" * 60)
print("ENGINE UNIT ANALYSIS")
print("=" * 60)

engine_non_null = df["engine"].dropna()

print(
    engine_non_null
    .str.extract(r"([A-Za-z]+)$")[0]
    .value_counts()
)


# --------------------------------------------------
# Max power units
# --------------------------------------------------

print("\n" + "=" * 60)
print("MAX POWER UNIT ANALYSIS")
print("=" * 60)

power_non_null = df["max_power"].dropna()

print(
    power_non_null
    .str.extract(r"([A-Za-z]+)$")[0]
    .value_counts()
)


# --------------------------------------------------
# Torque format analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("TORQUE FORMAT ANALYSIS")
print("=" * 60)

torque_non_null = df["torque"].dropna()

print("\nTotal non-missing torque values:")
print(len(torque_non_null))

print("\nTorque values containing 'Nm':")
print(
    torque_non_null
    .str.contains("Nm", case=False, na=False)
    .sum()
)

print("\nTorque values containing 'kgm':")
print(
    torque_non_null
    .str.contains("kgm", case=False, na=False)
    .sum()
)

print("\nTorque values containing '@':")
print(
    torque_non_null
    .str.contains("@", regex=False, na=False)
    .sum()
)

print("\nTorque values containing 'at':")
print(
    torque_non_null
    .str.contains("at", case=False, regex=False, na=False)
    .sum()
)


# --------------------------------------------------
# Examples of unusual torque formats
# --------------------------------------------------

print("\n" + "=" * 60)
print("TORQUE FORMAT EXAMPLES")
print("=" * 60)

print("\nFirst 30 unique torque values:")

print(
    torque_non_null
    .unique()[:30]
)