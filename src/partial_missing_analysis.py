import pandas as pd


# --------------------------------------------------
# Load latest recovered dataset
# --------------------------------------------------

data_path = "data/processed/car_data_model_recovered.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Technical columns
# --------------------------------------------------

technical_columns = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]


# --------------------------------------------------
# Calculate missing values per row
# --------------------------------------------------

missing_per_row = (
    df[technical_columns]
    .isnull()
    .sum(axis=1)
)


# --------------------------------------------------
# Select partially missing rows
# --------------------------------------------------

partial_mask = (
    (missing_per_row > 0)
    & (missing_per_row < 5)
)

partial_df = df.loc[
    partial_mask
].copy()


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 70)
print("PARTIAL MISSING VALUE ANALYSIS")
print("=" * 70)

print("\nNumber of partially missing rows:")
print(len(partial_df))


# --------------------------------------------------
# Missingness pattern
# --------------------------------------------------

print("\nMissingness patterns:")

patterns = (
    partial_df[technical_columns]
    .isnull()
    .astype(int)
    .astype(str)
    .agg("".join, axis=1)
    .value_counts()
)

print(patterns)


# --------------------------------------------------
# Show partially missing records
# --------------------------------------------------

display_columns = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
] + technical_columns


print("\nPartially missing records:")

print(
    partial_df[
        display_columns
    ]
    .sort_values("name")
    .to_string(index=False)
)


# --------------------------------------------------
# Analyze missing torque records
# --------------------------------------------------

print("\n" + "=" * 70)
print("MISSING TORQUE RECORDS")
print("=" * 70)

missing_torque = partial_df[
    partial_df["torque_nm"].isnull()
]

print("\nNumber of records:")
print(len(missing_torque))

print("\nRecords:")

print(
    missing_torque[
        display_columns
    ]
    .to_string(index=False)
)


# --------------------------------------------------
# Analyze missing max power records
# --------------------------------------------------

print("\n" + "=" * 70)
print("MISSING MAX POWER RECORDS")
print("=" * 70)

missing_power = partial_df[
    partial_df["max_power_bhp"].isnull()
]

print("\nNumber of records:")
print(len(missing_power))

print("\nRecords:")

print(
    missing_power[
        display_columns
    ]
    .to_string(index=False)
)