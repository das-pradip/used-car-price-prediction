import pandas as pd

# --------------------------------------------------
# Load parsed dataset
# --------------------------------------------------

data_path = "data/processed/car_data_parsed.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL ANOMALY ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# Numerical columns
# --------------------------------------------------

numerical_columns = [
    "selling_price",
    "km_driven",
    "year",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]


# --------------------------------------------------
# Zero-value analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("ZERO VALUE ANALYSIS")
print("=" * 60)

for column in numerical_columns:

    zero_count = (df[column] == 0).sum()

    print(f"\n{column}:")
    print(f"Zero values: {zero_count}")


# --------------------------------------------------
# Zero mileage investigation
# --------------------------------------------------

print("\n" + "=" * 60)
print("ZERO MILEAGE INVESTIGATION")
print("=" * 60)

zero_mileage = df[
    df["mileage_value"] == 0
]

print("\nRows with zero mileage:")

print(
    zero_mileage[
        [
            "name",
            "year",
            "fuel",
            "mileage",
            "mileage_value",
            "engine_cc",
            "max_power_bhp",
            "torque_nm"
        ]
    ]
    .to_string(index=False)
)


# --------------------------------------------------
# Zero max power investigation
# --------------------------------------------------

print("\n" + "=" * 60)
print("ZERO MAX POWER INVESTIGATION")
print("=" * 60)

zero_power = df[
    df["max_power_bhp"] == 0
]

print("\nRows with zero maximum power:")

print(
    zero_power[
        [
            "name",
            "year",
            "fuel",
            "max_power",
            "max_power_bhp",
            "engine_cc",
            "torque_nm"
        ]
    ]
    .to_string(index=False)
)


# --------------------------------------------------
# High torque investigation
# --------------------------------------------------

print("\n" + "=" * 60)
print("HIGH TORQUE INVESTIGATION")
print("=" * 60)

high_torque = df[
    df["torque_nm"] > 500
]

print("\nRows with torque greater than 500 Nm:")

print(
    high_torque[
        [
            "name",
            "year",
            "fuel",
            "engine",
            "max_power",
            "torque",
            "torque_nm"
        ]
    ]
    .sort_values("torque_nm", ascending=False)
    .to_string(index=False)
)

print("\nNumber of rows with torque > 500 Nm:")
print(len(high_torque))


# --------------------------------------------------
# High engine capacity investigation
# --------------------------------------------------

print("\n" + "=" * 60)
print("HIGH ENGINE CAPACITY INVESTIGATION")
print("=" * 60)

high_engine = df[
    df["engine_cc"] > 3000
]

print("\nRows with engine capacity greater than 3000 CC:")

print(
    high_engine[
        [
            "name",
            "year",
            "fuel",
            "engine",
            "engine_cc",
            "max_power_bhp",
            "torque_nm"
        ]
    ]
    .sort_values("engine_cc", ascending=False)
    .to_string(index=False)
)

print("\nNumber of rows with engine > 3000 CC:")
print(len(high_engine))


# --------------------------------------------------
# High power investigation
# --------------------------------------------------

print("\n" + "=" * 60)
print("HIGH POWER INVESTIGATION")
print("=" * 60)

high_power = df[
    df["max_power_bhp"] > 250
]

print("\nRows with maximum power greater than 250 bhp:")

print(
    high_power[
        [
            "name",
            "year",
            "fuel",
            "engine",
            "max_power",
            "max_power_bhp",
            "torque_nm"
        ]
    ]
    .sort_values("max_power_bhp", ascending=False)
    .to_string(index=False)
)

print("\nNumber of rows with power > 250 bhp:")
print(len(high_power))


# --------------------------------------------------
# Unusual seat counts
# --------------------------------------------------

print("\n" + "=" * 60)
print("SEAT COUNT INVESTIGATION")
print("=" * 60)

unusual_seats = df[
    ~df["seats"].isin([2, 4, 5, 6, 7, 8, 9, 10, 12, 14])
    & df["seats"].notna()
]

print("\nUnusual seat values:")

print(
    unusual_seats[
        [
            "name",
            "year",
            "seats"
        ]
    ]
    .to_string(index=False)
)


# --------------------------------------------------
# Numerical summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(
    df[numerical_columns]
    .describe()
)


# --------------------------------------------------
# Final message
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL ANOMALY ANALYSIS COMPLETED")
print("=" * 60)