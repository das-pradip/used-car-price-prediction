from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path("data/processed/car_data_feature_engineered.csv")

NUMERICAL_FEATURES = [
    "year",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("USED CAR INPUT RANGE ANALYSIS")
print("=" * 80)

print(f"\nDataset shape: {df.shape}")


# ---------------------------------------------------------
# Basic numerical statistics
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("1. NUMERICAL FEATURE STATISTICS")
print("=" * 80)

stats = df[NUMERICAL_FEATURES].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

stats["missing"] = df[NUMERICAL_FEATURES].isna().sum()

print(
    stats[
        [
            "count",
            "missing",
            "mean",
            "std",
            "min",
            "1%",
            "5%",
            "25%",
            "50%",
            "75%",
            "95%",
            "99%",
            "max",
        ]
    ].round(2)
)


# ---------------------------------------------------------
# Detailed ranges
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("2. USER INPUT RANGE SUMMARY")
print("=" * 80)

for column in NUMERICAL_FEATURES:

    series = df[column].dropna()

    print(f"\n{column}")
    print("-" * len(column))

    print(f"Minimum       : {series.min():.2f}")
    print(f"1st percentile: {series.quantile(0.01):.2f}")
    print(f"5th percentile: {series.quantile(0.05):.2f}")
    print(f"Median        : {series.median():.2f}")
    print(f"95th percentile: {series.quantile(0.95):.2f}")
    print(f"99th percentile: {series.quantile(0.99):.2f}")
    print(f"Maximum       : {series.max():.2f}")
    print(f"Missing       : {df[column].isna().sum()}")


# ---------------------------------------------------------
# Extreme observations
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("3. EXTREME / UNUSUAL VALUES")
print("=" * 80)

for column in NUMERICAL_FEATURES:

    series = df[column].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    print(f"\n{column}")
    print(f"IQR lower bound : {lower_bound:.2f}")
    print(f"IQR upper bound : {upper_bound:.2f}")
    print(f"IQR outlier count: {len(outliers)}")

    if len(outliers) > 0:
        print("Largest unusual values:")
        print(
            outliers
            .sort_values(ascending=False)
            .head(10)
            .round(2)
            .to_string()
        )


# ---------------------------------------------------------
# Categorical input analysis
# ---------------------------------------------------------

CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

print("\n" + "=" * 80)
print("4. CATEGORICAL INPUT VALUES")
print("=" * 80)

for column in CATEGORICAL_FEATURES:

    print(f"\n{column}")
    print("-" * len(column))

    values = (
        df[column]
        .value_counts()
        .sort_index()
    )

    print(values.to_string())


# ---------------------------------------------------------
# Year-specific analysis
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("5. YEAR ANALYSIS")
print("=" * 80)

print("\nYear distribution:")

year_counts = (
    df["year"]
    .value_counts()
    .sort_index()
)

print(year_counts.to_string())

print(f"\nOldest vehicle year : {df['year'].min()}")
print(f"Newest vehicle year : {df['year'].max()}")


# ---------------------------------------------------------
# Technical feature missingness
# ---------------------------------------------------------

TECHNICAL_FEATURES = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]

print("\n" + "=" * 80)
print("6. TECHNICAL FEATURE MISSINGNESS")
print("=" * 80)

missing_summary = (
    df[TECHNICAL_FEATURES]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing_summary.to_string())

print(
    f"\nRows with all technical features present: "
    f"{df[TECHNICAL_FEATURES].notna().all(axis=1).sum()}"
)

print(
    f"Rows with at least one technical feature missing: "
    f"{df[TECHNICAL_FEATURES].isna().any(axis=1).sum()}"
)


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)