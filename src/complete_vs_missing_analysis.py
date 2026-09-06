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
# Identify missingness groups
# --------------------------------------------------

missing_per_row = (
    df[technical_columns]
    .isnull()
    .sum(axis=1)
)


complete_rows = (
    missing_per_row == 0
)

completely_missing_rows = (
    missing_per_row == 5
)


# --------------------------------------------------
# Basic counts
# --------------------------------------------------

print("\n" + "=" * 60)
print("COMPLETE VS COMPLETELY MISSING RECORDS")
print("=" * 60)

print("\nComplete technical records:")
print(complete_rows.sum())

print("\nCompletely missing technical records:")
print(completely_missing_rows.sum())


# --------------------------------------------------
# Numerical comparison
# --------------------------------------------------

numerical_columns = [
    "selling_price",
    "year",
    "km_driven"
]


print("\n" + "=" * 60)
print("NUMERICAL FEATURE COMPARISON")
print("=" * 60)


complete_stats = (
    df.loc[
        complete_rows,
        numerical_columns
    ]
    .describe()
    .T
)

missing_stats = (
    df.loc[
        completely_missing_rows,
        numerical_columns
    ]
    .describe()
    .T
)


comparison = pd.DataFrame({
    "Complete Mean": complete_stats["mean"],
    "Complete Median": complete_stats["50%"],
    "Missing Mean": missing_stats["mean"],
    "Missing Median": missing_stats["50%"]
})


print("\nSummary comparison:")

print(
    comparison.round(2)
)


print("\nMean comparison:")

print(comparison)


# --------------------------------------------------
# Detailed numerical statistics
# --------------------------------------------------

print("\nComplete records - numerical statistics:")

print(
    df.loc[
        complete_rows,
        numerical_columns
    ]
    .describe()
    .round(2)
)


print("\nCompletely missing records - numerical statistics:")

print(
    df.loc[
        completely_missing_rows,
        numerical_columns
    ]
    .describe()
    .round(2)
)


# --------------------------------------------------
# Categorical comparison
# --------------------------------------------------

categorical_columns = [
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


for column in categorical_columns:

    print("\n" + "=" * 60)
    print(f"{column.upper()} COMPARISON")
    print("=" * 60)

    print("\nComplete records:")

    complete_distribution = (
        df.loc[
            complete_rows,
            column
        ]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(complete_distribution)


    print("\nCompletely missing records:")

    missing_distribution = (
        df.loc[
            completely_missing_rows,
            column
        ]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(missing_distribution)