"""
Build and analyze a reliable vehicle specification lookup.

Purpose:
    Investigate whether exact vehicle combinations
    (name + year + fuel + transmission) have consistent
    technical specifications in the dataset.

This script does NOT modify the production dataset.
It only analyzes the evidence available in the dataset.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "car_data_feature_engineered.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("VEHICLE SPECIFICATION LOOKUP ANALYSIS")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 3. COLUMNS USED
# ============================================================

group_columns = [
    "name",
    "year",
    "fuel",
    "transmission",
]

spec_columns = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]


# ============================================================
# 4. BASIC DATA CHECK
# ============================================================

required_columns = group_columns + spec_columns + ["brand"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. CREATE VEHICLE COMBINATION GROUPS
# ============================================================

print("\n" + "=" * 80)
print("1. VEHICLE COMBINATION ANALYSIS")
print("=" * 80)

grouped = (
    df.groupby(group_columns, dropna=False)
    .size()
    .reset_index(name="row_count")
)

print(f"\nUnique name + year + fuel + transmission combinations:")
print(len(grouped))

print("\nTop 20 combinations by number of records:")

print(
    grouped
    .sort_values("row_count", ascending=False)
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 6. SPECIFICATION CONSISTENCY
# ============================================================

print("\n" + "=" * 80)
print("2. SPECIFICATION CONSISTENCY")
print("=" * 80)


def count_unique_non_null(series):
    """Count distinct non-null values."""
    return series.dropna().nunique()


records = []

for keys, group in df.groupby(group_columns, dropna=False):

    name, year, fuel, transmission = keys

    record = {
        "name": name,
        "year": year,
        "fuel": fuel,
        "transmission": transmission,
        "row_count": len(group),
    }

    for column in spec_columns:
        record[f"{column}_unique"] = count_unique_non_null(
            group[column]
        )

    records.append(record)


consistency_df = pd.DataFrame(records)


# ============================================================
# 7. PERFECTLY CONSISTENT COMBINATIONS
# ============================================================

spec_unique_columns = [
    f"{column}_unique"
    for column in spec_columns
]

perfectly_consistent = consistency_df[
    (consistency_df[spec_unique_columns] <= 1).all(axis=1)
].copy()


print("\nTotal combinations:")
print(len(consistency_df))

print("\nPerfectly specification-consistent combinations:")
print(len(perfectly_consistent))

print(
    "\nPercentage of combinations with no specification conflicts:"
)

percentage = (
    len(perfectly_consistent)
    / len(consistency_df)
    * 100
)

print(f"{percentage:.2f}%")


# ============================================================
# 8. CONFLICTING COMBINATIONS
# ============================================================

conflicting = consistency_df[
    (consistency_df[spec_unique_columns] > 1).any(axis=1)
].copy()


print("\nConflicting combinations:")
print(len(conflicting))

print("\nTop conflicting combinations:")

print(
    conflicting
    .sort_values("row_count", ascending=False)
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 9. HIGH-CONFIDENCE LOOKUP CANDIDATES
# ============================================================

print("\n" + "=" * 80)
print("3. HIGH-CONFIDENCE LOOKUP CANDIDATES")
print("=" * 80)

# We require:
#   - at least 2 records
#   - exactly one unique value for every technical specification
#
# This prevents a single observation from being treated as strong
# evidence.

high_confidence = perfectly_consistent[
    perfectly_consistent["row_count"] >= 2
].copy()


print(
    "\nHigh-confidence combinations "
    "(>=2 records + all specifications consistent):"
)

print(len(high_confidence))


print("\nTop 30 high-confidence combinations:")

print(
    high_confidence
    .sort_values("row_count", ascending=False)
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 10. SINGLE-RECORD COMBINATIONS
# ============================================================

single_record = perfectly_consistent[
    perfectly_consistent["row_count"] == 1
].copy()

print("\n" + "=" * 80)
print("4. SINGLE-RECORD COMBINATIONS")
print("=" * 80)

print(
    "\nPerfectly consistent combinations with only one record:"
)

print(len(single_record))


# ============================================================
# 11. BUILD HIGH-CONFIDENCE LOOKUP TABLE
# ============================================================

print("\n" + "=" * 80)
print("5. BUILD HIGH-CONFIDENCE LOOKUP TABLE")
print("=" * 80)

lookup_records = []

for _, row in high_confidence.iterrows():

    mask = (
        (df["name"] == row["name"])
        & (df["year"] == row["year"])
        & (df["fuel"] == row["fuel"])
        & (df["transmission"] == row["transmission"])
    )

    subset = df.loc[mask]

    lookup_record = {
        "brand": subset["brand"].mode().iloc[0],
        "name": row["name"],
        "year": row["year"],
        "fuel": row["fuel"],
        "transmission": row["transmission"],
        "row_count": len(subset),
    }

    for column in spec_columns:

        values = subset[column].dropna()

        if len(values) == 0:
            lookup_record[column] = None
        else:
            lookup_record[column] = values.iloc[0]

    lookup_records.append(lookup_record)


lookup_df = pd.DataFrame(lookup_records)


print("\nLookup table rows:")
print(len(lookup_df))


# ============================================================
# 12. CHECK PARTIAL SPECIFICATION COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("6. SPECIFICATION COVERAGE")
print("=" * 80)

if len(lookup_df) > 0:

    for column in spec_columns:

        available = lookup_df[column].notna().sum()

        percentage = (
            available
            / len(lookup_df)
            * 100
        )

        print(
            f"{column:20s}: "
            f"{available:5d}/{len(lookup_df):5d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# 13. EXAMPLE LOOKUP RECORDS
# ============================================================

print("\n" + "=" * 80)
print("7. EXAMPLE LOOKUP RECORDS")
print("=" * 80)

if len(lookup_df) > 0:

    example_columns = [
        "brand",
        "name",
        "year",
        "fuel",
        "transmission",
        "mileage_value",
        "engine_cc",
        "max_power_bhp",
        "torque_nm",
        "seats",
        "row_count",
    ]

    print(
        lookup_df[
            example_columns
        ]
        .sort_values(
            "row_count",
            ascending=False
        )
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# 14. SPECIFIC EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("8. SPECIFIC VEHICLE EXAMPLES")
print("=" * 80)


examples = [
    "Maruti Swift Dzire VDI",
    "Maruti Swift VDI BSIV",
    "Toyota Etios GD",
    "Toyota Etios Liva GD",
    "Volvo S60 D4 SUMMUM",
    "Volvo XC90 T8 Excellence BSIV",
]


for vehicle_name in examples:

    matches = lookup_df[
        lookup_df["name"] == vehicle_name
    ]

    print(f"\n{vehicle_name}")

    if matches.empty:

        print("  No high-confidence lookup entry found.")

    else:

        print(
            matches[
                [
                    "brand",
                    "name",
                    "year",
                    "fuel",
                    "transmission",
                    "mileage_value",
                    "engine_cc",
                    "max_power_bhp",
                    "torque_nm",
                    "seats",
                    "row_count",
                ]
            ].to_string(index=False)
        )


# ============================================================
# 15. SAVE LOOKUP TABLE
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vehicle_spec_lookup.csv"
)

lookup_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nLookup file saved to:")
print(OUTPUT_FILE)