"""
Evaluate coverage of the high-confidence vehicle specification lookup.

The lookup is based on:
    name + year + fuel + transmission

A vehicle combination is considered covered when it exists
in the high-confidence lookup table.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "car_data_feature_engineered.csv"
)

LOOKUP_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vehicle_spec_lookup.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("VEHICLE SPECIFICATION LOOKUP COVERAGE EVALUATION")
print("=" * 80)

df = pd.read_csv(DATA_FILE)
lookup = pd.read_csv(LOOKUP_FILE)

print(f"\nDataset rows : {len(df)}")
print(f"Lookup rows  : {len(lookup)}")


# ============================================================
# 3. LOOKUP KEYS
# ============================================================

KEY_COLUMNS = [
    "name",
    "year",
    "fuel",
    "transmission",
]


# ============================================================
# 4. NORMALIZE KEY TYPES
# ============================================================

for column in KEY_COLUMNS:

    df[column] = df[column].astype(str)
    lookup[column] = lookup[column].astype(str)


# ============================================================
# 5. CREATE COVERAGE KEY
# ============================================================

def create_key(dataframe):
    return (
        dataframe["name"]
        + "|||"
        + dataframe["year"]
        + "|||"
        + dataframe["fuel"]
        + "|||"
        + dataframe["transmission"]
    )


df["lookup_key"] = create_key(df)
lookup["lookup_key"] = create_key(lookup)


# ============================================================
# 6. ROW-LEVEL COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("1. ROW-LEVEL COVERAGE")
print("=" * 80)

covered_keys = set(lookup["lookup_key"])

df["lookup_available"] = df["lookup_key"].isin(
    covered_keys
)

covered_rows = df["lookup_available"].sum()
uncovered_rows = len(df) - covered_rows

coverage_percentage = (
    covered_rows
    / len(df)
    * 100
)

print(f"\nCovered rows   : {covered_rows}")
print(f"Uncovered rows : {uncovered_rows}")
print(f"Coverage      : {coverage_percentage:.2f}%")


# ============================================================
# 7. UNIQUE COMBINATION COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("2. UNIQUE COMBINATION COVERAGE")
print("=" * 80)

unique_combinations = df[
    KEY_COLUMNS
].drop_duplicates()

unique_combinations["lookup_key"] = create_key(
    unique_combinations
)

unique_combinations["lookup_available"] = (
    unique_combinations["lookup_key"].isin(
        covered_keys
    )
)

unique_total = len(unique_combinations)

unique_covered = (
    unique_combinations["lookup_available"].sum()
)

unique_uncovered = (
    unique_total - unique_covered
)

unique_coverage = (
    unique_covered
    / unique_total
    * 100
)

print(f"\nUnique combinations : {unique_total}")
print(f"Covered             : {unique_covered}")
print(f"Uncovered           : {unique_uncovered}")
print(f"Coverage            : {unique_coverage:.2f}%")


# ============================================================
# 8. COVERAGE BY BRAND
# ============================================================

print("\n" + "=" * 80)
print("3. COVERAGE BY BRAND")
print("=" * 80)

brand_coverage = (
    df.groupby("brand")
    .agg(
        total_rows=("lookup_available", "size"),
        covered_rows=("lookup_available", "sum"),
    )
    .reset_index()
)

brand_coverage["coverage_percentage"] = (
    brand_coverage["covered_rows"]
    / brand_coverage["total_rows"]
    * 100
)

brand_coverage = brand_coverage.sort_values(
    "total_rows",
    ascending=False,
)

print(
    brand_coverage.to_string(
        index=False,
        formatters={
            "coverage_percentage": "{:.2f}".format
        },
    )
)


# ============================================================
# 9. COVERAGE BY YEAR
# ============================================================

print("\n" + "=" * 80)
print("4. COVERAGE BY YEAR")
print("=" * 80)

year_coverage = (
    df.groupby("year")
    .agg(
        total_rows=("lookup_available", "size"),
        covered_rows=("lookup_available", "sum"),
    )
    .reset_index()
)

year_coverage["coverage_percentage"] = (
    year_coverage["covered_rows"]
    / year_coverage["total_rows"]
    * 100
)

print(
    year_coverage.to_string(
        index=False,
        formatters={
            "coverage_percentage": "{:.2f}".format
        },
    )
)


# ============================================================
# 10. COVERAGE BY FUEL
# ============================================================

print("\n" + "=" * 80)
print("5. COVERAGE BY FUEL")
print("=" * 80)

fuel_coverage = (
    df.groupby("fuel")
    .agg(
        total_rows=("lookup_available", "size"),
        covered_rows=("lookup_available", "sum"),
    )
    .reset_index()
)

fuel_coverage["coverage_percentage"] = (
    fuel_coverage["covered_rows"]
    / fuel_coverage["total_rows"]
    * 100
)

print(
    fuel_coverage.to_string(
        index=False,
        formatters={
            "coverage_percentage": "{:.2f}".format
        },
    )
)


# ============================================================
# 11. COVERAGE BY TRANSMISSION
# ============================================================

print("\n" + "=" * 80)
print("6. COVERAGE BY TRANSMISSION")
print("=" * 80)

transmission_coverage = (
    df.groupby("transmission")
    .agg(
        total_rows=("lookup_available", "size"),
        covered_rows=("lookup_available", "sum"),
    )
    .reset_index()
)

transmission_coverage["coverage_percentage"] = (
    transmission_coverage["covered_rows"]
    / transmission_coverage["total_rows"]
    * 100
)

print(
    transmission_coverage.to_string(
        index=False,
        formatters={
            "coverage_percentage": "{:.2f}".format
        },
    )
)


# ============================================================
# 12. MOST COMMON UNCOVERED COMBINATIONS
# ============================================================

print("\n" + "=" * 80)
print("7. MOST COMMON UNCOVERED COMBINATIONS")
print("=" * 80)

uncovered = df[
    ~df["lookup_available"]
].copy()

uncovered_combinations = (
    uncovered
    .groupby(KEY_COLUMNS)
    .size()
    .reset_index(name="row_count")
    .sort_values(
        "row_count",
        ascending=False,
    )
)

print(
    uncovered_combinations
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 13. COMMON COVERED COMBINATIONS
# ============================================================

print("\n" + "=" * 80)
print("8. MOST COMMON COVERED COMBINATIONS")
print("=" * 80)

covered = df[
    df["lookup_available"]
].copy()

covered_combinations = (
    covered
    .groupby(KEY_COLUMNS)
    .size()
    .reset_index(name="row_count")
    .sort_values(
        "row_count",
        ascending=False,
    )
)

print(
    covered_combinations
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("9. FINAL SUMMARY")
print("=" * 80)

print(
    f"""
Dataset rows:
    {len(df)}

Unique vehicle combinations:
    {unique_total}

High-confidence lookup combinations:
    {len(lookup)}

Row-level lookup coverage:
    {coverage_percentage:.2f}%

Unique-combination coverage:
    {unique_coverage:.2f}%

Covered rows:
    {covered_rows}

Uncovered rows:
    {uncovered_rows}
"""
)

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)