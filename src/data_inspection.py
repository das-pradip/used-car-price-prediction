import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

data_path = "data/raw/Car details v3.csv"

df = pd.read_csv(data_path)


# ============================================================
# 2. BASIC DATA INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())


print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)


print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

print(df.columns.tolist())


print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


print("\n" + "=" * 60)
print("MISSING VALUE PERCENTAGE")
print("=" * 60)

missing_percentage = (
    df.isnull().sum() / len(df) * 100
).round(2)

print(missing_percentage)


print("\n" + "=" * 60)
print("NUMBER OF MISSING VALUES PER ROW")
print("=" * 60)

print(
    df.isnull()
      .sum(axis=1)
      .value_counts()
      .sort_index()
)


print("\n" + "=" * 60)
print("ROWS CONTAINING MISSING VALUES")
print("=" * 60)

print(
    df[df.isnull().any(axis=1)]
    .head(10)
)


# ============================================================
# 4. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Total duplicate rows: {duplicate_count}")


print("\n" + "=" * 60)
print("MOST REPEATED RECORDS")
print("=" * 60)

duplicate_counts = (
    df.value_counts()
      .head(10)
)

print(duplicate_counts)


# ============================================================
# 5. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ============================================================
# 6. SKEWNESS ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SKEWNESS")
print("=" * 60)

numerical_columns = [
    "year",
    "selling_price",
    "km_driven",
    "seats"
]

print(df[numerical_columns].skew())


# ============================================================
# 7. SELLING PRICE PERCENTILE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SELLING PRICE PERCENTILES")
print("=" * 60)

price_percentiles = df["selling_price"].quantile(
    [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.995,
        1.00
    ]
)

print(price_percentiles)


# ============================================================
# 8. SELLING PRICE RANGE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SELLING PRICE RANGE COUNTS")
print("=" * 60)

price_ranges = {
    "Below ₹2 lakh": df["selling_price"] < 200000,

    "₹2–5 lakh": (
        (df["selling_price"] >= 200000)
        & (df["selling_price"] < 500000)
    ),

    "₹5–10 lakh": (
        (df["selling_price"] >= 500000)
        & (df["selling_price"] < 1000000)
    ),

    "₹10–20 lakh": (
        (df["selling_price"] >= 1000000)
        & (df["selling_price"] < 2000000)
    ),

    "₹20–50 lakh": (
        (df["selling_price"] >= 2000000)
        & (df["selling_price"] <= 5000000)
    ),

    "Above ₹50 lakh": df["selling_price"] > 5000000
}


for label, condition in price_ranges.items():

    count = condition.sum()

    percentage = count / len(df) * 100

    print(
        f"{label}: "
        f"{count} cars "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 9. SELLING PRICE IQR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("IQR ANALYSIS FOR SELLING PRICE")
print("=" * 60)

Q1 = df["selling_price"].quantile(0.25)

Q3 = df["selling_price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


print(f"Q1: ₹{Q1:,.0f}")

print(f"Q3: ₹{Q3:,.0f}")

print(f"IQR: ₹{IQR:,.0f}")

print(f"Lower bound: ₹{lower_bound:,.0f}")

print(f"Upper bound: ₹{upper_bound:,.0f}")


# ============================================================
# 10. POTENTIAL IQR OUTLIERS
# ============================================================

potential_outliers = df[
    (df["selling_price"] < lower_bound)
    |
    (df["selling_price"] > upper_bound)
]


outlier_count = len(potential_outliers)

outlier_percentage = (
    outlier_count / len(df) * 100
)


print("\nPotential IQR outliers:")

print(
    f"{outlier_count} cars "
    f"({outlier_percentage:.2f}% of dataset)"
)


# ============================================================
# 11. SELLING PRICE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("SELLING PRICE DISTRIBUTION")
print("=" * 60)

# ============================================================
# 12. INVESTIGATE HIGH-PRICE OUTLIERS
# ============================================================

print("\n" + "=" * 60)
print("HIGHEST-PRICED CARS")
print("=" * 60)

highest_price_cars = (
    df.sort_values(
        by="selling_price",
        ascending=False
    )
    [
        [
            "name",
            "year",
            "selling_price",
            "km_driven",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "engine",
            "max_power",
            "seats"
        ]
    ]
    .head(30)
)

print(highest_price_cars.to_string(index=False))

# ============================================================
# 13. UNIQUE HIGH-PRICE CARS
# ============================================================

print("\n" + "=" * 60)
print("UNIQUE HIGH-PRICE CAR RECORDS")
print("=" * 60)

unique_high_price_cars = (
    df.sort_values(
        by="selling_price",
        ascending=False
    )
    [
        [
            "name",
            "year",
            "selling_price",
            "km_driven",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "engine",
            "max_power",
            "seats"
        ]
    ]
    .drop_duplicates()
    .head(30)
)

print(
    unique_high_price_cars.to_string(index=False)
)

# --------------------------------------------------
# Duplicate Record Investigation
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE RECORD INVESTIGATION")
print("=" * 60)

total_rows = len(df)
duplicate_rows = df.duplicated().sum()
unique_rows = df.drop_duplicates().shape[0]

print(f"\nTotal rows: {total_rows}")
print(f"Exact duplicate rows: {duplicate_rows}")
print(f"Unique rows: {unique_rows}")

duplicate_percentage = duplicate_rows / total_rows * 100

print(f"Duplicate percentage: {duplicate_percentage:.2f}%")

print("\nMost frequently repeated exact records:")

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nMost frequently repeated exact records:")

duplicate_frequency = (
    df.value_counts(dropna=False)
      .reset_index(name="frequency")
)

print(
    duplicate_frequency
    .head(20)
    .to_string(index=False)
)

rows_after_duplicate_removal = len(df.drop_duplicates())

rows_removed = len(df) - rows_after_duplicate_removal

print(f"\nRows after removing exact duplicates: {rows_after_duplicate_removal}")
print(f"Rows that would be removed: {rows_removed}")

print(
    f"Percentage of rows that would be removed: "
    f"{rows_removed / len(df) * 100:.2f}%"
)

# --------------------------------------------------
# Duplicate Distribution Analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE DISTRIBUTION ANALYSIS")
print("=" * 60)

duplicate_mask = df.duplicated(keep=False)

duplicate_rows = df[duplicate_mask].copy()

print("\nTotal rows involved in duplicate groups:")
print(len(duplicate_rows))

print("\nDuplicate rows by year:")
print(
    duplicate_rows["year"]
    .value_counts()
    .sort_index()
)

print("\nDuplicate rows by fuel type:")
print(
    duplicate_rows["fuel"]
    .value_counts()
)

print("\nDuplicate rows by seller type:")
print(
    duplicate_rows["seller_type"]
    .value_counts()
)

print("\nDuplicate rows by transmission:")
print(
    duplicate_rows["transmission"]
    .value_counts()
)

# --------------------------------------------------
# Duplicate Frequency Distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE FREQUENCY DISTRIBUTION")
print("=" * 60)

duplicate_frequency_counts = (
    df.value_counts()
      .value_counts()
      .sort_index()
)

print("\nFrequency of exact duplicate records:")
print(duplicate_frequency_counts)

print("\nInterpretation:")
print(
    "This shows how many unique records occur 1 time, 2 times, "
    "3 times, etc. in the dataset."
)

print("\nMaximum repetition of a single exact record:")
print(df.value_counts().max())

# --------------------------------------------------
# Duplicate Count Verification
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE COUNT VERIFICATION")
print("=" * 60)

duplicate_frequency = df.value_counts(dropna=False)

calculated_duplicate_rows = (
    (duplicate_frequency - 1)
    .clip(lower=0)
    .sum()
)

print("\nDuplicate rows from df.duplicated():")
print(df.duplicated().sum())

print("\nDuplicate rows calculated from frequency table:")
print(calculated_duplicate_rows)

print("\nDifference:")
print(df.duplicated().sum() - calculated_duplicate_rows)

# --------------------------------------------------
# Investigate Duplicate Rows with Missing Values
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATES WITH MISSING VALUES")
print("=" * 60)

duplicate_rows = df[df.duplicated(keep=False)]

duplicates_with_missing = duplicate_rows[
    duplicate_rows.isnull().any(axis=1)
]

print("\nTotal duplicate rows:")
print(len(duplicate_rows))

print("\nDuplicate rows containing at least one missing value:")
print(len(duplicates_with_missing))

print("\nMissing values within duplicate rows:")
print(duplicates_with_missing.isnull().sum())

# --------------------------------------------------
# Duplicate Impact Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE IMPACT SUMMARY")
print("=" * 60)

total_rows = len(df)

unique_rows = df.drop_duplicates().shape[0]

duplicate_rows = total_rows - unique_rows

duplicate_percentage = (
    duplicate_rows / total_rows * 100
)

duplicate_groups = (
    df.value_counts(dropna=False)
    .loc[lambda x: x > 1]
)

print("\nTotal rows:")
print(total_rows)

print("\nUnique rows:")
print(unique_rows)

print("\nDuplicate rows that would be removed:")
print(duplicate_rows)

print("\nPercentage of rows that are duplicates:")
print(f"{duplicate_percentage:.2f}%")

print("\nNumber of duplicate groups:")
print(len(duplicate_groups))

print("\nMaximum repetition of one exact record:")
print(duplicate_groups.max())

# --------------------------------------------------
# Remove Exact Duplicate Rows
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE REMOVAL")
print("=" * 60)

rows_before = len(df)

df_clean = df.drop_duplicates().copy()

rows_after = len(df_clean)

rows_removed = rows_before - rows_after

print("\nRows before duplicate removal:")
print(rows_before)

print("\nRows after duplicate removal:")
print(rows_after)

print("\nRows removed:")
print(rows_removed)

print("\nPercentage of rows removed:")
print(f"{rows_removed / rows_before * 100:.2f}%")

# Save cleaned dataset
processed_path = "data/processed/car_data_deduplicated.csv"

df_clean.to_csv(processed_path, index=False)

print("\nCleaned dataset saved to:")
print(processed_path)

