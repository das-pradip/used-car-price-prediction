import pandas as pd


# --------------------------------------------------
# File Paths
# --------------------------------------------------

processed_path = "data/processed/car_data_deduplicated.csv"


# --------------------------------------------------
# Load Processed Dataset
# --------------------------------------------------

df = pd.read_csv(processed_path)


# --------------------------------------------------
# Basic Dataset Validation
# --------------------------------------------------

print("\n" + "=" * 60)
print("PROCESSED DATASET VALIDATION")
print("=" * 60)


print("\nDataset shape:")
print(df.shape)


print("\nColumn names:")
print(df.columns.tolist())


print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# Data Types
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# Missing Values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


print("\nMissing value percentage:")
print(
    (df.isnull().sum() / len(df) * 100).round(2)
)


# --------------------------------------------------
# Duplicate Validation
# --------------------------------------------------

print("\nExact duplicate rows:")
print(df.duplicated().sum())


# --------------------------------------------------
# Numerical Summary
# --------------------------------------------------

print("\nNumerical summary:")
print(df.describe())


# --------------------------------------------------
# Validation Checks
# --------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)


print("\nExpected number of rows: 6926")
print(f"Actual number of rows: {len(df)}")


if len(df) == 6926:
    print("PASS: Correct number of rows.")
else:
    print("WARNING: Row count does not match expected value.")


print("\nExpected duplicate rows: 0")
print(f"Actual duplicate rows: {df.duplicated().sum()}")


if df.duplicated().sum() == 0:
    print("PASS: No exact duplicate rows remain.")
else:
    print("WARNING: Duplicate rows still exist.")


print("\nProcessed dataset validation completed.")