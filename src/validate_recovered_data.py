import pandas as pd


# --------------------------------------------------
# Load recovered dataset
# --------------------------------------------------

data_path = "data/processed/car_data_recovered.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 60)
print("RECOVERED DATASET VALIDATION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# --------------------------------------------------
# Duplicate validation
# --------------------------------------------------

print("\nExact duplicate rows:")

duplicate_count = df.duplicated().sum()

print(duplicate_count)


# --------------------------------------------------
# Missing value validation
# --------------------------------------------------

technical_columns = [
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats"
]

print("\nMissing values by technical column:")

missing_values = df[technical_columns].isnull().sum()

print(missing_values)


print("\nTotal remaining technical missing values:")

print(missing_values.sum())


# --------------------------------------------------
# Dataset size validation
# --------------------------------------------------

expected_rows = 6926

print("\nExpected number of rows:")
print(expected_rows)

print("\nActual number of rows:")
print(len(df))

if len(df) == expected_rows:
    print("PASS: Correct number of rows.")
else:
    print("WARNING: Row count differs from expected.")


# --------------------------------------------------
# Duplicate validation
# --------------------------------------------------

if duplicate_count == 0:
    print("PASS: No exact duplicate rows remain.")
else:
    print("WARNING: Duplicate rows detected.")


# --------------------------------------------------
# Final validation message
# --------------------------------------------------

print("\n" + "=" * 60)
print("RECOVERED DATASET VALIDATION COMPLETED")
print("=" * 60)