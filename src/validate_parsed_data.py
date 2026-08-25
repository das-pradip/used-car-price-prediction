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
print("PARSED DATASET VALIDATION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# --------------------------------------------------
# Expected columns
# --------------------------------------------------

expected_columns = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats",
    "mileage_value",
    "mileage_unit",
    "engine_cc",
    "max_power_bhp",
    "torque_nm"
]


print("\nChecking expected columns:")

missing_columns = [
    column
    for column in expected_columns
    if column not in df.columns
]

if len(missing_columns) == 0:
    print("PASS: All expected columns are present.")
else:
    print("FAIL: Missing expected columns:")
    print(missing_columns)


# --------------------------------------------------
# Row count validation
# --------------------------------------------------

expected_rows = 6926
actual_rows = len(df)

print("\nExpected number of rows:")
print(expected_rows)

print("\nActual number of rows:")
print(actual_rows)

if actual_rows == expected_rows:
    print("PASS: Correct number of rows.")
else:
    print("FAIL: Incorrect number of rows.")


# --------------------------------------------------
# Duplicate validation
# --------------------------------------------------

duplicate_rows = df.duplicated().sum()

print("\nExact duplicate rows:")
print(duplicate_rows)

if duplicate_rows == 0:
    print("PASS: No exact duplicate rows.")
else:
    print("WARNING: Duplicate rows found.")


# --------------------------------------------------
# Parsed feature data types
# --------------------------------------------------

parsed_columns = [
    "mileage_value",
    "mileage_unit",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]

print("\nParsed feature data types:")
print(df[parsed_columns].dtypes)


# --------------------------------------------------
# Missing values in parsed features
# --------------------------------------------------

print("\nMissing values in parsed features:")

print(
    df[parsed_columns]
    .isnull()
    .sum()
)


# --------------------------------------------------
# Numerical range checks
# --------------------------------------------------

print("\nNumerical range checks:")

print("\nMileage value range:")
print(
    df["mileage_value"]
    .describe()
)

print("\nEngine CC range:")
print(
    df["engine_cc"]
    .describe()
)

print("\nMaximum power range:")
print(
    df["max_power_bhp"]
    .describe()
)

print("\nTorque range:")
print(
    df["torque_nm"]
    .describe()
)

print("\nSeats range:")
print(
    df["seats"]
    .describe()
)


# --------------------------------------------------
# Mileage unit validation
# --------------------------------------------------

print("\nMileage units:")

print(
    df["mileage_unit"]
    .value_counts(dropna=False)
)


# --------------------------------------------------
# Sample parsed features
# --------------------------------------------------

print("\nFirst 10 parsed records:")

print(
    df[
        [
            "name",
            "year",
            "mileage_value",
            "mileage_unit",
            "engine_cc",
            "max_power_bhp",
            "torque_nm",
            "seats"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# --------------------------------------------------
# Final validation
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

validation_passed = True


if actual_rows != expected_rows:
    validation_passed = False


if duplicate_rows != 0:
    validation_passed = False


if len(missing_columns) != 0:
    validation_passed = False


if validation_passed:

    print("\nPASS: PARSED DATASET VALIDATION COMPLETED")

else:

    print("\nFAIL: PARSED DATASET VALIDATION FAILED")