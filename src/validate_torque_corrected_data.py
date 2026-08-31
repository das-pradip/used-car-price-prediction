import pandas as pd


# ============================================================
# LOAD CORRECTED DATASET
# ============================================================

input_path = "data/processed/car_data_torque_corrected.csv"

df = pd.read_csv(input_path)


print("\n" + "=" * 60)
print("TORQUE CORRECTED DATASET VALIDATION")
print("=" * 60)


# ============================================================
# BASIC DATASET CHECKS
# ============================================================

print("\nDataset shape:")
print(df.shape)


expected_rows = 6926

print("\nExpected rows:")
print(expected_rows)

print("\nActual rows:")
print(len(df))


if len(df) == expected_rows:
    print("PASS: Correct number of rows.")
else:
    print("FAIL: Incorrect number of rows.")


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nExact duplicate rows:")
print(duplicate_count)


if duplicate_count == 0:
    print("PASS: No exact duplicate rows.")
else:
    print("FAIL: Duplicate rows detected.")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
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
    "torque_nm",
    "torque_nm_before_correction",
]


print("\nChecking required columns:")

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if not missing_columns:

    print("PASS: All required columns are present.")

else:

    print("FAIL: Missing columns:")
    print(missing_columns)


# ============================================================
# TORQUE MISSING VALUES
# ============================================================

print("\nTorque missing values:")

torque_missing = df["torque_nm"].isna().sum()

print(torque_missing)


expected_torque_missing = 219

print("\nExpected torque missing values:")
print(expected_torque_missing)


if torque_missing == expected_torque_missing:

    print(
        "PASS: Correct torque missing-value count."
    )

else:

    print(
        "FAIL: Incorrect torque missing-value count."
    )


# ============================================================
# HIGH-CONFIDENCE RECOVERY VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("HIGH-CONFIDENCE RECOVERY VALIDATION")
print("=" * 60)


recovery_checks = {
    "110@ 3,000(kgm@ rpm)": 110.0,
    "190@ 21,800(kgm@ rpm)": 190.0,
}


for torque_text, expected_value in recovery_checks.items():

    rows = df[df["torque"] == torque_text]

    if len(rows) == 0:

        print(
            f"WARNING: {torque_text} not found."
        )

        continue

    actual_values = rows["torque_nm"].dropna().unique()

    print(
        f"\n{torque_text}"
    )

    print(
        "Expected torque_nm:",
        expected_value
    )

    print(
        "Actual torque_nm:",
        actual_values
    )

    if (
        len(actual_values) == 1
        and actual_values[0] == expected_value
    ):

        print("PASS: Recovery is correct.")

    else:

        print("FAIL: Recovery is incorrect.")


# ============================================================
# SUSPICIOUS TORQUE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("SUSPICIOUS TORQUE VALIDATION")
print("=" * 60)


suspicious_values = [
    "145@ 4,100(kgm@ rpm)",
    "115@ 2500(kgm@ rpm)",
    "130@ 2500(kgm@ rpm)",
    "115@ 2,500(kgm@ rpm)",
]


for torque_text in suspicious_values:

    rows = df[df["torque"] == torque_text]

    if len(rows) == 0:
        print(
            f"\n{torque_text}: "
            "No records found."
        )
        continue

    valid_numeric_values = (
        rows["torque_nm"]
        .dropna()
        .tolist()
    )

    print(
        f"\n{torque_text}"
    )

    print(
        "Records:",
        len(rows)
    )

    print(
        "Non-missing torque_nm values:",
        len(valid_numeric_values)
    )

    if len(valid_numeric_values) == 0:

        print(
            "PASS: All suspicious values "
            "were converted to missing."
        )

    else:

        print(
            "FAIL: Suspicious numeric values remain."
        )


# ============================================================
# TORQUE RANGE
# ============================================================

print("\n" + "=" * 60)
print("TORQUE NUMERICAL SUMMARY")
print("=" * 60)

print(
    df["torque_nm"].describe()
)


# ============================================================
# AUDIT COLUMN CHECK
# ============================================================

print("\n" + "=" * 60)
print("AUDIT COLUMN CHECK")
print("=" * 60)


if "torque_nm_before_correction" in df.columns:

    print(
        "PASS: Audit column exists."
    )

else:

    print(
        "FAIL: Audit column is missing."
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("TORQUE CORRECTED DATASET VALIDATION COMPLETED")
print("=" * 60)