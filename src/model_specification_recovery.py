import pandas as pd


# ============================================================
# MODEL SPECIFICATION CONTROLLED RECOVERY
# ============================================================

INPUT_PATH = "data/processed/car_data_torque_corrected.csv"

OUTPUT_PATH = "data/processed/car_data_model_recovered.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

original_shape = df.shape


print("\n" + "=" * 70)
print("MODEL SPECIFICATION CONTROLLED RECOVERY")
print("=" * 70)

print("\nInput dataset:")
print(INPUT_PATH)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# TECHNICAL FEATURES
# ============================================================

technical_columns = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]


# ============================================================
# CREATE AUDIT COLUMN
# ============================================================

df["model_recovery_applied"] = False


# ============================================================
# RECOVERY RULES
# ============================================================
#
# We only recover values where the same model has a strong,
# consistent specification across many known records.
#
# RECOVERY GROUP 1
# Maruti Swift VDI BSIV
#
# 16 missing rows × 5 features = 80 values
#
# RECOVERY GROUP 2
# Toyota Etios GD
#
# 10 missing rows × 4 features = 40 values
#
# max_power is NOT recovered because known values vary
# between 67.04 and 67.05 bhp.
#
# RECOVERY GROUP 3
# Toyota Etios Liva GD
#
# 6 missing rows × 4 features = 24 values
#
# max_power is NOT recovered because known values vary
# between 67.04 and 67.06 bhp.
#
# TOTAL = 80 + 40 + 24 = 144
# ============================================================


recovery_rules = {

    "Maruti Swift VDI BSIV": {
        "mileage_value": 25.20,
        "engine_cc": 1248.0,
        "max_power_bhp": 74.0,
        "torque_nm": 190.0,
        "seats": 5.0
    },

    "Toyota Etios GD": {
        "mileage_value": 23.59,
        "engine_cc": 1364.0,
        "torque_nm": 170.0,
        "seats": 5.0
    },

    "Toyota Etios Liva GD": {
        "mileage_value": 23.59,
        "engine_cc": 1364.0,
        "torque_nm": 170.0,
        "seats": 5.0
    }
}


# ============================================================
# MISSING VALUES BEFORE RECOVERY
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE RECOVERY")
print("=" * 70)

missing_before = df[technical_columns].isna().sum()

print(missing_before)


# ============================================================
# APPLY CONTROLLED RECOVERY
# ============================================================

recovery_log = []


for model, feature_values in recovery_rules.items():

    model_mask = df["name"] == model

    for column, value in feature_values.items():

        missing_mask = (
            model_mask
            & df[column].isna()
        )

        affected_rows = missing_mask.sum()

        if affected_rows > 0:

            df.loc[
                missing_mask,
                column
            ] = value

            df.loc[
                missing_mask,
                "model_recovery_applied"
            ] = True

            recovery_log.append({
                "name": model,
                "column": column,
                "value": value,
                "rows_recovered": affected_rows
            })


# ============================================================
# RECOVERY LOG
# ============================================================

recovery_log_df = pd.DataFrame(recovery_log)


print("\n" + "=" * 70)
print("RECOVERY LOG")
print("=" * 70)

if recovery_log_df.empty:

    print("No values were recovered.")

else:

    print(
        recovery_log_df.to_string(index=False)
    )


# ============================================================
# TOTAL VALUES RECOVERED
# ============================================================

if recovery_log_df.empty:

    total_recovered = 0

else:

    total_recovered = (
        recovery_log_df["rows_recovered"].sum()
    )


print("\nTotal values recovered:")
print(total_recovered)


# ============================================================
# EXPECTED RECOVERY
# ============================================================

EXPECTED_RECOVERY = 144


print("\nExpected values recovered:")
print(EXPECTED_RECOVERY)


if total_recovered == EXPECTED_RECOVERY:

    print("PASS: Exactly 144 values were recovered.")

else:

    print(
        f"FAIL: Expected {EXPECTED_RECOVERY}, "
        f"but recovered {total_recovered}."
    )


# ============================================================
# MISSING VALUES AFTER RECOVERY
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES AFTER RECOVERY")
print("=" * 70)

missing_after = df[technical_columns].isna().sum()

print(missing_after)


# ============================================================
# EXPECTED MISSING VALUES
# ============================================================

expected_missing = {
    "mileage_value": 167,
    "engine_cc": 167,
    "max_power_bhp": 184,
    "torque_nm": 187,
    "seats": 167
}

print("\nExpected missing values:")

print(
    pd.Series(expected_missing)
)


# ============================================================
# VALIDATE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE VALIDATION")
print("=" * 70)


validation_passed = True


for column in technical_columns:

    actual = missing_after[column]

    expected = expected_missing[column]

    if actual == expected:

        print(
            f"PASS: {column} "
            f"({actual} missing)"
        )

    else:

        print(
            f"FAIL: {column} "
            f"(expected {expected}, got {actual})"
        )

        validation_passed = False


# ============================================================
# ROW COUNT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET STRUCTURE VALIDATION")
print("=" * 70)


if df.shape[0] == original_shape[0]:

    print(
        f"PASS: Row count unchanged "
        f"({df.shape[0]} rows)"
    )

else:

    print(
        f"FAIL: Row count changed "
        f"from {original_shape[0]} to {df.shape[0]}"
    )

    validation_passed = False


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nExact duplicate rows:")
print(duplicate_count)


if duplicate_count == 0:

    print("PASS: No exact duplicate rows.")

else:

    print("WARNING: Exact duplicate rows detected.")


# ============================================================
# AUDIT COLUMN VALIDATION
# ============================================================

print("\nAudit column:")

print(
    df["model_recovery_applied"].value_counts()
)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("RECOVERED DATASET SAVED")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_PATH)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


if (
    total_recovered == EXPECTED_RECOVERY
    and all(
        missing_after[column] == expected_missing[column]
        for column in technical_columns
    )
    and df.shape[0] == original_shape[0]
):

    print(
        "\nPASS: MODEL SPECIFICATION RECOVERY "
        "COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "\nFAIL: MODEL SPECIFICATION RECOVERY "
        "VALIDATION FAILED"
    )