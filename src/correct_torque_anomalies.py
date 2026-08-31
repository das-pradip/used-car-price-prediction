import pandas as pd


# ============================================================
# LOAD PARSED DATASET
# ============================================================

input_path = "data/processed/car_data_parsed.csv"

df = pd.read_csv(input_path)


print("\n" + "=" * 60)
print("TORQUE ANOMALY CORRECTION")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# HIGH-CONFIDENCE RECOVERIES
# ============================================================

high_confidence_recoveries = {
    "110@ 3,000(kgm@ rpm)": 110.0,
    "190@ 21,800(kgm@ rpm)": 190.0,
}


# ============================================================
# SUSPICIOUS UNRECOVERABLE VALUES
# ============================================================

suspicious_unrecoverable = {
    "115@ 2,500(kgm@ rpm)",
    "115@ 2500(kgm@ rpm)",
    "145@ 4,100(kgm@ rpm)",
    "130@ 2500(kgm@ rpm)",
}


# ============================================================
# COUNT BEFORE CORRECTION
# ============================================================

print("\nHigh-confidence values before correction:")

for torque_value in high_confidence_recoveries:
    count = (df["torque"] == torque_value).sum()

    print(
        f"{torque_value}: {count} occurrence(s)"
    )


print("\nSuspicious values before correction:")

for torque_value in suspicious_unrecoverable:
    count = (df["torque"] == torque_value).sum()

    print(
        f"{torque_value}: {count} occurrence(s)"
    )


# ============================================================
# PRESERVE ORIGINAL PARSED TORQUE
# ============================================================

df["torque_nm_before_correction"] = df["torque_nm"]


# ============================================================
# APPLY HIGH-CONFIDENCE RECOVERIES
# ============================================================

high_confidence_count = 0

for torque_value, corrected_value in high_confidence_recoveries.items():

    mask = df["torque"] == torque_value

    high_confidence_count += mask.sum()

    df.loc[mask, "torque_nm"] = corrected_value


# ============================================================
# INVALIDATE SUSPICIOUS TORQUE VALUES
# ============================================================

suspicious_count = 0

for torque_value in suspicious_unrecoverable:

    mask = df["torque"] == torque_value

    suspicious_count += mask.sum()

    df.loc[mask, "torque_nm"] = pd.NA


# ============================================================
# CORRECTION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CORRECTION SUMMARY")
print("=" * 60)

print(
    "\nNumber of high-confidence values corrected:"
)

print(high_confidence_count)

print(
    "\nNumber of suspicious values converted to missing:"
)

print(suspicious_count)


# ============================================================
# VERIFY HIGH-CONFIDENCE RECOVERIES
# ============================================================

print("\n" + "=" * 60)
print("HIGH-CONFIDENCE RECOVERY RESULTS")
print("=" * 60)

for torque_value, corrected_value in high_confidence_recoveries.items():

    recovered_rows = df[
        df["torque"] == torque_value
    ]

    if len(recovered_rows) > 0:

        print(
            f"\nOriginal torque: {torque_value}"
        )

        print(
            f"Corrected torque_nm: {corrected_value}"
        )

        print(
            f"Rows affected: {len(recovered_rows)}"
        )


# ============================================================
# VERIFY SUSPICIOUS VALUES
# ============================================================

print("\n" + "=" * 60)
print("SUSPICIOUS TORQUE RESULTS")
print("=" * 60)

remaining_suspicious = 0

for torque_value in suspicious_unrecoverable:

    mask = df["torque"] == torque_value

    count = df.loc[
        mask & df["torque_nm"].isna()
    ].shape[0]

    remaining_suspicious += count

    if count > 0:

        print(
            f"{torque_value}: "
            f"{count} converted to NaN"
        )


# ============================================================
# TORQUE MISSING VALUE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("TORQUE MISSING VALUE VALIDATION")
print("=" * 60)

print("\nTorque missing values before correction:")

print(
    df["torque_nm_before_correction"].isna().sum()
)

print("\nTorque missing values after correction:")

print(
    df["torque_nm"].isna().sum()
)

expected_missing = 200 + suspicious_count

print("\nExpected torque missing values:")

print(expected_missing)


# ============================================================
# VALIDATE EXPECTED RESULT
# ============================================================

actual_missing = df["torque_nm"].isna().sum()

print("\nValidation:")

if actual_missing == expected_missing:

    print(
        "PASS: Expected torque missing value count."
    )

else:

    print(
        "FAIL: Unexpected torque missing value count."
    )


# ============================================================
# VALIDATE DATASET SHAPE
# ============================================================

print("\nDataset shape after correction:")

print(df.shape)

if df.shape[0] == 6926:

    print(
        "PASS: Row count remains unchanged."
    )

else:

    print(
        "FAIL: Row count changed."
    )


# ============================================================
# VALIDATE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nExact duplicate rows:")

print(duplicate_count)

if duplicate_count == 0:

    print(
        "PASS: No exact duplicate rows."
    )

else:

    print(
        "WARNING: Duplicate rows detected."
    )


# ============================================================
# SAVE CORRECTED DATASET
# ============================================================

output_path = (
    "data/processed/"
    "car_data_torque_corrected.csv"
)

df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 60)
print("CORRECTED DATASET SAVED")
print("=" * 60)

print("\nSaved dataset:")

print(output_path)

print("\nDataset shape:")

print(df.shape)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("TORQUE ANOMALY CORRECTION COMPLETED")
print("=" * 60)