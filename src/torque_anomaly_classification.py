import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/car_data_parsed.csv")


# ============================================================
# CLASSIFICATION
# ============================================================

high_confidence_recovery = {
    "110@ 3,000(kgm@ rpm)": 110.0,
    "190@ 21,800(kgm@ rpm)": 190.0,
}


suspicious_unrecoverable = {
    "145@ 4,100(kgm@ rpm)",
    "130@ 2500(kgm@ rpm)",
    "115@ 2,500(kgm@ rpm)",
    "115@ 2500(kgm@ rpm)",
}


valid_kgm = {
    "51@ 1,750-3,000(kgm@ rpm)",
    "53@ 2,000-2,750(kgm@ rpm)",
}


# ============================================================
# CREATE CLASSIFICATION
# ============================================================

def classify_torque(value):

    if pd.isna(value):
        return "MISSING"

    if value in high_confidence_recovery:
        return "HIGH_CONFIDENCE_RECOVERY"

    if value in suspicious_unrecoverable:
        return "SUSPICIOUS_UNRECOVERABLE"

    if value in valid_kgm:
        return "VALID_KGM"

    return "NORMAL"


df["torque_classification"] = df["torque"].apply(classify_torque)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TORQUE ANOMALY CLASSIFICATION")
print("=" * 60)

print("\nClassification counts:")

print(
    df["torque_classification"]
    .value_counts()
)


# ============================================================
# HIGH CONFIDENCE RECOVERY
# ============================================================

print("\n" + "=" * 60)
print("HIGH CONFIDENCE RECOVERIES")
print("=" * 60)

high = df[
    df["torque_classification"]
    == "HIGH_CONFIDENCE_RECOVERY"
].copy()

print(
    high[
        [
            "name",
            "year",
            "engine",
            "max_power",
            "torque",
            "torque_nm",
        ]
    ].to_string(index=False)
)


# ============================================================
# SUSPICIOUS UNRECOVERABLE
# ============================================================

print("\n" + "=" * 60)
print("SUSPICIOUS BUT UNRECOVERABLE")
print("=" * 60)

suspicious = df[
    df["torque_classification"]
    == "SUSPICIOUS_UNRECOVERABLE"
].copy()

print(
    suspicious[
        [
            "name",
            "year",
            "engine",
            "max_power",
            "torque",
            "torque_nm",
        ]
    ].to_string(index=False)
)


# ============================================================
# VALID KGM
# ============================================================

print("\n" + "=" * 60)
print("VALID KGM TORQUE")
print("=" * 60)

valid = df[
    df["torque_classification"]
    == "VALID_KGM"
].copy()

print(
    valid[
        [
            "name",
            "year",
            "engine",
            "max_power",
            "torque",
            "torque_nm",
        ]
    ].to_string(index=False)
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION COMPLETED")
print("=" * 60)