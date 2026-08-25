import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/car_data_parsed.csv")


# ============================================================
# SUSPICIOUS KGM VALUES
# ============================================================

suspicious_values = [
    "145@ 4,100(kgm@ rpm)",
    "130@ 2500(kgm@ rpm)",
    "115@ 2,500(kgm@ rpm)",
    "115@ 2500(kgm@ rpm)",
    "110@ 3,000(kgm@ rpm)",
    "190@ 21,800(kgm@ rpm)",
    "51@ 1,750-3,000(kgm@ rpm)",
    "53@ 2,000-2,750(kgm@ rpm)",
]


# ============================================================
# INVESTIGATION
# ============================================================

print("\n" + "=" * 60)
print("SUSPICIOUS TORQUE RECORD INVESTIGATION")
print("=" * 60)


for torque_value in suspicious_values:

    rows = df[df["torque"] == torque_value]

    if len(rows) == 0:
        continue

    print("\n" + "-" * 60)
    print("TORQUE:", torque_value)
    print("-" * 60)

    print(
        rows[
            [
                "name",
                "year",
                "fuel",
                "engine",
                "max_power",
                "torque",
                "torque_nm",
            ]
        ].to_string(index=False)
    )


# ============================================================
# COMPARE NORMAL KGM VALUES
# ============================================================

print("\n" + "=" * 60)
print("NORMAL KGM TORQUE EXAMPLES")
print("=" * 60)


normal_patterns = [
    "12.7@ 2,700(kgm@ rpm)",
    "16.3@ 2,000(kgm@ rpm)",
    "20.4@ 1400-3400(kgm@ rpm)",
    "22.4 kgm at 1750-2750rpm",
    "24 KGM at 1900-2750 RPM",
]


for torque_value in normal_patterns:

    rows = df[df["torque"] == torque_value]

    if len(rows) == 0:
        continue

    print("\n" + "-" * 60)
    print("TORQUE:", torque_value)
    print("-" * 60)

    print(
        rows[
            [
                "name",
                "year",
                "fuel",
                "engine",
                "max_power",
                "torque",
                "torque_nm",
            ]
        ].head(10).to_string(index=False)
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("INVESTIGATION COMPLETED")
print("=" * 60)