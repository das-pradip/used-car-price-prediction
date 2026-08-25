import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/car_data_parsed.csv")


# ============================================================
# SUSPICIOUS TORQUE VALUES
# ============================================================

suspicious_torque_values = [
    "145@ 4,100(kgm@ rpm)",
    "130@ 2500(kgm@ rpm)",
    "115@ 2,500(kgm@ rpm)",
    "115@ 2500(kgm@ rpm)",
    "110@ 3,000(kgm@ rpm)",
    "190@ 21,800(kgm@ rpm)",
]


# ============================================================
# FIND MODELS WITH SUSPICIOUS VALUES
# ============================================================

suspicious_rows = df[
    df["torque"].isin(suspicious_torque_values)
].copy()


models = suspicious_rows["name"].unique()


print("\n" + "=" * 60)
print("TORQUE MODEL RECOVERY ANALYSIS")
print("=" * 60)

print("\nNumber of affected rows:")
print(len(suspicious_rows))

print("\nNumber of affected models:")
print(len(models))


# ============================================================
# INVESTIGATE EACH MODEL
# ============================================================

for model in models:

    print("\n" + "-" * 60)
    print("MODEL:", model)
    print("-" * 60)

    model_rows = df[df["name"] == model].copy()

    print("\nAll torque values for this model:")

    print(
        model_rows[
            [
                "name",
                "year",
                "fuel",
                "engine",
                "max_power",
                "torque",
                "torque_nm",
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )


    print("\nTorque value frequency:")

    print(
        model_rows["torque"]
        .value_counts(dropna=False)
    )


# ============================================================
# FIND OTHER MODELS WITH SAME ENGINE + POWER
# ============================================================

print("\n" + "=" * 60)
print("ENGINE + POWER COMPARISON")
print("=" * 60)


for _, row in suspicious_rows.drop_duplicates(
    subset=["name", "torque"]
).iterrows():

    print("\n" + "-" * 60)

    print("Suspicious model:", row["name"])
    print("Engine:", row["engine"])
    print("Max power:", row["max_power"])
    print("Suspicious torque:", row["torque"])

    matching = df[
        (df["engine"] == row["engine"]) &
        (df["max_power"] == row["max_power"]) &
        (df["torque"] != row["torque"])
    ]

    print("\nOther records with same engine and power:")

    if len(matching) == 0:
        print("No matching records found.")
    else:
        print(
            matching[
                [
                    "name",
                    "year",
                    "engine",
                    "max_power",
                    "torque",
                    "torque_nm",
                ]
            ]
            .drop_duplicates()
            .head(30)
            .to_string(index=False)
        )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("TORQUE MODEL RECOVERY ANALYSIS COMPLETED")
print("=" * 60)