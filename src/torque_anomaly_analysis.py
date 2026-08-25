import pandas as pd

# --------------------------------------------------
# Load parsed dataset
# --------------------------------------------------

data_path = "data/processed/car_data_parsed.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Torque values originating from kgm formats
# --------------------------------------------------

kgm_rows = df[
    df["torque"].astype(str).str.contains(
        "kgm",
        case=False,
        na=False
    )
].copy()


print("\n" + "=" * 60)
print("TORQUE KGM ANOMALY ANALYSIS")
print("=" * 60)


print("\nTotal rows containing kgm:")
print(len(kgm_rows))


# --------------------------------------------------
# Show suspicious converted torque values
# --------------------------------------------------

suspicious_kgm = kgm_rows[
    kgm_rows["torque_nm"] > 500
].copy()


print("\nRows with kgm torque converted above 500 Nm:")

print(
    suspicious_kgm[
        [
            "name",
            "year",
            "fuel",
            "engine",
            "max_power",
            "torque",
            "torque_nm"
        ]
    ]
    .sort_values("torque_nm", ascending=False)
    .to_string(index=False)
)


print("\nNumber of suspicious kgm rows:")
print(len(suspicious_kgm))


# --------------------------------------------------
# Show all unique kgm formats
# --------------------------------------------------

print("\n" + "=" * 60)
print("UNIQUE KGM TORQUE FORMATS")
print("=" * 60)

unique_kgm = (
    kgm_rows["torque"]
    .value_counts()
)

print(unique_kgm.to_string())


# --------------------------------------------------
# Compare suspicious models with similar records
# --------------------------------------------------

print("\n" + "=" * 60)
print("SUSPICIOUS MODEL INVESTIGATION")
print("=" * 60)

suspicious_models = suspicious_kgm["name"].unique()


for model in suspicious_models:

    print("\n" + "-" * 60)
    print("MODEL:", model)
    print("-" * 60)

    model_rows = df[
        df["name"] == model
    ]

    print(
        model_rows[
            [
                "name",
                "year",
                "fuel",
                "engine",
                "max_power",
                "torque",
                "torque_nm"
            ]
        ]
        .to_string(index=False)
    )


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("TORQUE KGM ANALYSIS COMPLETED")
print("=" * 60)