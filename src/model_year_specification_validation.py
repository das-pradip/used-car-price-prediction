import pandas as pd


# ============================================================
# MODEL-YEAR SPECIFICATION VALIDATION
# ============================================================

DATA_PATH = "data/processed/car_data_torque_corrected.csv"

df = pd.read_csv(DATA_PATH)


print("\n" + "=" * 60)
print("MODEL-YEAR SPECIFICATION VALIDATION")
print("=" * 60)


technical_columns = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]


recoverable_models = [
    "Maruti Omni E 8 Str STD",
    "Maruti Swift VDI BSIV",
    "Toyota Etios GD",
    "Toyota Etios Liva G",
    "Toyota Etios Liva GD",
    "Toyota Etios Liva GD SP",
    "Toyota Etios VD"
]


# ============================================================
# ANALYZE EACH MODEL
# ============================================================

for model in recoverable_models:

    model_df = df[df["name"] == model]

    print("\n" + "-" * 60)
    print(f"MODEL: {model}")
    print("-" * 60)

    print("\nYears present:")

    print(
        sorted(
            model_df["year"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    for column in technical_columns:

        known = model_df.dropna(
            subset=[column]
        )

        if known.empty:
            continue

        print(f"\nCOLUMN: {column}")

        year_values = (
            known
            .groupby("year")[column]
            .agg(
                unique_values="nunique",
                values=lambda x: sorted(
                    x.dropna().unique().tolist()
                )
            )
            .reset_index()
        )

        print(
            year_values.to_string(index=False)
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL-YEAR VALIDATION SUMMARY")
print("=" * 60)


summary = []


for model in recoverable_models:

    model_df = df[df["name"] == model]

    for column in technical_columns:

        known = model_df.dropna(
            subset=[column]
        )

        if known.empty:
            continue

        year_groups = (
            known.groupby("year")[column]
            .nunique()
        )

        inconsistent_years = (
            year_groups[year_groups > 1]
        )

        summary.append({
            "name": model,
            "column": column,
            "number_of_years": len(year_groups),
            "years_with_conflicts": len(
                inconsistent_years
            ),
            "consistent": len(
                inconsistent_years
            ) == 0
        })


summary_df = pd.DataFrame(summary)


print(
    summary_df.to_string(index=False)
)


print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)


conflicts = summary_df[
    summary_df["consistent"] == False
]


if conflicts.empty:

    print(
        "PASS: No model-year specification "
        "conflicts found."
    )

else:

    print(
        "WARNING: Model-year specification "
        "conflicts found:"
    )

    print(
        conflicts.to_string(index=False)
    )


print("\n" + "=" * 60)
print("MODEL-YEAR SPECIFICATION VALIDATION COMPLETED")
print("=" * 60)