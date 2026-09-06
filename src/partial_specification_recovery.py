import pandas as pd


# --------------------------------------------------
# Load latest recovered dataset
# --------------------------------------------------

data_path = "data/processed/car_data_model_recovered.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Technical columns
# --------------------------------------------------

technical_columns = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]


# --------------------------------------------------
# Identify partially missing rows
# --------------------------------------------------

missing_per_row = (
    df[technical_columns]
    .isnull()
    .sum(axis=1)
)

partial_mask = (
    (missing_per_row > 0)
    & (missing_per_row < 5)
)


partial_df = df.loc[
    partial_mask
].copy()


# --------------------------------------------------
# Recovery analysis
# --------------------------------------------------

print("\n" + "=" * 70)
print("PARTIAL SPECIFICATION RECOVERY ANALYSIS")
print("=" * 70)

print("\nTotal dataset rows:")
print(len(df))

print("\nPartially missing rows:")
print(len(partial_df))


# --------------------------------------------------
# Exact-name recovery analysis
# --------------------------------------------------

recovery_results = []


for index, row in partial_df.iterrows():

    car_name = row["name"]

    # All records having exactly the same name
    same_name = df[
        df["name"] == car_name
    ]

    for column in technical_columns:

        # Only investigate if current row is missing
        if pd.isna(row[column]):

            known_values = (
                same_name[column]
                .dropna()
                .unique()
            )

            # --------------------------------------
            # No known value
            # --------------------------------------

            if len(known_values) == 0:

                confidence = "LOW"

                suggested_value = None

            # --------------------------------------
            # Exactly one known value
            # --------------------------------------

            elif len(known_values) == 1:

                confidence = "HIGH"

                suggested_value = known_values[0]

            # --------------------------------------
            # Multiple known values
            # --------------------------------------

            else:

                confidence = "CONFLICT"

                suggested_value = None


            recovery_results.append({

                "row_index": index,

                "name": car_name,

                "year": row["year"],

                "column": column,

                "known_records": len(
                    same_name[column].dropna()
                ),

                "unique_known_values": len(
                    known_values
                ),

                "known_values": (
                    list(known_values)
                    if len(known_values) > 0
                    else []
                ),

                "confidence": confidence,

                "suggested_value": suggested_value
            })


# --------------------------------------------------
# Convert to DataFrame
# --------------------------------------------------

recovery_df = pd.DataFrame(
    recovery_results
)


# --------------------------------------------------
# Recovery summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("RECOVERY SUMMARY")
print("=" * 70)


print("\nTotal missing feature values analyzed:")
print(len(recovery_df))


print("\nConfidence distribution:")

print(
    recovery_df["confidence"]
    .value_counts()
)


# --------------------------------------------------
# High-confidence values
# --------------------------------------------------

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE RECOVERY CANDIDATES")
print("=" * 70)


high_confidence = recovery_df[
    recovery_df["confidence"] == "HIGH"
]


print(
    high_confidence
    .to_string(index=False)
)


# --------------------------------------------------
# Conflicting values
# --------------------------------------------------

print("\n" + "=" * 70)
print("CONFLICTING SPECIFICATIONS")
print("=" * 70)


conflicting = recovery_df[
    recovery_df["confidence"] == "CONFLICT"
]


if len(conflicting) == 0:

    print("\nNo conflicting specifications found.")

else:

    print(
        conflicting
        .to_string(index=False)
    )


# --------------------------------------------------
# No known values
# --------------------------------------------------

print("\n" + "=" * 70)
print("NO-KNOWN-VALUE CASES")
print("=" * 70)


low_confidence = recovery_df[
    recovery_df["confidence"] == "LOW"
]


if len(low_confidence) == 0:

    print("\nNo low-confidence cases found.")

else:

    print(
        low_confidence
        .to_string(index=False)
    )


# --------------------------------------------------
# Save analysis
# --------------------------------------------------

output_path = (
    "data/processed/"
    "partial_specification_recovery_analysis.csv"
)


recovery_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 70)
print("ANALYSIS SAVED")
print("=" * 70)

print("\nOutput:")
print(output_path)