import pandas as pd

# --------------------------------------------------
# Load processed dataset
# --------------------------------------------------

data_path = "data/processed/car_data_deduplicated.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# Missing values by column
# --------------------------------------------------

print("\nMissing values by column:")

missing_counts = df.isnull().sum()

print(missing_counts)


# --------------------------------------------------
# Missing value percentages
# --------------------------------------------------

print("\nMissing value percentage:")

missing_percentage = (
    df.isnull().sum() / len(df) * 100
).round(2)

print(missing_percentage)


# --------------------------------------------------
# Rows containing missing values
# --------------------------------------------------

rows_with_missing = df[df.isnull().any(axis=1)]

print("\nTotal rows containing at least one missing value:")
print(len(rows_with_missing))

print("\nPercentage of rows containing missing values:")
print(
    f"{len(rows_with_missing) / len(df) * 100:.2f}%"
)


# --------------------------------------------------
# Number of missing values per row
# --------------------------------------------------

print("\nNumber of missing values per row:")

missing_per_row = (
    df.isnull()
      .sum(axis=1)
      .value_counts()
      .sort_index()
)

print(missing_per_row)


# --------------------------------------------------
# Missing value pattern
# --------------------------------------------------

print("\nMissing value patterns:")

missing_pattern = (
    df.isnull()
      .astype(int)
      .value_counts()
)

print(missing_pattern.head(20))


# --------------------------------------------------
# Rows with missing values
# --------------------------------------------------

print("\nFirst 20 rows containing missing values:")

print(
    rows_with_missing
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# Missing values by fuel type
# --------------------------------------------------

print("\nMissing values by fuel type:")

missing_by_fuel = (
    rows_with_missing["fuel"]
    .value_counts()
)

print(missing_by_fuel)


# --------------------------------------------------
# Missing values by seller type
# --------------------------------------------------

print("\nMissing values by seller type:")

missing_by_seller = (
    rows_with_missing["seller_type"]
    .value_counts()
)

print(missing_by_seller)


# --------------------------------------------------
# Missing values by transmission
# --------------------------------------------------

print("\nMissing values by transmission:")

missing_by_transmission = (
    rows_with_missing["transmission"]
    .value_counts()
)

print(missing_by_transmission)


# --------------------------------------------------
# Missing values by owner
# --------------------------------------------------

print("\nMissing values by owner:")

missing_by_owner = (
    rows_with_missing["owner"]
    .value_counts()
)

print(missing_by_owner)

# --------------------------------------------------
# Compare missing rows with available car models
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING ROW MODEL INVESTIGATION")
print("=" * 60)

# Extract the rows where technical specifications are missing
technical_columns = [
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats"
]

missing_technical = df[
    df[technical_columns].isnull().any(axis=1)
].copy()

print("\nNumber of rows with missing technical specifications:")
print(len(missing_technical))


# --------------------------------------------------
# Display car names with missing specifications
# --------------------------------------------------

print("\nCar names with missing technical specifications:")

print(
    missing_technical[
        ["name", "year", "fuel", "seller_type",
         "transmission", "owner"]
    ]
    .head(50)
    .to_string(index=False)
)


# --------------------------------------------------
# Check whether the same car name has complete
# records elsewhere in the dataset
# --------------------------------------------------

missing_names = missing_technical["name"].unique()

complete_model_rows = df[
    (~df[technical_columns].isnull().any(axis=1))
    & (df["name"].isin(missing_names))
]

print("\nComplete records available for the same car names:")

print(len(complete_model_rows))


# --------------------------------------------------
# Show matching complete records
# --------------------------------------------------

print("\nMatching complete records:")

print(
    complete_model_rows[
        ["name", "year", "fuel", "seller_type",
         "transmission", "owner",
         "mileage", "engine", "max_power",
         "torque", "seats"]
    ]
    .head(50)
    .to_string(index=False)
)

# --------------------------------------------------
# Check how many missing rows can be matched with
# complete records of the same car name
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE RECOVERY ANALYSIS")
print("=" * 60)

# Count missing technical specifications for each car name
missing_by_model = (
    missing_technical
    .groupby("name")
    .size()
    .sort_values(ascending=False)
)

print("\nNumber of missing rows by car model:")

print(
    missing_by_model
    .head(30)
    .to_string()
)


# --------------------------------------------------
# Count complete records available for each
# affected car model
# --------------------------------------------------

complete_by_model = (
    complete_model_rows
    .groupby("name")
    .size()
    .sort_values(ascending=False)
)

print("\nNumber of complete records available for affected models:")

print(
    complete_by_model
    .head(30)
    .to_string()
)


# --------------------------------------------------
# Compare missing models with complete models
# --------------------------------------------------

model_recovery = pd.DataFrame({
    "missing_rows": missing_by_model,
    "complete_rows": complete_by_model
}).fillna(0)

model_recovery["complete_rows"] = (
    model_recovery["complete_rows"]
    .astype(int)
)

model_recovery["recoverable"] = (
    model_recovery["complete_rows"] > 0
)

print("\nModel recovery summary:")

print(
    model_recovery
    .head(50)
    .to_string()
)


# --------------------------------------------------
# Overall recovery statistics
# --------------------------------------------------

recoverable_models = (
    model_recovery["recoverable"]
    .sum()
)

total_affected_models = len(model_recovery)

print("\nTotal affected car models:")
print(total_affected_models)

print("\nAffected car models with at least one complete record:")
print(recoverable_models)

print("\nAffected car models without a complete record:")
print(
    total_affected_models - recoverable_models
)


# --------------------------------------------------
# Row-level recovery potential
# --------------------------------------------------

print("\n" + "=" * 60)
print("ROW-LEVEL RECOVERY POTENTIAL")
print("=" * 60)

# Find missing rows whose car name has at least
# one complete record elsewhere
recoverable_rows = missing_technical[
    missing_technical["name"].isin(
        complete_model_rows["name"]
    )
].copy()

print("\nTotal rows with missing technical specifications:")
print(len(missing_technical))

print("\nRows potentially recoverable using same car name:")
print(len(recoverable_rows))

print("\nRows without a complete same-name record:")
print(
    len(missing_technical) - len(recoverable_rows)
)

print("\nPercentage of missing rows potentially recoverable:")

print(
    f"{len(recoverable_rows) / len(missing_technical) * 100:.2f}%"
)


# --------------------------------------------------
# Display potentially recoverable rows
# --------------------------------------------------

print("\nPotentially recoverable rows:")

print(
    recoverable_rows[
        ["name", "year", "fuel", "seller_type",
         "transmission", "owner"]
    ]
    .head(50)
    .to_string(index=False)
)

# --------------------------------------------------
# Check specification consistency for recoverable models
# --------------------------------------------------

print("\n" + "=" * 60)
print("RECOVERABLE MODEL SPECIFICATION CONSISTENCY")
print("=" * 60)

# Get the 10 car names that have complete records
recoverable_model_names = complete_model_rows["name"].unique()

# Select complete records for these models
recoverable_complete_records = df[
    (~df[technical_columns].isnull().any(axis=1))
    & (df["name"].isin(recoverable_model_names))
].copy()

# Count how many different values each technical
# specification has for each car model
specification_consistency = (
    recoverable_complete_records
    .groupby("name")[technical_columns]
    .nunique()
)

print("\nNumber of unique specification values per model:")

print(
    specification_consistency
    .to_string()
)

# --------------------------------------------------
# Inspect conflicting specifications
# --------------------------------------------------

print("\n" + "=" * 60)
print("CONFLICTING MODEL SPECIFICATIONS")
print("=" * 60)

# Models where at least one technical specification
# has more than one unique value
conflicting_models = specification_consistency[
    specification_consistency.gt(1).any(axis=1)
]

print("\nModels with conflicting specifications:")

print(
    conflicting_models.to_string()
)


# --------------------------------------------------
# Display actual specification values
# --------------------------------------------------

for model in conflicting_models.index:

    print("\n" + "-" * 60)
    print(f"MODEL: {model}")
    print("-" * 60)

    model_records = recoverable_complete_records[
        recoverable_complete_records["name"] == model
    ]

    print(
        model_records[
            ["name", "year", "fuel",
             "seller_type", "transmission",
             "owner", "mileage",
             "engine", "max_power",
             "torque", "seats"]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

  # --------------------------------------------------
# Analyze recoverable rows using model and year
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL-YEAR RECOVERY ANALYSIS")
print("=" * 60)

# Models that have both:
# 1. missing technical records
# 2. at least one complete record

affected_models = complete_model_rows["name"].unique()

recoverable_rows = df[
    df["name"].isin(affected_models)
    & df[technical_columns].isnull().any(axis=1)
].copy()

print("\nPotentially recoverable rows by model:")

print(
    recoverable_rows["name"]
    .value_counts()
)


# --------------------------------------------------
# Show missing rows for conflicting models
# --------------------------------------------------

conflicting_model_rows = recoverable_rows[
    recoverable_rows["name"].isin(conflicting_models.index)
]

print("\nMissing rows belonging to conflicting models:")

print(
    conflicting_model_rows[
        ["name", "year", "fuel",
         "seller_type", "transmission",
         "owner", "mileage",
         "engine", "max_power",
         "torque", "seats"]
    ]
    .sort_values(["name", "year"])
    .to_string(index=False)
)

# --------------------------------------------------
# Compact Model-Year Recovery Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL-YEAR RECOVERY SUMMARY")
print("=" * 60)

# Get the actual model names that have both
# missing records and at least one complete record
recoverable_model_names = (
    complete_model_rows["name"]
    .drop_duplicates()
    .tolist()
)

recovery_summary = []

for model_name in recoverable_model_names:

    missing_model_rows = missing_technical[
        missing_technical["name"] == model_name
    ]

    exact_matches = 0
    no_exact_matches = 0

    for year in missing_model_rows["year"].unique():

        complete_model_year = df[
            (~df[technical_columns].isnull().any(axis=1))
            & (df["name"] == model_name)
            & (df["year"] == year)
        ]

        missing_count = len(
            missing_model_rows[
                missing_model_rows["year"] == year
            ]
        )

        if len(complete_model_year) > 0:
            exact_matches += missing_count
        else:
            no_exact_matches += missing_count

    recovery_summary.append({
        "name": model_name,
        "missing_rows": len(missing_model_rows),
        "exact_model_year_match": exact_matches,
        "no_exact_model_year_match": no_exact_matches
    })


recovery_summary_df = pd.DataFrame(recovery_summary)

print(
    recovery_summary_df
    .to_string(index=False)
)

print("\nTotal potentially recoverable rows:")
print(
    recovery_summary_df["missing_rows"].sum()
)

print("\nRows with an exact model-year complete record:")
print(
    recovery_summary_df["exact_model_year_match"].sum()
)

print("\nRows without an exact model-year complete record:")
print(
    recovery_summary_df["no_exact_model_year_match"].sum()
)

# --------------------------------------------------
# Verify exact model-year specification consistency
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXACT MODEL-YEAR SPECIFICATION CONSISTENCY")
print("=" * 60)

exact_match_details = []

for _, missing_row in missing_technical.iterrows():

    model_name = missing_row["name"]
    model_year = missing_row["year"]

    # Find complete records with the exact same model and year
    matching_complete_rows = df[
        (~df[technical_columns].isnull().any(axis=1))
        & (df["name"] == model_name)
        & (df["year"] == model_year)
    ]

    if len(matching_complete_rows) == 0:
        continue

    # Count unique values for each technical specification
    unique_values = (
        matching_complete_rows[technical_columns]
        .nunique()
    )

    consistent = (
        unique_values <= 1
    ).all()

    exact_match_details.append({
        "name": model_name,
        "year": model_year,
        "missing_row_count": 1,
        "matching_complete_rows": len(matching_complete_rows),
        "consistent": consistent
    })


exact_match_df = pd.DataFrame(exact_match_details)

print("\nExact model-year matches found:")

print(
    exact_match_df
    .to_string(index=False)
)

print("\nTotal exact model-year matches:")
print(len(exact_match_df))

print("\nConsistent exact model-year matches:")
print(
    exact_match_df["consistent"].sum()
)

print("\nInconsistent exact model-year matches:")
print(
    (~exact_match_df["consistent"]).sum()
)

# --------------------------------------------------
# Column-Level Exact Model-Year Recovery Analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("COLUMN-LEVEL RECOVERY ANALYSIS")
print("=" * 60)

column_recovery = []

for _, missing_row in missing_technical.iterrows():

    model_name = missing_row["name"]
    model_year = missing_row["year"]

    # Find complete records with exact model + year
    matching_rows = df[
        (~df[technical_columns].isnull().any(axis=1))
        & (df["name"] == model_name)
        & (df["year"] == model_year)
    ]

    if len(matching_rows) == 0:
        continue

    for column in technical_columns:

        # Only investigate the column if it is missing
        # in the target row
        if pd.isnull(missing_row[column]):

            unique_values = (
                matching_rows[column]
                .dropna()
                .unique()
            )

            if len(unique_values) == 1:

                column_recovery.append({
                    "name": model_name,
                    "year": model_year,
                    "column": column,
                    "recovered_value": unique_values[0]
                })


column_recovery_df = pd.DataFrame(column_recovery)

print("\nPotentially recoverable missing values:")

print(
    column_recovery_df
    .to_string(index=False)
)

print("\nNumber of individual missing values that can be recovered:")

print(len(column_recovery_df))

print("\nRecoverable values by column:")

if len(column_recovery_df) > 0:

    print(
        column_recovery_df["column"]
        .value_counts()
    )

else:

    print("No recoverable values found.")

# --------------------------------------------------
# Recover Values Using Exact Model-Year Matches
# --------------------------------------------------

print("\n" + "=" * 60)
print("RECOVERING VALUES FROM EXACT MODEL-YEAR MATCHES")
print("=" * 60)

recovered_count = 0

for index, row in df.iterrows():

    model_name = row["name"]
    model_year = row["year"]

    # Find complete records with the same model and year
    matching_rows = df[
        (~df[technical_columns].isnull().any(axis=1))
        & (df["name"] == model_name)
        & (df["year"] == model_year)
    ]

    # If no complete model-year record exists, skip
    if len(matching_rows) == 0:
        continue

    # Recover only missing values
    for column in technical_columns:

        if pd.isnull(df.loc[index, column]):

            unique_values = (
                matching_rows[column]
                .dropna()
                .unique()
            )

            # Recover only when exactly one value exists
            if len(unique_values) == 1:

                df.loc[index, column] = unique_values[0]

                recovered_count += 1


print("\nNumber of values recovered:")
print(recovered_count)

# --------------------------------------------------
# Validate Recovered Values
# --------------------------------------------------

print("\n" + "=" * 60)
print("RECOVERY VALIDATION")
print("=" * 60)

# Count remaining missing technical values
remaining_missing = df[technical_columns].isnull().sum()

print("\nRemaining missing values by technical column:")
print(remaining_missing)

print("\nTotal remaining missing technical values:")
print(remaining_missing.sum())

print("\nTechnical missing values before recovery:")
print(missing_technical[technical_columns].isnull().sum())

print("\nTechnical missing values after recovery:")
print(df[technical_columns].isnull().sum())

print("\nTotal values recovered:")
print(recovered_count)

# Verify that the expected number was recovered
expected_recovered = 42

print("\nExpected recovered values:")
print(expected_recovered)

if recovered_count == expected_recovered:
    print("PASS: Expected number of values recovered.")
else:
    print("WARNING: Recovered value count differs from expected.")


# --------------------------------------------------
# Save Dataset After Missing Value Recovery
# --------------------------------------------------

recovered_data_path = (
    "data/processed/car_data_recovered.csv"
)

df.to_csv(
    recovered_data_path,
    index=False
)

print("\n" + "=" * 60)
print("RECOVERED DATASET SAVED")
print("=" * 60)

print("\nSaved dataset:")
print(recovered_data_path)

print("\nDataset shape:")
print(df.shape)