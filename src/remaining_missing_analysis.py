import pandas as pd


# --------------------------------------------------
# Load recovered dataset
# --------------------------------------------------

data_path = "data/processed/car_data_recovered.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Technical columns
# --------------------------------------------------

technical_columns = [
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats"
]


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 60)
print("REMAINING MISSING VALUE ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# Remaining missing values
# --------------------------------------------------

print("\nRemaining missing values by column:")

missing_counts = df[technical_columns].isnull().sum()

print(missing_counts)


print("\nRemaining missing value percentages:")

missing_percentage = (
    df[technical_columns].isnull().sum()
    / len(df)
    * 100
).round(2)

print(missing_percentage)


# --------------------------------------------------
# Missing values by fuel
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES BY FUEL TYPE")
print("=" * 60)

for column in technical_columns:

    print(f"\n{column}:")

    print(
        df.loc[
            df[column].isnull(),
            "fuel"
        ].value_counts()
    )


# --------------------------------------------------
# Missing values by year
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES BY YEAR")
print("=" * 60)

for column in technical_columns:

    print(f"\n{column}:")

    print(
        df.loc[
            df[column].isnull(),
            "year"
        ].value_counts()
        .sort_index()
    )


# --------------------------------------------------
# Missing values by seller type
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES BY SELLER TYPE")
print("=" * 60)

for column in technical_columns:

    print(f"\n{column}:")

    print(
        df.loc[
            df[column].isnull(),
            "seller_type"
        ].value_counts()
    )


# --------------------------------------------------
# Missing values by transmission
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES BY TRANSMISSION")
print("=" * 60)

for column in technical_columns:

    print(f"\n{column}:")

    print(
        df.loc[
            df[column].isnull(),
            "transmission"
        ].value_counts()
    )


# --------------------------------------------------
# Missing values by owner
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES BY OWNER")
print("=" * 60)

for column in technical_columns:

    print(f"\n{column}:")

    print(
        df.loc[
            df[column].isnull(),
            "owner"
        ].value_counts()
    )


# --------------------------------------------------
# Missing values by model
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES BY CAR MODEL")
print("=" * 60)

for column in technical_columns:

    print(f"\n{column}:")

    model_missing = (
        df.loc[
            df[column].isnull(),
            "name"
        ]
        .value_counts()
        .head(20)
    )

    print(model_missing)



    # --------------------------------------------------
# Model-Level Recovery Potential
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL-LEVEL RECOVERY POTENTIAL")
print("=" * 60)

model_recovery_results = []

for column in technical_columns:

    missing_rows = df[df[column].isnull()]

    for model_name in missing_rows["name"].unique():

        model_rows = df[
            df["name"] == model_name
        ]

        complete_values = (
            model_rows[column]
            .dropna()
            .unique()
        )

        model_recovery_results.append({
            "column": column,
            "name": model_name,
            "missing_rows": len(
                missing_rows[
                    missing_rows["name"] == model_name
                ]
            ),
            "complete_records": len(
                model_rows[
                    model_rows[column].notna()
                ]
            ),
            "unique_values": len(complete_values)
        })


model_recovery_df = pd.DataFrame(
    model_recovery_results
)


print("\nModels with missing values:")

print(
    model_recovery_df
    .head(50)
    .to_string(index=False)
)


# --------------------------------------------------
# Models with exactly one known specification
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODELS WITH UNIQUE KNOWN SPECIFICATIONS")
print("=" * 60)

unique_model_specs = model_recovery_df[
    (model_recovery_df["complete_records"] > 0)
    & (model_recovery_df["unique_values"] == 1)
]

print(
    unique_model_specs
    .head(50)
    .to_string(index=False)
)

print("\nTotal model-column combinations:")
print(len(model_recovery_df))

print("\nModel-column combinations with exactly one known value:")
print(len(unique_model_specs))

# --------------------------------------------------
# Model-Level Recoverable Value Count
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL-LEVEL RECOVERABLE VALUE COUNT")
print("=" * 60)

model_level_recoverable = []

for _, row in unique_model_specs.iterrows():

    column = row["column"]
    model_name = row["name"]

    missing_count = row["missing_rows"]

    model_level_recoverable.append({
        "column": column,
        "name": model_name,
        "missing_rows": missing_count
    })


model_level_recoverable_df = pd.DataFrame(
    model_level_recoverable
)

print("\nPotentially recoverable values by model and column:")

print(
    model_level_recoverable_df
    .to_string(index=False)
)

print("\nTotal potentially recoverable values:")

print(
    model_level_recoverable_df["missing_rows"].sum()
)

print("\nPotentially recoverable values by column:")

print(
    model_level_recoverable_df
    .groupby("column")["missing_rows"]
    .sum()
)

# --------------------------------------------------
# Model-Level Specification vs Year Analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL-LEVEL SPECIFICATION VS YEAR ANALYSIS")
print("=" * 60)

model_year_analysis = []

for _, row in unique_model_specs.iterrows():

    column = row["column"]
    model_name = row["name"]

    model_data = df[
        df["name"] == model_name
    ][
        ["name", "year", column]
    ].dropna(subset=[column])

    unique_years = model_data["year"].nunique()

    year_value_combinations = (
        model_data[["year", column]]
        .drop_duplicates()
        .sort_values("year")
    )

    model_year_analysis.append({
        "column": column,
        "name": model_name,
        "number_of_years": unique_years,
        "year_value_combinations":
            len(year_value_combinations)
    })


model_year_analysis_df = pd.DataFrame(
    model_year_analysis
)

print("\nModel-level year consistency:")

print(
    model_year_analysis_df
    .to_string(index=False)
)


# --------------------------------------------------
# Show detailed year/specification combinations
# --------------------------------------------------

print("\n" + "=" * 60)
print("DETAILED MODEL-YEAR SPECIFICATIONS")
print("=" * 60)

for _, row in unique_model_specs.iterrows():

    column = row["column"]
    model_name = row["name"]

    model_data = df[
        (df["name"] == model_name)
        & (df[column].notna())
    ][
        ["year", column]
    ].drop_duplicates().sort_values("year")

    print("\n" + "-" * 60)
    print(f"MODEL: {model_name}")
    print(f"COLUMN: {column}")
    print("-" * 60)

    print(
        model_data.to_string(index=False)
    )


    # --------------------------------------------------
# Model-Level Recovery Confidence Analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL-LEVEL RECOVERY CONFIDENCE ANALYSIS")
print("=" * 60)

confidence_results = []

for _, row in unique_model_specs.iterrows():

    column = row["column"]
    model_name = row["name"]

    # All rows where this model has a missing value
    missing_rows = df[
        (df["name"] == model_name)
        & (df[column].isnull())
    ][["name", "year"]]

    # Years where a complete specification exists
    known_years = set(
        df[
            (df["name"] == model_name)
            & (df[column].notna())
        ]["year"]
    )

    for _, missing_row in missing_rows.iterrows():

        missing_year = missing_row["year"]

        if missing_year in known_years:
            confidence = "HIGH"
        else:
            confidence = "LOWER"

        confidence_results.append({
            "name": model_name,
            "year": missing_year,
            "column": column,
            "confidence": confidence
        })


confidence_df = pd.DataFrame(
    confidence_results
)


print("\nRecovery confidence distribution:")

print(
    confidence_df["confidence"]
    .value_counts()
)


print("\nRecovery confidence by column:")

print(
    pd.crosstab(
        confidence_df["column"],
        confidence_df["confidence"]
    )
)


print("\nPotentially high-confidence values:")

print(
    confidence_df[
        confidence_df["confidence"] == "HIGH"
    ]
    .to_string(index=False)
)