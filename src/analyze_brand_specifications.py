from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/car_data_feature_engineered.csv")

TECHNICAL_FEATURES = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]


df = pd.read_csv(DATA_PATH)


print("=" * 80)
print("BRAND + TECHNICAL SPECIFICATION ANALYSIS")
print("=" * 80)

print(f"\nDataset shape: {df.shape}")


# ---------------------------------------------------------
# Overall technical ranges by brand
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("1. TECHNICAL RANGES BY BRAND")
print("=" * 80)

brand_summary = (
    df.groupby("brand")[TECHNICAL_FEATURES]
    .agg(["count", "min", "median", "max"])
    .round(2)
)

for brand in sorted(df["brand"].dropna().unique()):

    print("\n" + "-" * 80)
    print(f"BRAND: {brand}")
    print("-" * 80)

    print(
        brand_summary.loc[brand].to_string()
    )


# ---------------------------------------------------------
# Individual vehicle examples
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("2. EXAMPLE VEHICLE SPECIFICATIONS")
print("=" * 80)

example_columns = [
    "brand",
    "name",
    "year",
    "fuel",
    "transmission",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
    "selling_price",
]

examples = (
    df[
        example_columns
    ]
    .dropna(
        subset=[
            "engine_cc",
            "max_power_bhp",
            "torque_nm",
        ]
    )
    .sort_values(
        ["brand", "name", "year"]
    )
)

print(
    examples.head(100).to_string(index=False)
)


# ---------------------------------------------------------
# Volvo-specific analysis
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("3. VOLVO SPECIFICATION ANALYSIS")
print("=" * 80)

volvo = df[df["brand"] == "Volvo"].copy()

if len(volvo) == 0:

    print("No Volvo records found.")

else:

    print(f"Volvo rows: {len(volvo)}")

    print(
        volvo[
            example_columns
        ]
        .sort_values(["name", "year"])
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Brand/model frequency
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("4. BRAND + MODEL FREQUENCY")
print("=" * 80)

model_frequency = (
    df.groupby(["brand", "name"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(
    model_frequency.head(100).to_string(index=False)
)


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)