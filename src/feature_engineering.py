import pandas as pd


# --------------------------------------------------
# Load latest recovered dataset
# --------------------------------------------------

data_path = "data/processed/car_data_model_recovered.csv"

df = pd.read_csv(data_path)


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

print("\nOriginal shape:")
print(df.shape)


# --------------------------------------------------
# Extract brand
# --------------------------------------------------

def extract_brand(name):
    name = str(name).strip()

    # Multi-word / special brands
    multi_word_brands = [
        "Land Rover",
        "Rolls-Royce",
        "Mercedes-Benz",
    ]

    for brand in multi_word_brands:
        if name.startswith(brand):
            return brand

    # Default: first word
    return name.split()[0]


df["brand"] = df["name"].apply(extract_brand)


# --------------------------------------------------
# Create car age
# --------------------------------------------------

reference_year = df["year"].max()

df["car_age"] = (
    reference_year - df["year"]
)


# --------------------------------------------------
# Feature groups
# --------------------------------------------------

target_column = "selling_price"

numerical_features = [
    "year",
    "car_age",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]

categorical_features = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


# --------------------------------------------------
# Display feature information
# --------------------------------------------------

print("\nReference year:")
print(reference_year)


print("\nNew feature preview:")

print(
    df[
        [
            "name",
            "brand",
            "year",
            "car_age"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# --------------------------------------------------
# Feature summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("MODEL FEATURES")
print("=" * 70)


print("\nTarget:")
print(target_column)


print("\nNumerical features:")

for feature in numerical_features:
    print("-", feature)


print("\nCategorical features:")

for feature in categorical_features:
    print("-", feature)


# --------------------------------------------------
# Missing values in model features
# --------------------------------------------------

model_features = (
    numerical_features
    + categorical_features
)


print("\n" + "=" * 70)
print("MISSING VALUES IN MODEL FEATURES")
print("=" * 70)


print(
    df[model_features]
    .isnull()
    .sum()
)


# --------------------------------------------------
# Brand distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("BRAND DISTRIBUTION")
print("=" * 70)


print(
    df["brand"]
    .value_counts()
    .head(20)
)


# --------------------------------------------------
# Final shape
# --------------------------------------------------

print("\nFinal shape:")
print(df.shape)

# --------------------------------------------------
# Save engineered dataset
# --------------------------------------------------

output_path = (
    "data/processed/"
    "car_data_feature_engineered.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("ENGINEERED DATASET SAVED")
print("=" * 70)

print("\nOutput:")
print(output_path)