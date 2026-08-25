import pandas as pd
import numpy as np
import re


# --------------------------------------------------
# Load recovered dataset
# --------------------------------------------------

data_path = "data/processed/car_data_recovered.csv"

df = pd.read_csv(data_path)


print("\n" + "=" * 60)
print("FEATURE PARSING")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)


# --------------------------------------------------
# Parse mileage
# --------------------------------------------------

def parse_mileage(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    match = re.search(
        r"([\d.]+)\s*(kmpl|km/kg)",
        value,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return np.nan


def parse_mileage_unit(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    match = re.search(
        r"(kmpl|km/kg)",
        value,
        re.IGNORECASE
    )

    if match:
        return match.group(1).lower()

    return np.nan


df["mileage_value"] = df["mileage"].apply(
    parse_mileage
)

df["mileage_unit"] = df["mileage"].apply(
    parse_mileage_unit
)


# --------------------------------------------------
# Parse engine
# --------------------------------------------------

def parse_engine(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    match = re.search(
        r"([\d.]+)\s*CC",
        value,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return np.nan


df["engine_cc"] = df["engine"].apply(
    parse_engine
)


# --------------------------------------------------
# Parse max power
# --------------------------------------------------

def parse_max_power(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # ----------------------------------------------
    # Standard format:
    # 74 bhp
    # 103.52 bhp
    # ----------------------------------------------

    match = re.search(
        r"([\d.]+)\s*bhp",
        value,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))


    # ----------------------------------------------
    # Numeric value without explicit unit:
    # 0
    #
    # The dataset contains a few records where
    # max_power is stored simply as "0".
    # ----------------------------------------------

    if re.fullmatch(r"[\d.]+", value):
        return float(value)


    return np.nan


df["max_power_bhp"] = df["max_power"].apply(
    parse_max_power
)


# --------------------------------------------------
# Parse torque
# --------------------------------------------------

def parse_torque(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    # ----------------------------------------------
    # Torque explicitly expressed in Nm
    # ----------------------------------------------

    nm_match = re.search(
        r"([\d.]+)\s*nm",
        value
    )

    if nm_match:
        return float(nm_match.group(1))


    # ----------------------------------------------
    # Torque explicitly expressed in kgm
    # ----------------------------------------------

    kgm_match = re.search(
        r"([\d.]+)\s*kgm",
        value
    )

    if kgm_match:
        torque_kgm = float(
            kgm_match.group(1)
        )

        return torque_kgm * 9.80665


    # ----------------------------------------------
    # Formats such as:
    # 12.7@ 2,700(kgm@ rpm)
    # ----------------------------------------------

    special_kgm_match = re.search(
        r"([\d.]+)\s*@.*kgm",
        value
    )

    if special_kgm_match:
        torque_kgm = float(
            special_kgm_match.group(1)
        )

        return torque_kgm * 9.80665


    # ----------------------------------------------
    # Formats where torque is given before @
    #
    # Examples:
    # 250@ 1250-5000rpm
    # 48@ 3,000+/-500(NM@ rpm)
    # 510@ 1600-2400
    # ----------------------------------------------

    at_match = re.search(
        r"^\s*([\d.]+)\s*@",
        value
    )

    if at_match:
        return float(
            at_match.group(1)
        )


    # ----------------------------------------------
    # Format:
    # 110(11.2)@ 4800
    #
    # The first number is torque in Nm.
    # ----------------------------------------------

    parentheses_match = re.search(
        r"^\s*([\d.]+)\s*\(",
        value
    )

    if parentheses_match:
        return float(
            parentheses_match.group(1)
        )


    # ----------------------------------------------
    # Format:
    # 210 / 1900
    #
    # First number = torque in Nm
    # Second number = RPM
    # ----------------------------------------------

    slash_match = re.search(
        r"^\s*([\d.]+)\s*/",
        value
    )

    if slash_match:
        return float(
            slash_match.group(1)
        )


    return np.nan

    value = str(value).strip().lower()

    # ----------------------------------------------
    # Torque expressed in Nm
    # ----------------------------------------------

    nm_match = re.search(
        r"([\d.]+)\s*nm",
        value
    )

    if nm_match:
        return float(nm_match.group(1))


    # ----------------------------------------------
    # Torque expressed in kgm
    # ----------------------------------------------

    kgm_match = re.search(
        r"([\d.]+)\s*kgm",
        value
    )

    if kgm_match:
        torque_kgm = float(kgm_match.group(1))

        return torque_kgm * 9.80665


    # ----------------------------------------------
    # Formats such as:
    # 12.7@ 2,700(kgm@ rpm)
    # ----------------------------------------------

    special_kgm_match = re.search(
        r"([\d.]+)\s*@.*kgm",
        value
    )

    if special_kgm_match:
        torque_kgm = float(
            special_kgm_match.group(1)
        )

        return torque_kgm * 9.80665


    return np.nan


df["torque_nm"] = df["torque"].apply(
    parse_torque
)


# --------------------------------------------------
# Seats
# --------------------------------------------------

df["seats"] = pd.to_numeric(
    df["seats"],
    errors="coerce"
)


# --------------------------------------------------
# Parsing validation
# --------------------------------------------------

print("\n" + "=" * 60)
print("PARSED FEATURE SUMMARY")
print("=" * 60)

parsed_columns = [
    "mileage_value",
    "mileage_unit",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]

print("\nParsed columns:")
print(parsed_columns)


print("\nParsed data types:")
print(
    df[parsed_columns].dtypes
)


print("\nMissing values after parsing:")

print(
    df[parsed_columns]
    .isnull()
    .sum()
)


# --------------------------------------------------
# Show sample conversions
# --------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE PARSED VALUES")
print("=" * 60)

print(
    df[
        [
            "mileage",
            "mileage_value",
            "mileage_unit",
            "engine",
            "engine_cc",
            "max_power",
            "max_power_bhp",
            "torque",
            "torque_nm",
            "seats"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# --------------------------------------------------
# Identify parsing failures
# --------------------------------------------------

print("\n" + "=" * 60)
print("PARSING FAILURE ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# Max power parsing failures
# --------------------------------------------------

max_power_failures = df[
    df["max_power"].notna()
    & df["max_power_bhp"].isna()
]

print("\nMax power values that could not be parsed:")

print(
    max_power_failures["max_power"]
    .drop_duplicates()
    .to_string(index=False)
)

print("\nNumber of max power parsing failures:")
print(len(max_power_failures))


# --------------------------------------------------
# Torque parsing failures
# --------------------------------------------------

torque_failures = df[
    df["torque"].notna()
    & df["torque_nm"].isna()
]

print("\nTorque values that could not be parsed:")

print(
    torque_failures["torque"]
    .drop_duplicates()
    .to_string(index=False)
)

print("\nNumber of torque parsing failures:")
print(len(torque_failures))

# --------------------------------------------------
# Investigate unusual torque formats
# --------------------------------------------------

print("\n" + "=" * 60)
print("UNUSUAL TORQUE RECORD INVESTIGATION")
print("=" * 60)

failed_torque_values = [
    "250@ 1250-5000rpm",
    "48@ 3,000+/-500(NM@ rpm)",
    "510@ 1600-2400",
    "110(11.2)@ 4800",
    "210 / 1900"
]

for torque_value in failed_torque_values:

    print("\n" + "-" * 60)
    print(f"TORQUE VALUE: {torque_value}")
    print("-" * 60)

    matching_rows = df[
        df["torque"] == torque_value
    ]

    print(
        matching_rows[
            [
                "name",
                "year",
                "fuel",
                "seller_type",
                "transmission",
                "owner",
                "engine",
                "max_power",
                "torque"
            ]
        ].to_string(index=False)
    )

    print("\nNumber of occurrences:")
    print(len(matching_rows))

    # --------------------------------------------------
# Investigate max power parsing failures
# --------------------------------------------------

print("\n" + "=" * 60)
print("UNUSUAL MAX POWER RECORD INVESTIGATION")
print("=" * 60)

max_power_failure_values = (
    df.loc[
        df["max_power"].notna()
        & df["max_power_bhp"].isna(),
        "max_power"
    ]
    .drop_duplicates()
)

for power_value in max_power_failure_values:

    print("\n" + "-" * 60)
    print(f"MAX POWER VALUE: {repr(power_value)}")
    print("-" * 60)

    matching_rows = df[
        df["max_power"] == power_value
    ]

    print(
        matching_rows[
            [
                "name",
                "year",
                "fuel",
                "seller_type",
                "transmission",
                "owner",
                "engine",
                "max_power",
                "torque"
            ]
        ].to_string(index=False)
    )

    print("\nNumber of occurrences:")
    print(len(matching_rows))

    # --------------------------------------------------
# Final Parsing Validation
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL PARSING VALIDATION")
print("=" * 60)


# --------------------------------------------------
# Original missing values
# --------------------------------------------------

original_missing = {
    "mileage": df["mileage"].isnull().sum(),
    "engine": df["engine"].isnull().sum(),
    "max_power": df["max_power"].isnull().sum(),
    "torque": df["torque"].isnull().sum(),
    "seats": df["seats"].isnull().sum()
}


# --------------------------------------------------
# Parsed missing values
# --------------------------------------------------

parsed_missing = {
    "mileage_value": df["mileage_value"].isnull().sum(),
    "engine_cc": df["engine_cc"].isnull().sum(),
    "max_power_bhp": df["max_power_bhp"].isnull().sum(),
    "torque_nm": df["torque_nm"].isnull().sum(),
    "seats": df["seats"].isnull().sum()
}


print("\nOriginal missing values:")
print(original_missing)


print("\nParsed missing values:")
print(parsed_missing)


# --------------------------------------------------
# Expected vs actual missing values
# --------------------------------------------------

# --------------------------------------------------
# Account for known invalid max-power value
# --------------------------------------------------

invalid_max_power_count = len(
    df[
        df["max_power"].notna()
        & df["max_power_bhp"].isna()
    ]
)

expected_missing = {
    "mileage_value": original_missing["mileage"],
    "engine_cc": original_missing["engine"],
    "max_power_bhp": (
        original_missing["max_power"]
        + invalid_max_power_count
    ),
    "torque_nm": original_missing["torque"],
    "seats": original_missing["seats"]
}


print("\nExpected parsed missing values:")
print(expected_missing)


# --------------------------------------------------
# Validation
# --------------------------------------------------
print("\nKnown invalid max-power values:")
print(invalid_max_power_count)

print(
    "\nMax-power expected missing values = "
    f"{original_missing['max_power']} originally missing "
    f"+ {invalid_max_power_count} invalid value"
)

print("\n" + "-" * 60)
print("VALIDATION RESULTS")
print("-" * 60)


validation_passed = True


for column in expected_missing:

    expected = expected_missing[column]
    actual = parsed_missing[column]

    if expected == actual:

        print(
            f"PASS: {column} "
            f"({actual} missing values)"
        )

    else:

        print(
            f"FAIL: {column} "
            f"(expected {expected}, got {actual})"
        )

        validation_passed = False


# --------------------------------------------------
# Final result
# --------------------------------------------------

print("\n" + "=" * 60)

if validation_passed:

    print("PASS: FEATURE PARSING VALIDATION COMPLETED")

else:

    print("FAIL: FEATURE PARSING VALIDATION FAILED")

print("=" * 60)

# --------------------------------------------------
# Save Parsed Dataset
# --------------------------------------------------

parsed_output_path = "data/processed/car_data_parsed.csv"

df.to_csv(parsed_output_path, index=False)

print("\n" + "=" * 60)
print("PARSED DATASET SAVED")
print("=" * 60)

print("\nSaved dataset:")
print(parsed_output_path)

print("\nDataset shape:")
print(df.shape)