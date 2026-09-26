from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOKUP_PATH = PROJECT_ROOT / "data" / "processed" / "vehicle_spec_lookup.csv"


TECHNICAL_COLUMNS = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]


def load_vehicle_lookup():
    """
    Load the vehicle specification lookup table.

    Returns
    -------
    pandas.DataFrame
        Vehicle specification lookup data.
    """
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"Vehicle specification lookup not found: {LOOKUP_PATH}"
        )

    return pd.read_csv(LOOKUP_PATH)


def find_vehicle_spec(name, year, fuel, transmission, lookup_df=None):
    """
    Find a known specification for an exact vehicle configuration.

    The lookup key is:
        name + year + fuel + transmission

    Returns
    -------
    dict or None
        Reference specifications if a matching lookup record exists.
    """

    if lookup_df is None:
        lookup_df = load_vehicle_lookup()

    matches = lookup_df[
        (lookup_df["name"] == name)
        & (lookup_df["year"] == year)
        & (lookup_df["fuel"] == fuel)
        & (lookup_df["transmission"] == transmission)
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]

    return {
    "name": row["name"],
    "year": row["year"],
    "fuel": row["fuel"],
    "transmission": row["transmission"],
    "mileage_value": row["mileage_value"],
    "engine_cc": row["engine_cc"],
    "max_power_bhp": row["max_power_bhp"],
    "torque_nm": row["torque_nm"],
    "seats": row["seats"],
    "row_count": row["row_count"],
}


def _is_missing(value):
    """Return True if a value is missing."""
    return pd.isna(value)


def _within_tolerance(user_value, reference_value, absolute_tolerance, percentage_tolerance):
    """
    Check whether a user-provided numerical value is reasonably close
    to the reference value.

    The allowed tolerance is the larger of:
        absolute tolerance
        percentage tolerance
    """

    if _is_missing(user_value) or _is_missing(reference_value):
        return True

    allowed_difference = max(
        absolute_tolerance,
        abs(reference_value) * percentage_tolerance,
    )

    return abs(user_value - reference_value) <= allowed_difference


def compare_specifications(user_specs, reference_specs):
    """
    Compare user-entered specifications against reference specifications.

    Returns
    -------
    list of dict
        Differences found between user and reference specifications.
    """

    differences = []

    # Mileage
    if not _within_tolerance(
        user_specs.get("mileage_value"),
        reference_specs.get("mileage_value"),
        absolute_tolerance=1.0,
        percentage_tolerance=0.05,
    ):
        differences.append(
            {
                "field": "mileage_value",
                "user_value": user_specs.get("mileage_value"),
                "reference_value": reference_specs.get("mileage_value"),
                "message": "Mileage differs substantially from the known specification.",
            }
        )

    # Engine
    if not _within_tolerance(
        user_specs.get("engine_cc"),
        reference_specs.get("engine_cc"),
        absolute_tolerance=50,
        percentage_tolerance=0.05,
    ):
        differences.append(
            {
                "field": "engine_cc",
                "user_value": user_specs.get("engine_cc"),
                "reference_value": reference_specs.get("engine_cc"),
                "message": "Engine capacity differs substantially from the known specification.",
            }
        )

    # Power
    if not _within_tolerance(
        user_specs.get("max_power_bhp"),
        reference_specs.get("max_power_bhp"),
        absolute_tolerance=5,
        percentage_tolerance=0.05,
    ):
        differences.append(
            {
                "field": "max_power_bhp",
                "user_value": user_specs.get("max_power_bhp"),
                "reference_value": reference_specs.get("max_power_bhp"),
                "message": "Maximum power differs substantially from the known specification.",
            }
        )

    # Torque
    if not _within_tolerance(
        user_specs.get("torque_nm"),
        reference_specs.get("torque_nm"),
        absolute_tolerance=10,
        percentage_tolerance=0.05,
    ):
        differences.append(
            {
                "field": "torque_nm",
                "user_value": user_specs.get("torque_nm"),
                "reference_value": reference_specs.get("torque_nm"),
                "message": "Torque differs substantially from the known specification.",
            }
        )

    # Seats
    user_seats = user_specs.get("seats")
    reference_seats = reference_specs.get("seats")

    if not _is_missing(user_seats) and not _is_missing(reference_seats):
        if int(user_seats) != int(reference_seats):
            differences.append(
                {
                    "field": "seats",
                    "user_value": user_seats,
                    "reference_value": reference_seats,
                    "message": "Number of seats differs from the known specification.",
                }
            )

    return differences


def format_spec_name(field):
    """Convert internal feature names into user-friendly names."""

    names = {
        "mileage_value": "Mileage",
        "engine_cc": "Engine",
        "max_power_bhp": "Maximum Power",
        "torque_nm": "Torque",
        "seats": "Seats",
    }

    return names.get(field, field)


def create_specification_warning(differences):
    """
    Create a user-friendly warning from specification differences.

    Returns
    -------
    str or None
        Warning text, or None if there are no differences.
    """

    if not differences:
        return None

    lines = [
        "WARNING: Some specifications do not match the known vehicle data.",
        "",
        "Field comparison:",
    ]

    for difference in differences:
        field = format_spec_name(difference["field"])
        user_value = difference["user_value"]
        reference_value = difference["reference_value"]

        lines.append(
            f"- {field}: entered={user_value}, "
            f"reference={reference_value}"
        )

    lines.extend(
        [
            "",
            "Please verify the specifications from the vehicle's "
            "registration documents, owner's manual, or manufacturer data.",
        ]
    )

    return "\n".join(lines)