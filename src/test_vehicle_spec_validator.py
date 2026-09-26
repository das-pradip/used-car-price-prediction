"""
Test the vehicle specification validation module.
"""

from vehicle_spec_validator import (
    find_vehicle_spec,
    compare_specifications,
    create_specification_warning,
)


print("=" * 80)
print("VEHICLE SPECIFICATION VALIDATOR TEST")
print("=" * 80)


# ============================================================
# TEST 1 — KNOWN VEHICLE
# ============================================================

print("\n1. TEST KNOWN VEHICLE")

vehicle = find_vehicle_spec(
    name="Maruti Swift VDI BSIV",
    year=2014,
    fuel="Diesel",
    transmission="Manual",
)

if vehicle is None:

    print("FAIL: Vehicle not found.")

else:

    print("PASS: Vehicle found.")

    print(
        f"Vehicle: {vehicle['name']}"
    )

    print(
        f"Engine: {vehicle['engine_cc']} cc"
    )

    print(
        f"Power: {vehicle['max_power_bhp']} bhp"
    )

    print(
        f"Torque: {vehicle['torque_nm']} Nm"
    )

    print(
        f"Seats: {vehicle['seats']}"
    )


# ============================================================
# TEST 2 — MATCHING SPECIFICATIONS
# ============================================================

print("\n2. TEST MATCHING SPECIFICATIONS")

user_specs = {
    "mileage_value": 25.20,
    "engine_cc": 1248,
    "max_power_bhp": 74,
    "torque_nm": 190,
    "seats": 5,
}

differences = compare_specifications(
    user_specs,
    vehicle,
)

if differences:

    print("FAIL: Unexpected differences found.")

    print(differences)

else:

    print(
        "PASS: No specification differences."
    )


# ============================================================
# TEST 3 — INTENTIONALLY INCORRECT SPECIFICATIONS
# ============================================================

print("\n3. TEST INCORRECT SPECIFICATIONS")

bad_specs = {
    "mileage_value": 12,
    "engine_cc": 1200,
    "max_power_bhp": 222,
    "torque_nm": 55,
    "seats": 7,
}

differences = compare_specifications(
    bad_specs,
    vehicle,
)

print(
    f"Differences detected: {len(differences)}"
)

warning = create_specification_warning(
    differences
)

if warning:

    print("\nWARNING:")
    print("-" * 60)
    print(warning)

else:

    print(
        "FAIL: No warning was generated."
    )


# ============================================================
# TEST 4 — UNKNOWN VEHICLE
# ============================================================

print("\n4. TEST UNKNOWN VEHICLE")

unknown = find_vehicle_spec(
    name="Some Unknown Vehicle",
    year=2020,
    fuel="Petrol",
    transmission="Manual",
)

if unknown is None:

    print(
        "PASS: Unknown vehicle correctly "
        "returned no lookup result."
    )

else:

    print(
        "FAIL: Unknown vehicle should not "
        "have a lookup result."
    )


print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)