from pathlib import Path

import joblib
import pandas as pd


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = Path("models/used_car_price_model.joblib")

REFERENCE_YEAR = 2020

BRANDS = [
    "Ambassador",
    "Ashok",
    "Audi",
    "BMW",
    "Chevrolet",
    "Daewoo",
    "Datsun",
    "Fiat",
    "Force",
    "Ford",
    "Honda",
    "Hyundai",
    "Isuzu",
    "Jaguar",
    "Jeep",
    "Kia",
    "Land Rover",
    "Lexus",
    "MG",
    "Mahindra",
    "Maruti",
    "Mercedes-Benz",
    "Mitsubishi",
    "Nissan",
    "Opel",
    "Peugeot",
    "Renault",
    "Skoda",
    "Tata",
    "Toyota",
    "Volkswagen",
    "Volvo",
]

FUEL_TYPES = [
    "Petrol",
    "Diesel",
    "CNG",
    "LPG",
]

SELLER_TYPES = [
    "Individual",
    "Dealer",
    "Trustmark Dealer",
]

TRANSMISSION_TYPES = [
    "Manual",
    "Automatic",
]

OWNER_TYPES = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car",
]

SEAT_OPTIONS = [
    2,
    4,
    5,
    7,
    8,
    9,
    10,
    14,
]


# =========================================================
# Load model
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Please run src/save_final_model.py first."
    )

model = joblib.load(MODEL_PATH)


# =========================================================
# General helper functions
# =========================================================

def print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def get_integer(prompt, minimum=None, maximum=None):
    """Safely collect an integer."""

    while True:

        value = input(prompt).strip()

        try:
            number = int(value)
        except ValueError:
            print("❌ Please enter a whole number.")
            continue

        if minimum is not None and number < minimum:
            print(f"❌ Value must be at least {minimum}.")
            continue

        if maximum is not None and number > maximum:
            print(f"❌ Value must not exceed {maximum}.")
            continue

        return number


def get_float(prompt, minimum=None, maximum=None):
    """Safely collect a floating-point number."""

    while True:

        value = input(prompt).strip()

        try:
            number = float(value)
        except ValueError:
            print("❌ Please enter a valid number.")
            continue

        if minimum is not None and number < minimum:
            print(f"❌ Value must be at least {minimum}.")
            continue

        if maximum is not None and number > maximum:
            print(f"❌ Value must not exceed {maximum}.")
            continue

        return number


def choose_from_list(title, options):
    """Display numbered options and return selected value."""

    print(f"\n{title}")

    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")

    while True:

        choice = input("Enter option number: ").strip()

        try:
            choice = int(choice)
        except ValueError:
            print("❌ Please enter the option number.")
            continue

        if 1 <= choice <= len(options):
            return options[choice - 1]

        print(
            f"❌ Please choose a number between 1 and {len(options)}."
        )


def ask_confirmation(prompt):
    """Ask the user for yes/no confirmation."""

    while True:

        answer = input(prompt).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("❌ Please enter y or n.")


# =========================================================
# Technical specification helpers
# =========================================================

def show_technical_help():

    print(
        "\nWhere can you find these specifications?"
        "\n"
        "\n• Vehicle registration/RC documents"
        "\n• Owner's manual"
        "\n• Original vehicle brochure"
        "\n• Manufacturer's official specification page"
        "\n• Vehicle specification websites"
        "\n"
    )


def get_mileage():

    print("\n" + "-" * 72)
    print("MILEAGE")
    print("-" * 72)

    print(
        "Mileage means the approximate distance the vehicle can travel "
        "using one litre of fuel."
    )
    print("Unit: km/l")
    print("Example: 18.5 km/l")
    print("Typical dataset median: about 19.56 km/l")

    show_technical_help()

    while True:

        mileage = get_float(
            "Enter mileage (km/l): ",
            minimum=0.1,
        )

        if mileage > 30:

            print(
                "\n⚠️ WARNING"
                "\nThis mileage is unusually high compared with "
                "most vehicles in the training dataset."
            )

            print(f"Entered value: {mileage:.2f} km/l")
            print("99th percentile of dataset: about 28.40 km/l")

            if not ask_confirmation(
                "Please verify the value. Continue? (y/n): "
            ):
                continue

        return mileage


def get_engine_cc():

    print("\n" + "-" * 72)
    print("ENGINE CAPACITY")
    print("-" * 72)

    print(
        "Engine capacity describes the size/displacement of the engine."
    )
    print("Unit: CC (cubic centimetres)")
    print("Example: 1197 CC → enter 1197")
    print("Dataset median: 1248 CC")
    print("Common examples in the dataset: 796, 1197, 1248, 1498 CC")

    show_technical_help()

    while True:

        engine = get_float(
            "Enter engine capacity (CC): ",
            minimum=1,
        )

        if engine < 500:

            print(
                "\n⚠️ WARNING"
                "\nThis engine capacity is extremely unusual "
                "for the vehicles represented in this dataset."
            )

            print(f"Entered value: {engine:.0f} CC")
            print("Dataset minimum: 624 CC")

            if not ask_confirmation(
                "Did you enter the correct engine capacity? (y/n): "
            ):
                continue

        elif engine > 3000:

            print(
                "\n⚠️ WARNING"
                "\nThis engine capacity is in the upper range "
                "of the training dataset."
            )

            print(f"Entered value: {engine:.0f} CC")
            print("99th percentile: about 2956 CC")
            print("Dataset maximum: 3604 CC")

            if not ask_confirmation(
                "Please verify the value. Continue? (y/n): "
            ):
                continue

        return engine


def get_max_power():

    print("\n" + "-" * 72)
    print("MAXIMUM POWER")
    print("-" * 72)

    print(
        "Maximum power represents the engine's rated power output."
    )
    print("Unit: BHP (brake horsepower)")
    print("Example: 74 BHP → enter 74")
    print("Dataset median: about 81.83 BHP")

    show_technical_help()

    while True:

        power = get_float(
            "Enter maximum power (BHP): ",
            minimum=0.1,
        )

        if power < 20:

            print(
                "\n⚠️ WARNING"
                "\nThis power value is extremely unusual "
                "for the vehicles represented in this dataset."
            )

            print(f"Entered value: {power:.2f} BHP")
            print("Dataset 1st percentile: about 37 BHP")

            if not ask_confirmation(
                "Did you enter the correct power value? (y/n): "
            ):
                continue

        elif power > 200:

            print(
                "\n⚠️ WARNING"
                "\nThis power value is unusually high compared "
                "with most vehicles in the training dataset."
            )

            print(f"Entered value: {power:.2f} BHP")
            print("99th percentile: about 190 BHP")
            print("Dataset maximum: 400 BHP")

            if not ask_confirmation(
                "Please verify the value. Continue? (y/n): "
            ):
                continue

        return power


def get_torque():

    print("\n" + "-" * 72)
    print("TORQUE")
    print("-" * 72)

    print(
        "Torque describes the engine's twisting force."
    )
    print("Unit: Nm (Newton-metre)")
    print("Example: 190 Nm → enter 190")
    print("Dataset median: about 160 Nm")

    show_technical_help()

    while True:

        torque = get_float(
            "Enter torque (Nm): ",
            minimum=0.1,
        )

        if torque < 40:

            print(
                "\n⚠️ WARNING"
                "\nThis torque value is extremely unusual "
                "for the vehicles represented in this dataset."
            )

            print(f"Entered value: {torque:.2f} Nm")
            print("Dataset minimum: 47.07 Nm")

            if not ask_confirmation(
                "Did you enter the correct torque value? (y/n): "
            ):
                continue

        elif torque > 450:

            print(
                "\n⚠️ WARNING"
                "\nThis torque value is unusually high compared "
                "with most vehicles in the training dataset."
            )

            print(f"Entered value: {torque:.2f} Nm")
            print("99th percentile: about 420 Nm")
            print("Dataset maximum: 789 Nm")

            if not ask_confirmation(
                "Please verify the value. Continue? (y/n): "
            ):
                continue

        return torque


def get_seats():

    print("\n" + "-" * 72)
    print("NUMBER OF SEATS")
    print("-" * 72)

    print(
        "Select the seating capacity of the vehicle."
    )

    return choose_from_list(
        "Number of seats:",
        SEAT_OPTIONS,
    )


# =========================================================
# Basic vehicle inputs
# =========================================================

def get_year():

    while True:

        year = get_integer(
            "Manufacturing year (1983-2020): ",
            minimum=1983,
            maximum=2020,
        )

        return year


def get_km_driven():

    while True:

        km = get_integer(
            "Kilometres driven: ",
            minimum=0,
        )

        if km > 239863:

            print(
                "\n⚠️ WARNING"
                "\nThis mileage is above approximately the "
                "99th percentile of the training dataset."
            )

            print(f"Entered value: {km:,} km")
            print("99th percentile: about 239,863 km")

            if not ask_confirmation(
                "Please verify the value. Continue? (y/n): "
            ):
                continue

        return km


# =========================================================
# Main prediction workflow
# =========================================================

def main():

    print_header("USED CAR PRICE PREDICTION")

    print(
        "\nPlease enter the vehicle information below."
    )

    print(
        "\nIMPORTANT:"
        "\n• Enter specifications as accurately as possible."
        "\n• Mileage should be in km/l."
        "\n• Engine capacity should be in CC."
        "\n• Power should be in BHP."
        "\n• Torque should be in Nm."
    )

    # -----------------------------------------------------
    # Basic vehicle information
    # -----------------------------------------------------

    print_header("1. BASIC VEHICLE INFORMATION")

    brand = choose_from_list(
        "Select vehicle brand:",
        BRANDS,
    )

    year = get_year()

    km_driven = get_km_driven()

    fuel = choose_from_list(
        "Select fuel type:",
        FUEL_TYPES,
    )

    seller_type = choose_from_list(
        "Select seller type:",
        SELLER_TYPES,
    )

    transmission = choose_from_list(
        "Select transmission:",
        TRANSMISSION_TYPES,
    )

    owner = choose_from_list(
        "Select ownership status:",
        OWNER_TYPES,
    )

    # -----------------------------------------------------
    # Technical information
    # -----------------------------------------------------

    print_header("2. VEHICLE SPECIFICATIONS")

    print(
        "\nThese specifications are important because the "
        "current production model uses them directly."
    )

    print(
        "\nIf you don't know these values, don't guess."
        "\nCheck your RC, owner's manual, vehicle brochure, "
        "or manufacturer specification page."
    )

    mileage = get_mileage()

    engine_cc = get_engine_cc()

    max_power_bhp = get_max_power()

    torque_nm = get_torque()

    seats = get_seats()

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    car_age = REFERENCE_YEAR - year

    input_data = pd.DataFrame(
        [
            {
                "car_age": car_age,
                "km_driven": km_driven,
                "mileage_value": mileage,
                "engine_cc": engine_cc,
                "max_power_bhp": max_power_bhp,
                "torque_nm": torque_nm,
                "seats": seats,
                "brand": brand,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
            }
        ]
    )

    # -----------------------------------------------------
    # Input summary
    # -----------------------------------------------------

    print_header("3. INPUT SUMMARY")

    print(f"Brand             : {brand}")
    print(f"Manufacturing year: {year}")
    print(f"Car age           : {car_age} years")
    print(f"Kilometres driven : {km_driven:,} km")
    print(f"Fuel              : {fuel}")
    print(f"Seller type       : {seller_type}")
    print(f"Transmission      : {transmission}")
    print(f"Owner             : {owner}")
    print(f"Mileage           : {mileage:.2f} km/l")
    print(f"Engine            : {engine_cc:.0f} CC")
    print(f"Maximum power     : {max_power_bhp:.2f} BHP")
    print(f"Torque            : {torque_nm:.2f} Nm")
    print(f"Seats             : {seats}")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    print_header("4. PRICE PREDICTION")

    print(
        f"\nEstimated used-car price: ₹{prediction:,.0f}"
    )

    print(
        "\nNote: This is a machine-learning estimate based on "
        "the training dataset. Actual market prices can differ "
        "depending on vehicle condition, location, variant, "
        "service history, and other factors."
    )

    print("\n" + "=" * 72)
    print("Prediction completed successfully.")
    print("=" * 72)


# =========================================================
# Program entry point
# =========================================================

if __name__ == "__main__":
    main()