import pandas as pd
from collections import Counter


DATA_PATH = "data/processed/car_data_feature_engineered.csv"


# Tokens that usually represent variants/specifications,
# rather than the core vehicle model.
SPECIFICATION_TOKENS = {
    "BS",
    "BSII",
    "BSIII",
    "BSIV",
    "BSVI",
    "Diesel",
    "Petrol",
    "CNG",
    "LPG",
    "VXI",
    "VDI",
    "VDi",
    "LXI",
    "LXi",
    "LDI",
    "ZDI",
    "ZDi",
    "ZXI",
    "ZX",
    "SX",
    "SXO",
    "S",
    "G",
    "V",
    "VX",
    "LX",
    "EX",
    "E",
    "L",
    "AT",
    "AMT",
    "MT",
    "CVT",
    "ABS",
    "AC",
    "Option",
    "Optional",
    "Plus",
    "Edition",
    "Seater",
    "Str",
    "2WD",
    "4WD",
    "AWD",
    "TDI",
    "CRDi",
    "TDCi",
    "Kappa",
    "Revotron",
    "Revotorq",
    "Quadrajet",
    "mFALCON",
    "D75",
    "G80",
    "Highline",
    "Sportz",
    "Magna",
    "Asta",
    "Titanium",
    "Era",
    "RXT",
    "R",
    "Xing",
}


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("MODEL FAMILY CANDIDATE ANALYSIS")
    print("=" * 70)

    print(f"\nDataset rows: {len(df)}")
    print(f"Unique car names: {df['name'].nunique()}")

    # ---------------------------------------------------------
    # Count tokens after removing brand and obvious specs
    # ---------------------------------------------------------

    model_tokens = []

    for name in df["name"]:
        parts = str(name).split()

        # Remove first token because it represents the brand
        remaining = parts[1:]

        for token in remaining:
            if token not in SPECIFICATION_TOKENS:
                model_tokens.append(token)

    token_counts = Counter(model_tokens)

    print("\nTop candidate model tokens:")
    print("-" * 70)

    for token, count in token_counts.most_common(100):
        print(f"{token:<30} {count:>5}")

    # ---------------------------------------------------------
    # Candidate model token coverage
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("HIGH-FREQUENCY MODEL TOKEN COVERAGE")
    print("=" * 70)

    thresholds = [5, 10, 20, 50, 100]

    for threshold in thresholds:
        candidates = {
            token
            for token, count in token_counts.items()
            if count >= threshold
        }

        matching_rows = 0

        for name in df["name"]:
            parts = str(name).split()[1:]

            if any(token in candidates for token in parts):
                matching_rows += 1

        print(
            f"Tokens >= {threshold:3} occurrences: "
            f"{len(candidates):3} tokens | "
            f"{matching_rows:4} rows "
            f"({matching_rows / len(df) * 100:.2f}%)"
        )

    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()