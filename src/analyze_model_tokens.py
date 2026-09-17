import pandas as pd
from collections import Counter


DATA_PATH = "data/processed/car_data_feature_engineered.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("MODEL TOKEN ANALYSIS")
    print("=" * 70)

    print(f"\nDataset rows: {len(df)}")
    print(f"Unique car names: {df['name'].nunique()}")

    # ---------------------------------------------------------
    # Token analysis
    # ---------------------------------------------------------

    tokens = []

    for name in df["name"]:
        parts = str(name).split()

        # Remove brand token
        if len(parts) > 1:
            tokens.extend(parts[1:])

    token_counts = Counter(tokens)

    print("\nMost common tokens after removing brand:")

    for token, count in token_counts.most_common(100):
        print(f"{token:<25} {count:>5}")

    # ---------------------------------------------------------
    # Second + third token combinations
    # ---------------------------------------------------------

    combinations = []

    for name in df["name"]:
        parts = str(name).split()

        if len(parts) >= 3:
            combinations.append(f"{parts[1]} {parts[2]}")

    combination_counts = Counter(combinations)

    print("\n" + "=" * 70)
    print("MOST COMMON SECOND + THIRD TOKEN COMBINATIONS")
    print("=" * 70)

    for combination, count in combination_counts.most_common(50):
        print(f"{combination:<35} {count:>5}")

    # ---------------------------------------------------------
    # Full model-name examples
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXAMPLES OF LONG MODEL NAMES")
    print("=" * 70)

    word_counts = df["name"].str.split().str.len()

    long_names = (
        df.loc[word_counts >= 7, "name"]
        .drop_duplicates()
        .head(50)
    )

    for name in long_names:
        print(name)

    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()