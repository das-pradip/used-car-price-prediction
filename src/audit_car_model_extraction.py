import pandas as pd


DATA_PATH = "data/processed/car_data_feature_engineered.csv"


def extract_brand_model(name: str) -> str:
    """
    Extract brand + model prefix from the car name.
    Handles known multi-word brands.
    """
    parts = str(name).split()

    if len(parts) < 2:
        return str(name)

    if parts[0] == "Land" and len(parts) >= 3 and parts[1] == "Rover":
        return "Land Rover " + parts[2]

    if parts[0] == "Rolls-Royce":
        return "Rolls-Royce " + parts[1]

    if parts[0] == "Mercedes-Benz":
        return "Mercedes-Benz " + parts[1]

    return f"{parts[0]} {parts[1]}"


def main():
    df = pd.read_csv(DATA_PATH)

    df["brand_model"] = df["name"].apply(extract_brand_model)

    model_audit = (
        df.groupby("brand_model")
        .agg(
            car_count=("name", "count"),
            unique_original_names=("name", "nunique"),
        )
        .sort_values(
            ["unique_original_names", "car_count"],
            ascending=[False, False],
        )
    )

    print("=" * 70)
    print("CAR MODEL EXTRACTION AUDIT")
    print("=" * 70)

    print(f"\nDataset rows: {len(df)}")
    print(f"Extracted brand-model categories: {df['brand_model'].nunique()}")

    print("\nModels mapping to multiple original names:")
    print(
        model_audit[
            model_audit["unique_original_names"] > 1
        ].head(40).to_string()
    )

    print("\n" + "=" * 70)
    print("EXAMPLES OF AMBIGUOUS EXTRACTIONS")
    print("=" * 70)

    ambiguous_models = model_audit[
        model_audit["unique_original_names"] > 1
    ].head(20)

    for brand_model in ambiguous_models.index:
        print(f"\n{brand_model}")

        names = (
            df.loc[
                df["brand_model"] == brand_model,
                "name"
            ]
            .drop_duplicates()
            .tolist()
        )

        for name in names[:10]:
            print(f"  - {name}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()