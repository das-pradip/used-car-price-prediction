import pandas as pd


DATA_PATH = "data/processed/car_data_feature_engineered.csv"


def extract_brand_model(name: str) -> str:
    """
    Extract the first two meaningful words as brand + model.
    Handles common multi-word brands explicitly.
    """
    parts = str(name).split()

    if len(parts) < 2:
        return str(name)

    # Handle multi-word brands
    if parts[0] in {"Land", "Rolls-Royce", "Mercedes-Benz"} and parts[0] == "Land":
        return "Land Rover " + " ".join(parts[2:3])

    if parts[0] == "Rolls-Royce":
        return "Rolls-Royce " + parts[1]

    if parts[0] == "Mercedes-Benz":
        return "Mercedes-Benz " + parts[1]

    return f"{parts[0]} {parts[1]}"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("CAR MODEL PRICE ANALYSIS")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")

    # Create brand + model
    df["brand_model"] = df["name"].apply(extract_brand_model)

    print(f"Unique brand-model combinations: {df['brand_model'].nunique()}")

    # Price statistics for each brand-model
    model_stats = (
        df.groupby("brand_model")["selling_price"]
        .agg(
            car_count="count",
            mean_price="mean",
            median_price="median",
            min_price="min",
            max_price="max",
        )
        .sort_values("car_count", ascending=False)
    )

    print("\nTop 30 brand-model combinations by number of cars:")
    print(model_stats.head(30).to_string())

    print("\nModels with only one observation:")
    print((model_stats["car_count"] == 1).sum())

    print("\nModels with 2 or fewer observations:")
    print((model_stats["car_count"] <= 2).sum())

    print("\nModels with 5 or more observations:")
    print((model_stats["car_count"] >= 5).sum())

    print("\nMost expensive models by median price (minimum 5 cars):")
    expensive_models = (
        model_stats[model_stats["car_count"] >= 5]
        .sort_values("median_price", ascending=False)
        .head(20)
    )

    print(expensive_models.to_string())

    print("\nMost common models:")
    print(model_stats.head(20).to_string())

    print("\nModel frequency distribution:")

    frequency_thresholds = [1, 2, 5, 10, 20, 50, 100]

    for threshold in frequency_thresholds:
        model_count = (model_stats["car_count"] >= threshold).sum()
        car_count = model_stats.loc[
            model_stats["car_count"] >= threshold,
            "car_count"
        ].sum()

        print(
            f"Models with >= {threshold:3} cars: "
            f"{model_count:3} models | "
            f"{car_count:4} cars "
            f"({car_count / len(df) * 100:.2f}%)"
        )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()