import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/processed/car_data_feature_engineered.csv"

NUMERICAL_FEATURES = [
    "car_age",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]

CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "model_family",
]

TARGET = "selling_price"


def extract_model_family(name):
    parts = str(name).split()

    if len(parts) < 2:
        return str(name)

    # Multi-word brands
    if parts[0] == "Land" and len(parts) >= 3 and parts[1] == "Rover":
        start = 3
        brand = "Land Rover"
    elif parts[0] == "Mercedes-Benz":
        start = 1
        brand = "Mercedes-Benz"
    elif parts[0] == "Rolls-Royce":
        start = 1
        brand = "Rolls-Royce"
    else:
        start = 1
        brand = parts[0]

    remaining = parts[start:]

    if not remaining:
        return brand

    # Known multi-token model families
    model_patterns = [
        ["Swift", "Dzire"],
        ["Alto", "800"],
        ["Alto", "K10"],
        ["Wagon", "R"],
        ["Grand", "i10"],
        ["Innova", "Crysta"],
        ["Etios", "Liva"],
        ["Indica", "Vista"],
        ["Indica", "V2"],
        ["Indigo", "CS"],
        ["Vitara", "Brezza"],
        ["Zen", "Estilo"],
        ["Figo", "Aspire"],
        ["New", "Safari"],
        ["KUV", "100"],
        ["TUV", "300"],
        ["SX4", "S"],
        ["S", "Cross"],
    ]

    for pattern in model_patterns:
        if remaining[:len(pattern)] == pattern:
            return f"{brand} {' '.join(pattern)}"

    # Default: first model token
    return f"{brand} {remaining[0]}"


def build_preprocessor():
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)

    df["model_family"] = df["name"].apply(extract_model_family)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1
                )
            ),
        ]
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_predictions)
    train_rmse = mean_squared_error(
        y_train,
        train_predictions
    ) ** 0.5
    train_r2 = r2_score(y_train, train_predictions)

    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = mean_squared_error(
        y_test,
        test_predictions
    ) ** 0.5
    test_r2 = r2_score(y_test, test_predictions)

    print("=" * 70)
    print("MODEL FAMILY EXPERIMENT")
    print("=" * 70)

    print(f"\nDataset rows: {len(df)}")
    print(f"Model family categories: {df['model_family'].nunique()}")

    print("\nTop model families:")
    print(
        df["model_family"]
        .value_counts()
        .head(30)
        .to_string()
    )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print("\nTraining:")
    print(f"MAE:  ₹{train_mae:,.2f}")
    print(f"RMSE: ₹{train_rmse:,.2f}")
    print(f"R²:   {train_r2:.4f}")

    print("\nTesting:")
    print(f"MAE:  ₹{test_mae:,.2f}")
    print(f"RMSE: ₹{test_rmse:,.2f}")
    print(f"R²:   {test_r2:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT CHAMPION")
    print("=" * 70)

    champion_mae = 70200.55
    champion_rmse = 120388.55
    champion_r2 = 0.9339

    print(f"\nChampion MAE:  ₹{champion_mae:,.2f}")
    print(f"New MAE:       ₹{test_mae:,.2f}")
    print(f"MAE change:    ₹{test_mae - champion_mae:,.2f}")

    print(f"\nChampion RMSE: ₹{champion_rmse:,.2f}")
    print(f"New RMSE:      ₹{test_rmse:,.2f}")
    print(f"RMSE change:   ₹{test_rmse - champion_rmse:,.2f}")

    print(f"\nChampion R²:   {champion_r2:.4f}")
    print(f"New R²:        {test_r2:.4f}")
    print(f"R² change:     {test_r2 - champion_r2:+.4f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()