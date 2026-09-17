import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/processed/car_data_feature_engineered.csv"

TARGET = "selling_price"

NUMERICAL_FEATURES = [
    "car_age",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
]

BASE_CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]


def extract_model_family(name):
    parts = str(name).split()

    if len(parts) < 2:
        return str(name)

    if parts[0] == "Land" and len(parts) >= 3 and parts[1] == "Rover":
        brand = "Land Rover"
        remaining = parts[2:]
    elif parts[0] == "Mercedes-Benz":
        brand = "Mercedes-Benz"
        remaining = parts[1:]
    elif parts[0] == "Rolls-Royce":
        brand = "Rolls-Royce"
        remaining = parts[1:]
    else:
        brand = parts[0]
        remaining = parts[1:]

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

    return f"{brand} {remaining[0]}"


def build_pipeline(numerical_features, categorical_features):
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
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


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE:  ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    return mae, rmse, r2


def main():
    df = pd.read_csv(DATA_PATH)

    df["model_family"] = df["name"].apply(extract_model_family)

    y = df[TARGET]

    # ---------------------------------------------------------
    # Same split for every experiment
    # ---------------------------------------------------------

    train_indices, test_indices = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42
    )

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    # ---------------------------------------------------------
    # Model A: model family only
    # ---------------------------------------------------------

    X_train_model_only = train_df[["model_family"]]
    X_test_model_only = test_df[["model_family"]]

    model_only = build_pipeline(
        [],
        ["model_family"]
    )

    evaluate_model(
        "MODEL A — MODEL FAMILY ONLY",
        model_only,
        X_train_model_only,
        X_test_model_only,
        y_train,
        y_test
    )

    # ---------------------------------------------------------
    # Model B: existing features only
    # ---------------------------------------------------------

    X_train_base = train_df[
        NUMERICAL_FEATURES + BASE_CATEGORICAL_FEATURES
    ]

    X_test_base = test_df[
        NUMERICAL_FEATURES + BASE_CATEGORICAL_FEATURES
    ]

    base_model = build_pipeline(
        NUMERICAL_FEATURES,
        BASE_CATEGORICAL_FEATURES
    )

    evaluate_model(
        "MODEL B — EXISTING FEATURES",
        base_model,
        X_train_base,
        X_test_base,
        y_train,
        y_test
    )

    # ---------------------------------------------------------
    # Model C: existing features + model family
    # ---------------------------------------------------------

    X_train_combined = train_df[
        NUMERICAL_FEATURES
        + BASE_CATEGORICAL_FEATURES
        + ["model_family"]
    ]

    X_test_combined = test_df[
        NUMERICAL_FEATURES
        + BASE_CATEGORICAL_FEATURES
        + ["model_family"]
    ]

    combined_model = build_pipeline(
        NUMERICAL_FEATURES,
        BASE_CATEGORICAL_FEATURES + ["model_family"]
    )

    evaluate_model(
        "MODEL C — EXISTING FEATURES + MODEL FAMILY",
        combined_model,
        X_train_combined,
        X_test_combined,
        y_train,
        y_test
    )

    print("\n" + "=" * 70)
    print("SIGNAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()