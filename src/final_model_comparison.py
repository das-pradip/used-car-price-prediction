import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor
import numpy as np


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
]

TARGET = "selling_price"


def create_preprocessor():
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
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ]
    )


def create_pipeline(regressor):
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("regressor", regressor),
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

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # ---------------------------------------------------------
    # Same train/test split used throughout the project
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    results = []

    # ---------------------------------------------------------
    # 1. Mean baseline
    # ---------------------------------------------------------

    mean_prediction = y_train.mean()

    mean_predictions = np.full(
        len(y_test),
        mean_prediction
    )

    results.append(
        {
            "model": "Mean Baseline",
            "MAE": mean_absolute_error(
                y_test,
                mean_predictions
            ),
            "RMSE": mean_squared_error(
                y_test,
                mean_predictions
            ) ** 0.5,
            "R2": r2_score(
                y_test,
                mean_predictions
            ),
        }
    )

    # ---------------------------------------------------------
    # 2. Median baseline
    # ---------------------------------------------------------

    median_prediction = y_train.median()

    median_predictions = np.full(
        len(y_test),
        median_prediction
    )

    results.append(
        {
            "model": "Median Baseline",
            "MAE": mean_absolute_error(
                y_test,
                median_predictions
            ),
            "RMSE": mean_squared_error(
                y_test,
                median_predictions
            ) ** 0.5,
            "R2": r2_score(
                y_test,
                median_predictions
            ),
        }
    )

    # ---------------------------------------------------------
    # 3. Linear Regression
    # ---------------------------------------------------------

    linear_regression = create_pipeline(
        LinearRegression()
    )

    results.append(
        evaluate_model(
            "Linear Regression",
            linear_regression,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ---------------------------------------------------------
    # 4. Decision Tree
    # ---------------------------------------------------------

    decision_tree = create_pipeline(
        DecisionTreeRegressor(
            random_state=42
        )
    )

    results.append(
        evaluate_model(
            "Decision Tree",
            decision_tree,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ---------------------------------------------------------
    # 5. Original Random Forest
    # ---------------------------------------------------------

    random_forest = create_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )

    results.append(
        evaluate_model(
            "Random Forest",
            random_forest,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ---------------------------------------------------------
    # 6. Tuned Random Forest
    # ---------------------------------------------------------

    tuned_random_forest = create_pipeline(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    )

    results.append(
        evaluate_model(
            "Tuned Random Forest",
            tuned_random_forest,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ---------------------------------------------------------
    # 7. Gradient Boosting
    # ---------------------------------------------------------

    gradient_boosting = create_pipeline(
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42
        )
    )

    results.append(
        evaluate_model(
            "Gradient Boosting",
            gradient_boosting,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ---------------------------------------------------------
    # 8. Log-target Random Forest
    # ---------------------------------------------------------

    log_target_rf = create_pipeline(
        TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )
    )

    results.append(
        evaluate_model(
            "Log-Target Random Forest",
            log_target_rf,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "MAE",
        ascending=True
    ).reset_index(drop=True)

    print("=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print("\nTest-set performance:")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "MAE": lambda x: f"₹{x:,.2f}",
                "RMSE": lambda x: f"₹{x:,.2f}",
                "R2": lambda x: f"{x:.4f}",
            }
        )
    )

    print("\n" + "=" * 80)
    print("CURRENT CHAMPION")
    print("=" * 80)

    champion = results_df.iloc[0]

    print(f"\nModel: {champion['model']}")
    print(f"MAE:   ₹{champion['MAE']:,.2f}")
    print(f"RMSE:  ₹{champion['RMSE']:,.2f}")
    print(f"R²:    {champion['R2']:.4f}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()