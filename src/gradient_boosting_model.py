import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_preprocessing import create_preprocessor
from prepare_ml_data import prepare_ml_data


# ============================================================
# 1. LOAD DATA
# ============================================================

X, y = prepare_ml_data()


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("=" * 70)
print("GRADIENT BOOSTING REGRESSOR")
print("=" * 70)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")


# ============================================================
# 3. PREPROCESSING
# ============================================================

preprocessor = create_preprocessor()


# ============================================================
# 4. GRADIENT BOOSTING MODEL
# ============================================================

gradient_boosting = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)


# ============================================================
# 5. COMPLETE PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", gradient_boosting)
    ]
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining Gradient Boosting Regressor...")

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 7. PREDICTIONS
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ============================================================
# 8. TRAINING METRICS
# ============================================================

train_mae = mean_absolute_error(y_train, y_train_pred)

train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)

train_r2 = r2_score(y_train, y_train_pred)


# ============================================================
# 9. TEST METRICS
# ============================================================

test_mae = mean_absolute_error(y_test, y_test_pred)

test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)

test_r2 = r2_score(y_test, y_test_pred)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING RESULTS")
print("=" * 70)

print("\nTraining Performance:")
print(f"MAE:  ₹{train_mae:,.2f}")
print(f"RMSE: ₹{train_rmse:,.2f}")
print(f"R²:   {train_r2:.4f}")

print("\nTesting Performance:")
print(f"MAE:  ₹{test_mae:,.2f}")
print(f"RMSE: ₹{test_rmse:,.2f}")
print(f"R²:   {test_r2:.4f}")


# ============================================================
# 11. OVERFITTING GAP
# ============================================================

print("\n" + "-" * 70)
print("OVERFITTING GAP")
print("-" * 70)

print(f"MAE gap:  ₹{test_mae - train_mae:,.2f}")
print(f"RMSE gap: ₹{test_rmse - train_rmse:,.2f}")
print(f"R² gap:   {train_r2 - test_r2:.4f}")


# ============================================================
# 12. CURRENT CHAMPION
# ============================================================

current_mae = 70200.55
current_rmse = 120388.55
current_r2 = 0.9339


print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT CHAMPION")
print("=" * 70)

print("\nCurrent Tuned Random Forest:")
print(f"MAE:  ₹{current_mae:,.2f}")
print(f"RMSE: ₹{current_rmse:,.2f}")
print(f"R²:   {current_r2:.4f}")

print("\nGradient Boosting:")
print(f"MAE:  ₹{test_mae:,.2f}")
print(f"RMSE: ₹{test_rmse:,.2f}")
print(f"R²:   {test_r2:.4f}")


# ============================================================
# 13. CHANGE FROM CURRENT CHAMPION
# ============================================================

mae_change = current_mae - test_mae
rmse_change = current_rmse - test_rmse
r2_change = test_r2 - current_r2


print("\n" + "-" * 70)
print("CHANGE FROM CURRENT CHAMPION")
print("-" * 70)

print(f"MAE improvement:  ₹{mae_change:,.2f}")
print(f"RMSE improvement: ₹{rmse_change:,.2f}")
print(f"R² improvement:   {r2_change:.4f}")


# ============================================================
# 14. FINAL COMPARISON
# ============================================================

print("\nModel comparison:")

if test_mae < current_mae:
    print("MAE:  Gradient Boosting is BETTER.")
else:
    print("MAE:  Gradient Boosting is WORSE.")

if test_rmse < current_rmse:
    print("RMSE: Gradient Boosting is BETTER.")
else:
    print("RMSE: Gradient Boosting is WORSE.")

if test_r2 > current_r2:
    print("R²:   Gradient Boosting is BETTER.")
else:
    print("R²:   Gradient Boosting is WORSE.")


print("\n" + "=" * 70)
print("GRADIENT BOOSTING ANALYSIS COMPLETED")
print("=" * 70)