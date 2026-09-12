import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
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
print("LOG-TARGET RANDOM FOREST")
print("=" * 70)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")


# ============================================================
# 3. CREATE PREPROCESSING + RANDOM FOREST PIPELINE
# ============================================================

preprocessor = create_preprocessor()

random_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", random_forest)
    ]
)


# ============================================================
# 4. LOG-TRANSFORM THE TARGET
# ============================================================

model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)


# ============================================================
# 5. TRAIN
# ============================================================

print("\nTraining Log-Target Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 6. PREDICTIONS
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Protect against tiny negative numerical values
y_train_pred = np.maximum(y_train_pred, 0)
y_test_pred = np.maximum(y_test_pred, 0)


# ============================================================
# 7. TRAINING METRICS
# ============================================================

train_mae = mean_absolute_error(y_train, y_train_pred)

train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)

train_r2 = r2_score(y_train, y_train_pred)


# ============================================================
# 8. TEST METRICS
# ============================================================

test_mae = mean_absolute_error(y_test, y_test_pred)

test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)

test_r2 = r2_score(y_test, y_test_pred)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOG-TARGET RANDOM FOREST RESULTS")
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
# 10. OVERFITTING GAP
# ============================================================

print("\n" + "-" * 70)
print("OVERFITTING GAP")
print("-" * 70)

print(f"MAE gap:  ₹{test_mae - train_mae:,.2f}")
print(f"RMSE gap: ₹{test_rmse - train_rmse:,.2f}")
print(f"R² gap:   {train_r2 - test_r2:.4f}")


# ============================================================
# 11. CURRENT CHAMPION
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

print("\nLog-Target Random Forest:")
print(f"MAE:  ₹{test_mae:,.2f}")
print(f"RMSE: ₹{test_rmse:,.2f}")
print(f"R²:   {test_r2:.4f}")


# ============================================================
# 12. CHANGE FROM CURRENT CHAMPION
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
# 13. FINAL COMPARISON
# ============================================================

print("\nModel comparison:")

if test_mae < current_mae:
    print("MAE:  Log-target model is BETTER.")
else:
    print("MAE:  Log-target model is WORSE.")

if test_rmse < current_rmse:
    print("RMSE: Log-target model is BETTER.")
else:
    print("RMSE: Log-target model is WORSE.")

if test_r2 > current_r2:
    print("R²:   Log-target model is BETTER.")
else:
    print("R²:   Log-target model is WORSE.")


print("\n" + "=" * 70)
print("LOG-TARGET RANDOM FOREST ANALYSIS COMPLETED")
print("=" * 70)