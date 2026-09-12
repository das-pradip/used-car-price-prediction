import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/car_data_feature_engineered.csv"

df = pd.read_csv(DATA_PATH)

price = df["selling_price"]


# ============================================================
# 2. BASIC STATISTICS
# ============================================================

print("=" * 70)
print("SELLING PRICE DISTRIBUTION ANALYSIS")
print("=" * 70)

print(f"Number of cars: {len(price)}")
print(f"Minimum price: ₹{price.min():,.2f}")
print(f"Maximum price: ₹{price.max():,.2f}")
print(f"Mean price: ₹{price.mean():,.2f}")
print(f"Median price: ₹{price.median():,.2f}")
print(f"Standard deviation: ₹{price.std():,.2f}")

print("\nPrice percentiles:")

percentiles = price.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

for percentile, value in percentiles.items():
    print(f"{percentile * 100:.0f}th percentile: ₹{value:,.2f}")


# ============================================================
# 3. SKEWNESS
# ============================================================

skewness = price.skew()

print("\n" + "-" * 70)
print("SKEWNESS")
print("-" * 70)

print(f"Selling price skewness: {skewness:.4f}")

if skewness > 1:
    print("Interpretation: Strong right skew.")
elif skewness > 0.5:
    print("Interpretation: Moderate right skew.")
elif skewness < -1:
    print("Interpretation: Strong left skew.")
elif skewness < -0.5:
    print("Interpretation: Moderate left skew.")
else:
    print("Interpretation: Approximately symmetric.")


# ============================================================
# 4. LOG-TRANSFORMED TARGET
# ============================================================

log_price = np.log1p(price)

print("\n" + "-" * 70)
print("LOG-TRANSFORMED PRICE")
print("-" * 70)

print(f"Original skewness: {price.skew():.4f}")
print(f"Log-transformed skewness: {log_price.skew():.4f}")

print(f"\nOriginal mean: ₹{price.mean():,.2f}")
print(f"Original median: ₹{price.median():,.2f}")

print(f"\nLog price mean: {log_price.mean():.4f}")
print(f"Log price median: {log_price.median():.4f}")


# ============================================================
# 5. PRICE RANGE COUNTS
# ============================================================

bins = [0, 200000, 500000, 1000000, 2000000, 5000000, np.inf]

labels = [
    "< ₹2L",
    "₹2L–₹5L",
    "₹5L–₹10L",
    "₹10L–₹20L",
    "₹20L–₹50L",
    "> ₹50L"
]

price_range = pd.cut(
    price,
    bins=bins,
    labels=labels,
    right=False
)

price_counts = price_range.value_counts().sort_index()

print("\n" + "-" * 70)
print("PRICE RANGE DISTRIBUTION")
print("-" * 70)

for price_range_name, count in price_counts.items():
    percentage = count / len(df) * 100
    print(
        f"{price_range_name:12} : "
        f"{count:4} cars ({percentage:6.2f}%)"
    )


# ============================================================
# 6. HISTOGRAM — ORIGINAL PRICE
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(price, bins=50)

plt.title("Distribution of Used Car Selling Prices")
plt.xlabel("Selling Price (₹)")
plt.ylabel("Number of Cars")

plt.tight_layout()

plt.savefig(
    "data/processed/selling_price_distribution.png",
    dpi=150
)

plt.show()


# ============================================================
# 7. HISTOGRAM — LOG PRICE
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(log_price, bins=50)

plt.title("Distribution of Log-Transformed Selling Prices")
plt.xlabel("log1p(Selling Price)")
plt.ylabel("Number of Cars")

plt.tight_layout()

plt.savefig(
    "data/processed/log_selling_price_distribution.png",
    dpi=150
)

plt.show()


# ============================================================
# 8. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION ANALYSIS COMPLETED")
print("=" * 70)