import pandas as pd

data_path = "data/raw/Car details v3.csv"

df = pd.read_csv(data_path)

print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nMissing value percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\nNumber of missing values per row:")
print(df.isnull().sum(axis=1).value_counts().sort_index())

print("\nRows containing missing values:")
print(df[df.isnull().any(axis=1)].head(10))

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nMost repeated records:")

duplicate_counts = (
    df.value_counts()
      .head(10)
)

print(duplicate_counts)