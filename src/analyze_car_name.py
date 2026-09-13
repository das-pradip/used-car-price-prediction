import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/car_data_feature_engineered.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# 2. BASIC NAME ANALYSIS
# ============================================================

print("=" * 70)
print("CAR NAME ANALYSIS")
print("=" * 70)

print(f"Number of rows: {len(df)}")
print(f"Unique car names: {df['name'].nunique()}")
print(f"Missing car names: {df['name'].isna().sum()}")


# ============================================================
# 3. EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE CAR NAMES")
print("=" * 70)

for name in df["name"].drop_duplicates().head(30):
    print(name)


# ============================================================
# 4. NAME LENGTH
# ============================================================

df["name_word_count"] = (
    df["name"]
    .astype(str)
    .str.split()
    .str.len()
)

print("\n" + "=" * 70)
print("NAME WORD COUNT")
print("=" * 70)

print(df["name_word_count"].describe())


# ============================================================
# 5. WORD COUNT DISTRIBUTION
# ============================================================

print("\nWord count distribution:")

word_counts = df["name_word_count"].value_counts().sort_index()

for count, frequency in word_counts.items():
    percentage = frequency / len(df) * 100

    print(
        f"{int(count)} words: "
        f"{frequency} cars "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 6. SECOND WORD ANALYSIS
# ============================================================

df["name_second_word"] = (
    df["name"]
    .astype(str)
    .str.split()
    .str[1]
)

print("\n" + "=" * 70)
print("SECOND WORD ANALYSIS")
print("=" * 70)

print(
    f"Unique second words: "
    f"{df['name_second_word'].nunique()}"
)

print("\nMost common second words:")

second_word_counts = (
    df["name_second_word"]
    .value_counts()
    .head(30)
)

print(second_word_counts)


# ============================================================
# 7. FIRST TWO WORDS
# ============================================================

df["name_first_two_words"] = (
    df["name"]
    .astype(str)
    .str.split()
    .str[:2]
    .str.join(" ")
)

print("\n" + "=" * 70)
print("FIRST TWO WORDS")
print("=" * 70)

print(
    f"Unique first-two-word combinations: "
    f"{df['name_first_two_words'].nunique()}"
)

print("\nMost common combinations:")

print(
    df["name_first_two_words"]
    .value_counts()
    .head(30)
)


# ============================================================
# 8. DUPLICATE NAME FREQUENCY
# ============================================================

name_frequency = df["name"].value_counts()

print("\n" + "=" * 70)
print("NAME FREQUENCY")
print("=" * 70)

print(
    f"Names appearing once: "
    f"{(name_frequency == 1).sum()}"
)

print(
    f"Names appearing more than once: "
    f"{(name_frequency > 1).sum()}"
)

print(
    f"Most frequent name: "
    f"{name_frequency.index[0]}"
)

print(
    f"Most frequent name count: "
    f"{name_frequency.iloc[0]}"
)


# ============================================================
# 9. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("CAR NAME ANALYSIS COMPLETED")
print("=" * 70)