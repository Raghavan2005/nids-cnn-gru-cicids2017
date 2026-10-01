from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2017_features_cleaned.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CICIDS2017 TRAIN / TEST SPLIT")
print("=" * 70)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# SEPARATE FEATURES AND LABEL
# ============================================================

X = df.drop(columns=["Label"])
y = df["Label"]

print(f"\nFeatures : {X.shape[1]}")
print(f"Rows     : {X.shape[0]}")


# ============================================================
# STRATIFIED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# RECONSTRUCT DATAFRAMES
# ============================================================

train_df = X_train.copy()
train_df["Label"] = y_train.values

test_df = X_test.copy()
test_df["Label"] = y_test.values


# ============================================================
# RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print(f"\nTraining shape : {train_df.shape}")
print(f"Testing shape  : {test_df.shape}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("-" * 70)

print(train_df["Label"].value_counts())


print("\n" + "-" * 70)
print("TEST CLASS DISTRIBUTION")
print("-" * 70)

print(test_df["Label"].value_counts())


# ============================================================
# CLASS DISTRIBUTION (%)
# ============================================================

print("\n" + "-" * 70)
print("TRAINING CLASS DISTRIBUTION (%)")
print("-" * 70)

print(
    train_df["Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(4)
)


print("\n" + "-" * 70)
print("TEST CLASS DISTRIBUTION (%)")
print("-" * 70)

print(
    test_df["Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(4)
)


# ============================================================
# SAVE
# ============================================================

train_file = SPLIT_DIR / "train.csv"
test_file = SPLIT_DIR / "test.csv"

train_df.to_csv(
    train_file,
    index=False
)

test_df.to_csv(
    test_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print("\nTraining file:")
print(train_file)

print("\nTesting file:")
print(test_file)