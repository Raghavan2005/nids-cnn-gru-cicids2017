from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# DEEP TRAIN / TEST LEAKAGE CHECK
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SCALED_DIR = BASE_DIR / "data" / "processed" / "scaled"
RESULT_DIR = BASE_DIR / "results" / "protocol_check"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SCALED_DIR / "train_top_30_scaled.csv"
TEST_FILE = SCALED_DIR / "test_top_30_scaled.csv"

LABEL_COLUMN = "Label"

print("=" * 70)
print("DEEP TRAIN / TEST LEAKAGE CHECK")
print("TOP-30 FEATURES")
print("=" * 70)

print("\nLoading training data:")
print(TRAIN_FILE)

train = pd.read_csv(TRAIN_FILE)

print("Training shape:", train.shape)

print("\nLoading testing data:")
print(TEST_FILE)

test = pd.read_csv(TEST_FILE)

print("Testing shape:", test.shape)


# ============================================================
# CHECK COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("1. COLUMN CHECK")
print("=" * 70)

train_features = [
    c for c in train.columns
    if c != LABEL_COLUMN
]

test_features = [
    c for c in test.columns
    if c != LABEL_COLUMN
]

print("Training features:", len(train_features))
print("Testing features :", len(test_features))

if train_features == test_features:
    print("[PASS] Train/test feature columns match")
else:
    print("[FAIL] Train/test feature columns do NOT match")
    raise SystemExit


# ============================================================
# CHECK LABELS
# ============================================================

print("\n" + "=" * 70)
print("2. LABEL CHECK")
print("=" * 70)

print("\nTraining labels:")
print(train[LABEL_COLUMN].value_counts().sort_index())

print("\nTesting labels:")
print(test[LABEL_COLUMN].value_counts().sort_index())


# ============================================================
# CREATE HASH FOR EVERY FEATURE ROW
# ============================================================

print("\n" + "=" * 70)
print("3. CREATING ROW SIGNATURES")
print("=" * 70)

print("\nThis may take some time for the full dataset...")

train_hash = pd.util.hash_pandas_object(
    train[train_features],
    index=False
)

test_hash = pd.util.hash_pandas_object(
    test[test_features],
    index=False
)

print("[DONE] Row signatures created")


# ============================================================
# FIND EXACT FEATURE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("4. EXACT FEATURE ROW OVERLAP")
print("=" * 70)

train_hash_set = set(train_hash)
test_hash_set = set(test_hash)

common_hashes = train_hash_set.intersection(test_hash_set)

print("Unique train feature rows:", len(train_hash_set))
print("Unique test feature rows :", len(test_hash_set))
print("Common feature rows      :", len(common_hashes))

if len(common_hashes) == 0:
    print("\n[PASS] No exact feature-row overlap found.")
else:
    print("\n[WARNING] Exact feature-row overlap detected.")


# ============================================================
# ANALYZE LABEL CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("5. LABEL CONSISTENCY OF DUPLICATES")
print("=" * 70)

# Keep only rows whose feature hash occurs in both datasets
train_overlap = train[
    train_hash.isin(common_hashes)
].copy()

test_overlap = test[
    test_hash.isin(common_hashes)
].copy()

train_overlap["_row_hash"] = train_hash[
    train_hash.isin(common_hashes)
].values

test_overlap["_row_hash"] = test_hash[
    test_hash.isin(common_hashes)
].values


# Map each feature signature to labels appearing in training/testing
train_labels_by_hash = (
    train_overlap
    .groupby("_row_hash")[LABEL_COLUMN]
    .apply(lambda x: set(x))
)

test_labels_by_hash = (
    test_overlap
    .groupby("_row_hash")[LABEL_COLUMN]
    .apply(lambda x: set(x))
)


same_label = 0
different_label = 0

for h in common_hashes:

    train_labels = train_labels_by_hash.get(h, set())
    test_labels = test_labels_by_hash.get(h, set())

    if train_labels.intersection(test_labels):
        same_label += 1

    if train_labels != test_labels:
        different_label += 1


print("Duplicate signatures with shared label :", same_label)
print("Duplicate signatures with label mismatch:", different_label)


# ============================================================
# COUNT OVERLAPPING ROWS
# ============================================================

print("\n" + "=" * 70)
print("6. OVERLAPPING ROW COUNTS")
print("=" * 70)

train_overlap_count = len(train_overlap)
test_overlap_count = len(test_overlap)

print("Training rows involved in overlap:", train_overlap_count)
print("Testing rows involved in overlap :", test_overlap_count)


train_overlap_percentage = (
    train_overlap_count / len(train) * 100
)

test_overlap_percentage = (
    test_overlap_count / len(test) * 100
)

print(
    f"Training overlap percentage: "
    f"{train_overlap_percentage:.6f}%"
)

print(
    f"Testing overlap percentage : "
    f"{test_overlap_percentage:.6f}%"
)


# ============================================================
# LABEL CROSS-TABULATION
# ============================================================

print("\n" + "=" * 70)
print("7. LABEL DISTRIBUTION OF OVERLAPPING ROWS")
print("=" * 70)

print("\nTraining labels among overlapping rows:")
print(
    train_overlap[LABEL_COLUMN]
    .value_counts()
    .sort_index()
)

print("\nTesting labels among overlapping rows:")
print(
    test_overlap[LABEL_COLUMN]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAVE DUPLICATE EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("8. SAVING DUPLICATE EXAMPLES")
print("=" * 70)

example_file = RESULT_DIR / "top30_overlap_examples.csv"

if len(common_hashes) > 0:

    # Take a manageable sample
    example_hashes = list(common_hashes)[:100]

    train_examples = train_overlap[
        train_overlap["_row_hash"].isin(example_hashes)
    ].copy()

    test_examples = test_overlap[
        test_overlap["_row_hash"].isin(example_hashes)
    ].copy()

    train_examples["_dataset"] = "TRAIN"
    test_examples["_dataset"] = "TEST"

    examples = pd.concat(
        [train_examples, test_examples],
        ignore_index=True
    )

    examples.to_csv(
        example_file,
        index=False
    )

    print("Saved:")
    print(example_file)

else:

    print("No duplicate examples to save.")


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

if len(common_hashes) == 0:

    conclusion = (
        "No exact train/test feature-row overlap was found. "
        "The duplicate-row check does not indicate direct "
        "train/test duplication."
    )

elif different_label == 0:

    conclusion = (
        "Exact feature-row overlap exists, but the overlapping "
        "feature signatures do not show label disagreement. "
        "This is a WARNING that repeated flows/features exist, "
        "but it is not by itself proof of leakage."
    )

else:

    conclusion = (
        "Exact feature-row overlap exists and some overlapping "
        "feature signatures have different labels. "
        "This requires investigation because identical feature "
        "vectors receiving different labels can affect evaluation."
    )

print("\n" + conclusion)


# ============================================================
# SAVE REPORT
# ============================================================

report_file = RESULT_DIR / "deep_leakage_check_top30.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("DEEP TRAIN / TEST LEAKAGE CHECK - TOP 30\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Training shape: {train.shape}\n")
    f.write(f"Testing shape : {test.shape}\n\n")

    f.write(
        f"Unique train feature rows: "
        f"{len(train_hash_set)}\n"
    )

    f.write(
        f"Unique test feature rows: "
        f"{len(test_hash_set)}\n"
    )

    f.write(
        f"Common feature rows: "
        f"{len(common_hashes)}\n\n"
    )

    f.write(
        f"Training rows involved in overlap: "
        f"{train_overlap_count}\n"
    )

    f.write(
        f"Testing rows involved in overlap: "
        f"{test_overlap_count}\n"
    )

    f.write(
        f"Training overlap percentage: "
        f"{train_overlap_percentage:.6f}%\n"
    )

    f.write(
        f"Testing overlap percentage: "
        f"{test_overlap_percentage:.6f}%\n\n"
    )

    f.write(
        f"Duplicate signatures with shared label: "
        f"{same_label}\n"
    )

    f.write(
        f"Duplicate signatures with label mismatch: "
        f"{different_label}\n\n"
    )

    f.write("CONCLUSION\n")
    f.write("-" * 60 + "\n")
    f.write(conclusion + "\n")


print("\nSaved:")
print(report_file)

print("\n" + "=" * 70)
print("DEEP LEAKAGE CHECK COMPLETE")
print("=" * 70)