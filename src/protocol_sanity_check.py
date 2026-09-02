from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(r"D:\PS26-PAPER1")

ANOVA_DIR = BASE_DIR / "data" / "processed" / "anova_datasets"
SCALED_DIR = BASE_DIR / "data" / "processed" / "scaled"
SEQ_DIR = BASE_DIR / "data" / "processed" / "sequences_class"
RESULT_DIR = BASE_DIR / "results" / "protocol_check"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COUNTS = [10, 20, 30, 40]

LABEL_COLUMN = "Label"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PAPER 1 PROTOCOL / LEAKAGE SANITY CHECK")
print("=" * 70)


# ============================================================
# 1. CHECK ANOVA DATASETS
# ============================================================

print("\n" + "=" * 70)
print("1. ANOVA FEATURE DATASETS")
print("=" * 70)

anova_ok = True

for n in FEATURE_COUNTS:

    train_file = ANOVA_DIR / f"train_top_{n}.csv"
    test_file = ANOVA_DIR / f"test_top_{n}.csv"

    print(f"\nTop {n} features")

    if not train_file.exists():
        print("  [FAIL] Training file missing:")
        print(f"         {train_file}")
        anova_ok = False
        continue

    if not test_file.exists():
        print("  [FAIL] Testing file missing:")
        print(f"         {test_file}")
        anova_ok = False
        continue

    train = pd.read_csv(train_file, nrows=5)
    test = pd.read_csv(test_file, nrows=5)

    train_features = [c for c in train.columns if c != LABEL_COLUMN]
    test_features = [c for c in test.columns if c != LABEL_COLUMN]

    print(f"  Training columns : {len(train_features)}")
    print(f"  Testing columns  : {len(test_features)}")

    if train_features == test_features:
        print("  [PASS] Same feature columns")
    else:
        print("  [FAIL] Training/testing feature mismatch")
        anova_ok = False


# ============================================================
# 2. CHECK SCALING FILES
# ============================================================

print("\n" + "=" * 70)
print("2. SCALING FILES")
print("=" * 70)

scaling_ok = True

for n in FEATURE_COUNTS:

    train_file = SCALED_DIR / f"train_top_{n}_scaled.csv"
    test_file = SCALED_DIR / f"test_top_{n}_scaled.csv"

    print(f"\nTop {n} features")

    if not train_file.exists():
        print("  [FAIL] Missing scaled training file")
        scaling_ok = False
        continue

    if not test_file.exists():
        print("  [FAIL] Missing scaled testing file")
        scaling_ok = False
        continue

    train = pd.read_csv(train_file, nrows=5)
    test = pd.read_csv(test_file, nrows=5)

    train_features = [c for c in train.columns if c != LABEL_COLUMN]
    test_features = [c for c in test.columns if c != LABEL_COLUMN]

    if train_features == test_features:
        print("  [PASS] Feature columns match")
    else:
        print("  [FAIL] Feature columns do not match")
        scaling_ok = False

    if len(train_features) == n and len(test_features) == n:
        print(f"  [PASS] Correctly contains {n} features")
    else:
        print("  [FAIL] Incorrect feature count")
        scaling_ok = False


# ============================================================
# 3. CHECK SEQUENCE DATA
# ============================================================

print("\n" + "=" * 70)
print("3. CLASS-PURE SEQUENCE DATA")
print("=" * 70)

sequence_ok = True

for n in FEATURE_COUNTS:

    train_x_file = SEQ_DIR / f"train_top_{n}_X.npy"
    train_y_file = SEQ_DIR / f"train_top_{n}_y.npy"

    test_x_file = SEQ_DIR / f"test_top_{n}_X.npy"
    test_y_file = SEQ_DIR / f"test_top_{n}_y.npy"

    print(f"\nTop {n} features")

    files_exist = all([
        train_x_file.exists(),
        train_y_file.exists(),
        test_x_file.exists(),
        test_y_file.exists()
    ])

    if not files_exist:
        print("  [FAIL] One or more sequence files missing")
        sequence_ok = False
        continue

    X_train = np.load(train_x_file, mmap_mode="r")
    y_train = np.load(train_y_file, mmap_mode="r")

    X_test = np.load(test_x_file, mmap_mode="r")
    y_test = np.load(test_y_file, mmap_mode="r")

    print(f"  X_train shape : {X_train.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  X_test shape  : {X_test.shape}")
    print(f"  y_test shape  : {y_test.shape}")

    # Check feature dimension
    if X_train.shape[2] == n:
        print(f"  [PASS] Training has {n} features")
    else:
        print("  [FAIL] Training feature dimension incorrect")
        sequence_ok = False

    if X_test.shape[2] == n:
        print(f"  [PASS] Testing has {n} features")
    else:
        print("  [FAIL] Testing feature dimension incorrect")
        sequence_ok = False

    # Check sequence length
    if X_train.shape[1] == 10 and X_test.shape[1] == 10:
        print("  [PASS] Sequence length = 10")
    else:
        print("  [FAIL] Sequence length mismatch")
        sequence_ok = False

    # Check labels
    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)

    print(f"  Train classes : {train_classes}")
    print(f"  Test classes  : {test_classes}")

    if set(train_classes) == set(range(7)):
        print("  [PASS] Training contains all 7 classes")
    else:
        print("  [WARNING] Training does not contain all 7 classes")

    if set(test_classes) == set(range(7)):
        print("  [PASS] Testing contains all 7 classes")
    else:
        print("  [WARNING] Testing does not contain all 7 classes")


# ============================================================
# 4. CHECK TRAIN/TEST ROW OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("4. TRAIN / TEST OVERLAP CHECK")
print("=" * 70)

print("""
The original datasets are large, so this check uses a reproducible
sample of feature rows.

If identical rows are found between train and test, investigate them.
Some duplicate network-flow records can naturally occur in CICIDS2017,
so an overlap is a warning rather than automatic proof of leakage.
""")

overlap_ok = True

for n in FEATURE_COUNTS:

    train_file = SCALED_DIR / f"train_top_{n}_scaled.csv"
    test_file = SCALED_DIR / f"test_top_{n}_scaled.csv"

    if not train_file.exists() or not test_file.exists():
        continue

    print(f"\nTop {n} features")

    train_sample = pd.read_csv(
        train_file,
        nrows=10000
    )

    test_sample = pd.read_csv(
        test_file,
        nrows=10000
    )

    feature_cols = [
        c for c in train_sample.columns
        if c != LABEL_COLUMN
    ]

    train_hash = pd.util.hash_pandas_object(
        train_sample[feature_cols],
        index=False
    )

    test_hash = pd.util.hash_pandas_object(
        test_sample[feature_cols],
        index=False
    )

    overlap = len(
        set(train_hash).intersection(set(test_hash))
    )

    print(f"  Sampled training rows : {len(train_sample)}")
    print(f"  Sampled testing rows  : {len(test_sample)}")
    print(f"  Identical feature rows: {overlap}")

    if overlap == 0:
        print("  [PASS] No sampled identical feature rows")
    else:
        print("  [WARNING] Identical sampled rows detected")
        overlap_ok = False


# ============================================================
# 5. FINAL PROTOCOL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SANITY-CHECK SUMMARY")
print("=" * 70)

print(f"\nANOVA dataset structure : {'PASS' if anova_ok else 'CHECK'}")
print(f"Scaling structure       : {'PASS' if scaling_ok else 'CHECK'}")
print(f"Sequence structure      : {'PASS' if sequence_ok else 'CHECK'}")
print(f"Sample overlap check    : {'PASS' if overlap_ok else 'CHECK'}")

print("""
IMPORTANT:

This script verifies the DATA PIPELINE structure.

It does NOT prove that the model is free from every possible form
of information leakage.

For Paper 1, the most important issue is the zero-day/LOAO protocol:
the unseen attack class must not be available during training for
that particular LOAO experiment.
""")

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = RESULT_DIR / "protocol_sanity_summary.txt"

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("PAPER 1 PROTOCOL / LEAKAGE SANITY CHECK\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"ANOVA dataset structure : "
        f"{'PASS' if anova_ok else 'CHECK'}\n"
    )

    f.write(
        f"Scaling structure       : "
        f"{'PASS' if scaling_ok else 'CHECK'}\n"
    )

    f.write(
        f"Sequence structure      : "
        f"{'PASS' if sequence_ok else 'CHECK'}\n"
    )

    f.write(
        f"Sample overlap check    : "
        f"{'PASS' if overlap_ok else 'CHECK'}\n"
    )

print("\nSaved:")
print(summary_file)

print("\n" + "=" * 70)
print("PROTOCOL SANITY CHECK COMPLETE")
print("=" * 70)