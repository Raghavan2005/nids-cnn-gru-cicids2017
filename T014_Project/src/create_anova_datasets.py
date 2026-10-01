from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)

ANOVA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_selection"
    / "anova_feature_ranking.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_selection"
    / "anova_datasets"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

FEATURE_COUNTS = [10, 20, 30, 40, 61]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ANOVA FEATURE SUBSET CREATION")
print("=" * 70)

print("\nLoading training data:")
print(TRAIN_FILE)

train_df = pd.read_csv(TRAIN_FILE)

print("\nLoading testing data:")
print(TEST_FILE)

test_df = pd.read_csv(TEST_FILE)

print("\nLoading ANOVA ranking:")
print(ANOVA_FILE)

anova_df = pd.read_csv(ANOVA_FILE)


# ============================================================
# IDENTIFY FEATURES
# ============================================================

if "Feature" not in anova_df.columns:
    raise ValueError(
        "ANOVA ranking file must contain a 'Feature' column."
    )

anova_features = anova_df["Feature"].tolist()

print("\nTotal ANOVA-ranked features:", len(anova_features))


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_train = [
    feature
    for feature in anova_features
    if feature not in train_df.columns
]

missing_test = [
    feature
    for feature in anova_features
    if feature not in test_df.columns
]

if missing_train:
    raise ValueError(
        f"Features missing from training data:\n{missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Features missing from testing data:\n{missing_test}"
    )


# ============================================================
# CREATE SUBSETS
# ============================================================

for n_features in FEATURE_COUNTS:

    print("\n" + "-" * 70)
    print(f"Creating Top {n_features} ANOVA feature dataset")
    print("-" * 70)

    selected_features = anova_features[:n_features]

    print("\nSelected features:")

    for rank, feature in enumerate(
        selected_features,
        start=1
    ):
        print(f"{rank:2d}. {feature}")

    # --------------------------------------------------------
    # Keep Label
    # --------------------------------------------------------

    columns = selected_features + ["Label"]

    train_subset = train_df[columns].copy()
    test_subset = test_df[columns].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_output = (
        OUTPUT_DIR
        / f"train_top_{n_features}.csv"
    )

    test_output = (
        OUTPUT_DIR
        / f"test_top_{n_features}.csv"
    )

    train_subset.to_csv(
        train_output,
        index=False
    )

    test_subset.to_csv(
        test_output,
        index=False
    )

    print("\nTraining shape:", train_subset.shape)
    print("Testing shape :", test_subset.shape)

    print("\nSaved:")
    print(train_output)
    print(test_output)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ANOVA DATASET CREATION COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)