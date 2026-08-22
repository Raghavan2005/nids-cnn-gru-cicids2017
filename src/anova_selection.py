from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.feature_selection import f_classif


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("ANOVA FEATURE SELECTION")
print("=" * 70)

print("\nLoading training data:")
print(TRAIN_FILE)

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

train_df = pd.read_csv(TRAIN_FILE)

print(f"\nTraining shape: {train_df.shape}")


# ============================================================
# SEPARATE FEATURES AND LABEL
# ============================================================

X = train_df.drop(columns=["Label"])
y = train_df["Label"]

print(f"Number of features: {X.shape[1]}")
print(f"Number of samples : {X.shape[0]}")


# ============================================================
# ENSURE NUMERIC FEATURES
# ============================================================

non_numeric = X.select_dtypes(
    exclude=np.number
).columns.tolist()

if non_numeric:
    raise TypeError(
        "Non-numeric features found:\n"
        + "\n".join(non_numeric)
    )


# ============================================================
# ANOVA F-TEST
# ============================================================

print("\nCalculating ANOVA F-scores...")

f_scores, p_values = f_classif(
    X,
    y
)


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

anova_results = pd.DataFrame({
    "Feature": X.columns,
    "F_Score": f_scores,
    "P_Value": p_values
})


# ============================================================
# HANDLE INVALID SCORES
# ============================================================

anova_results["F_Score"] = (
    anova_results["F_Score"]
    .replace([np.inf, -np.inf], np.nan)
)

anova_results["P_Value"] = (
    anova_results["P_Value"]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# SORT BY F-SCORE
# ============================================================

anova_results = anova_results.sort_values(
    by="F_Score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# ADD RANK
# ============================================================

anova_results.insert(
    0,
    "Rank",
    range(1, len(anova_results) + 1)
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 FEATURES BY ANOVA F-SCORE")
print("=" * 70)

print(
    anova_results.head(30).to_string(
        index=False
    )
)


# ============================================================
# SAVE FULL ANOVA RESULTS
# ============================================================

anova_file = (
    OUTPUT_DIR
    / "anova_feature_ranking.csv"
)

anova_results.to_csv(
    anova_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANOVA ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nFeatures analyzed: "
    f"{len(anova_results)}"
)

print("\nSaved ranking to:")
print(anova_file)