from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SELECTED CLASSES
# ============================================================

SELECTED_LABELS = [
    "BENIGN",
    "DoS Hulk",
    "DDoS",
    "PortScan",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
]


# ============================================================
# REMOVE CONFLICTING FEATURE GROUPS
# ============================================================

def remove_conflicting_groups(df):
    """
    Removes all rows belonging to feature vectors
    that occur with more than one label.

    Consistent duplicates are retained.
    """

    feature_cols = [
        col for col in df.columns
        if col != "Label"
    ]

    # Number of unique labels for each feature vector
    label_counts = (
        df.groupby(
            feature_cols,
            dropna=False
        )["Label"]
        .transform("nunique")
    )

    # Conflicting feature vectors have > 1 unique label
    conflict_mask = label_counts > 1

    conflicting_rows = conflict_mask.sum()

    df_clean = df.loc[
        ~conflict_mask
    ].copy()

    return df_clean, conflicting_rows


# ============================================================
# REMOVE NaN AND INFINITY
# ============================================================

def remove_invalid_rows(df):

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    invalid_mask = (
        df[numeric_cols]
        .isna()
        .any(axis=1)
        |
        np.isinf(
            df[numeric_cols].to_numpy()
        ).any(axis=1)
    )

    invalid_rows = invalid_mask.sum()

    df_clean = df.loc[
        ~invalid_mask
    ].copy()

    return df_clean, invalid_rows


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path):

    print("\n" + "=" * 70)
    print(f"Processing: {file_path.name}")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(file_path)

    original_rows = len(df)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Check label column
    if "Label" not in df.columns:
        raise ValueError(
            f"Label column not found in {file_path.name}"
        )

    # --------------------------------------------------------
    # STEP 1: Remove conflicting feature vectors
    # --------------------------------------------------------

    df, conflicting_rows = remove_conflicting_groups(df)

    # --------------------------------------------------------
    # STEP 2: Remove NaN and Infinity
    # --------------------------------------------------------

    df, invalid_rows = remove_invalid_rows(df)

    # --------------------------------------------------------
    # STEP 3: Keep selected labels
    # --------------------------------------------------------

    before_filter = len(df)

    df = df[
        df["Label"].isin(
            SELECTED_LABELS
        )
    ].copy()

    removed_by_label_filter = (
        before_filter - len(df)
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print(f"Original rows                 : {original_rows:,}")
    print(f"Conflicting rows removed      : {conflicting_rows:,}")
    print(f"NaN/Inf rows removed          : {invalid_rows:,}")
    print(f"Other labels removed          : {removed_by_label_filter:,}")
    print(f"Final rows from this file     : {len(df):,}")

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("CICIDS2017 PREPROCESSING PIPELINE")
    print("=" * 70)

    # Find CSV files
    csv_files = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    print(f"\nFiles found: {len(csv_files)}")

    all_data = []

    # Process every file
    for file_path in csv_files:

        cleaned_df = process_file(
            file_path
        )

        all_data.append(
            cleaned_df
        )

    # ========================================================
    # COMBINE ALL FILES
    # ========================================================

    print("\n" + "=" * 70)
    print("COMBINING FILES")
    print("=" * 70)

    combined_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # ========================================================
    # FINAL DATASET SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal rows    : {len(combined_df):,}"
    )

    print(
        f"Total columns : {len(combined_df.columns)}"
    )

    print("\nClass distribution:\n")

    print(
        combined_df["Label"]
        .value_counts()
    )

    # ========================================================
    # SAVE DATASET
    # ========================================================

    output_file = (
        PROCESSED_DIR /
        "cicids2017_final.csv"
    )

    combined_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print("\nSaved processed dataset to:")

    print(output_file)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()