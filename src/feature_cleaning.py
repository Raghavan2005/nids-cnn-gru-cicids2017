from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2017_final.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2017_features_cleaned.csv"
)


# ============================================================
# FEATURES ALREADY IDENTIFIED AS CONSTANT
# ============================================================

CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]


# ============================================================
# EXACT DUPLICATE FEATURES
# Keep one feature from each duplicate group.
# ============================================================

DUPLICATE_FEATURES_TO_REMOVE = [
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Fwd Header Length.1",
    "CWE Flag Count",
    "SYN Flag Count",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("FEATURE CLEANING")
    print("=" * 70)

    print("\nLoading:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nOriginal shape: {df.shape}")

    return df


# ============================================================
# REMOVE ANOMALOUS NEGATIVE ROWS
# ============================================================

def remove_negative_anomalies(df):
    """
    Remove the confirmed 35 anomalous rows where:
        min_seg_size_forward < 0
        OR
        Fwd Header Length < 0

    These rows were identified during the previous
    anomaly analysis and all were labelled BENIGN.
    """

    anomaly_mask = (
        (df["min_seg_size_forward"] < 0)
        |
        (df["Fwd Header Length"] < 0)
    )

    rows_removed = anomaly_mask.sum()

    df = df.loc[~anomaly_mask].copy()

    print("\n" + "-" * 70)
    print("ANOMALOUS ROW REMOVAL")
    print("-" * 70)

    print(f"Rows removed: {rows_removed:,}")

    return df, rows_removed


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

def remove_constant_features(df):

    existing = [
        col
        for col in CONSTANT_FEATURES
        if col in df.columns
    ]

    df = df.drop(columns=existing)

    print("\n" + "-" * 70)
    print("CONSTANT FEATURE REMOVAL")
    print("-" * 70)

    print(f"Features removed: {len(existing)}")

    for feature in existing:
        print(f"  - {feature}")

    return df


# ============================================================
# REMOVE EXACT DUPLICATE FEATURES
# ============================================================

def remove_duplicate_features(df):

    existing = [
        col
        for col in DUPLICATE_FEATURES_TO_REMOVE
        if col in df.columns
    ]

    df = df.drop(columns=existing)

    print("\n" + "-" * 70)
    print("EXACT DUPLICATE FEATURE REMOVAL")
    print("-" * 70)

    print(f"Features removed: {len(existing)}")

    for feature in existing:
        print(f"  - {feature}")

    return df


# ============================================================
# VERIFY DATASET
# ============================================================

def verify_dataset(df):

    print("\n" + "=" * 70)
    print("FINAL FEATURE CLEANING VERIFICATION")
    print("=" * 70)

    print(f"\nShape:")
    print(df.shape)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = df.isna().sum().sum()

    print(f"\nMissing values:")
    print(missing_values)

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    infinite_values = np.isinf(
        df[numeric_cols].to_numpy()
    ).sum()

    print(f"\nInfinite values:")
    print(infinite_values)

    # --------------------------------------------------------
    # Constant features
    # --------------------------------------------------------

    feature_cols = [
        col
        for col in df.columns
        if col != "Label"
    ]

    constant_features = [
        col
        for col in feature_cols
        if df[col].nunique(dropna=False) <= 1
    ]

    print("\nRemaining constant features:")
    print(constant_features)

    # --------------------------------------------------------
    # Duplicate feature pairs
    # --------------------------------------------------------

    duplicate_pairs = []

    features = df[feature_cols]

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):

            col1 = feature_cols[i]
            col2 = feature_cols[j]

            if features[col1].equals(features[col2]):
                duplicate_pairs.append(
                    (col1, col2)
                )

    print("\nRemaining exact duplicate feature pairs:")
    print(len(duplicate_pairs))

    for pair in duplicate_pairs:
        print(f"  {pair[0]} == {pair[1]}")

    # --------------------------------------------------------
    # Negative anomaly verification
    # --------------------------------------------------------

    negative_min_seg = (
        df["min_seg_size_forward"] < 0
    ).sum()

    negative_header = (
        df["Fwd Header Length"] < 0
    ).sum()

    print("\nNegative min_seg_size_forward:")
    print(negative_min_seg)

    print("\nNegative Fwd Header Length:")
    print(negative_header)

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    print("\nClass distribution:")
    print(df["Label"].value_counts())

    # --------------------------------------------------------
    # Final feature count
    # --------------------------------------------------------

    print(
        f"\nNumber of features: "
        f"{len(df.columns) - 1}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    original_rows = len(df)
    original_columns = len(df.columns)

    # --------------------------------------------------------
    # STEP 1
    # Remove confirmed anomalous rows
    # --------------------------------------------------------

    df, anomaly_rows = remove_negative_anomalies(df)

    # --------------------------------------------------------
    # STEP 2
    # Remove constant features
    # --------------------------------------------------------

    df = remove_constant_features(df)

    # --------------------------------------------------------
    # STEP 3
    # Remove exact duplicate features
    # --------------------------------------------------------

    df = remove_duplicate_features(df)

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    verify_dataset(df)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FEATURE CLEANING COMPLETE")
    print("=" * 70)

    print(
        f"\nOriginal rows    : {original_rows:,}"
    )

    print(
        f"Final rows       : {len(df):,}"
    )

    print(
        f"Rows removed     : "
        f"{original_rows - len(df):,}"
    )

    print(
        f"\nOriginal columns : {original_columns}"
    )

    print(
        f"Final columns    : {len(df.columns)}"
    )

    print(
        f"Features         : "
        f"{len(df.columns) - 1}"
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()