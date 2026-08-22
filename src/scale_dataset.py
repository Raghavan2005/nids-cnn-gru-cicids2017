from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_selection"
    / "anova_datasets"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scaled"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASETS TO SCALE
# ============================================================

FEATURE_SIZES = [10, 20, 30, 40, 61]


# ============================================================
# LABEL ENCODING
# ============================================================

LABEL_MAPPING = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "DDoS": 2,
    "PortScan": 3,
    "DoS GoldenEye": 4,
    "FTP-Patator": 5,
    "SSH-Patator": 6,
}


# ============================================================
# SCALE ONE DATASET
# ============================================================

def scale_dataset(feature_size):

    print("\n" + "=" * 70)
    print(f"SCALING TOP {feature_size} FEATURES")
    print("=" * 70)

    train_file = INPUT_DIR / f"train_top_{feature_size}.csv"
    test_file = INPUT_DIR / f"test_top_{feature_size}.csv"

    print(f"\nLoading training data:")
    print(train_file)

    print("\nLoading testing data:")
    print(test_file)

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print(f"\nTraining shape: {train_df.shape}")
    print(f"Testing shape : {test_df.shape}")

    # --------------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------------

    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    # --------------------------------------------------------
    # Check feature consistency
    # --------------------------------------------------------

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    # --------------------------------------------------------
    # Check for invalid values
    # --------------------------------------------------------

    if X_train.isna().sum().sum() != 0:
        raise ValueError("NaN values found in training data.")

    if X_test.isna().sum().sum() != 0:
        raise ValueError("NaN values found in testing data.")

    if np.isinf(X_train.to_numpy()).any():
        raise ValueError("Infinite values found in training data.")

    if np.isinf(X_test.to_numpy()).any():
        raise ValueError("Infinite values found in testing data.")

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    print("\nFitting StandardScaler on TRAINING data only...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    print("Transforming testing data...")

    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # Convert back to DataFrame
    # --------------------------------------------------------

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    y_train_encoded = y_train.map(LABEL_MAPPING)
    y_test_encoded = y_test.map(LABEL_MAPPING)

    if y_train_encoded.isna().any():
        raise ValueError(
            "Unknown label found in training data."
        )

    if y_test_encoded.isna().any():
        raise ValueError(
            "Unknown label found in testing data."
        )

    # --------------------------------------------------------
    # Add encoded labels
    # --------------------------------------------------------

    X_train_scaled["Label"] = y_train_encoded.astype(int)
    X_test_scaled["Label"] = y_test_encoded.astype(int)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_output = (
        OUTPUT_DIR
        / f"train_top_{feature_size}_scaled.csv"
    )

    test_output = (
        OUTPUT_DIR
        / f"test_top_{feature_size}_scaled.csv"
    )

    X_train_scaled.to_csv(
        train_output,
        index=False
    )

    X_test_scaled.to_csv(
        test_output,
        index=False
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("SCALING VERIFICATION")
    print("-" * 70)

    print(
        "\nTraining feature mean "
        "(approximately 0):"
    )

    print(
        X_train_scaled.drop(columns=["Label"])
        .mean()
        .abs()
        .max()
    )

    print(
        "\nTraining feature standard deviation "
        "(approximately 1):"
    )

    print(
        X_train_scaled.drop(columns=["Label"])
        .std()
        .sub(1)
        .abs()
        .max()
    )

    print("\nTraining labels:")
    print(
        X_train_scaled["Label"]
        .value_counts()
        .sort_index()
    )

    print("\nTesting labels:")
    print(
        X_test_scaled["Label"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved:")
    print(train_output)
    print(test_output)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CICIDS2017 FEATURE SCALING")
    print("=" * 70)

    for feature_size in FEATURE_SIZES:
        scale_dataset(feature_size)

    print("\n" + "=" * 70)
    print("SCALING COMPLETE")
    print("=" * 70)

    print("\nOutput directory:")
    print(OUTPUT_DIR)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()