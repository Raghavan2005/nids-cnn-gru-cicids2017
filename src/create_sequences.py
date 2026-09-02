from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCALED_DIR = PROJECT_ROOT / "data" / "processed" / "scaled"
SEQUENCE_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"

SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_COUNTS = [10, 20, 30, 40, 61]

SEQUENCE_LENGTH = 10
STRIDE = 10


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_NAMES = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
    6: "SSH-Patator",
}


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(X, y, sequence_length=10, stride=10):

    sequences = []
    labels = []

    n_rows = len(X)

    for start in range(
        0,
        n_rows - sequence_length + 1,
        stride
    ):

        end = start + sequence_length

        sequence = X[start:end]

        # Majority label inside the sequence
        sequence_labels = y[start:end]

        values, counts = np.unique(
            sequence_labels,
            return_counts=True
        )

        majority_label = values[np.argmax(counts)]

        sequences.append(sequence)
        labels.append(majority_label)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int64)
    )


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(y):

    print("\nSequence class distribution:")

    counts = pd.Series(y).value_counts().sort_index()

    for class_id, count in counts.items():

        label_name = LABEL_NAMES.get(
            int(class_id),
            f"UNKNOWN_{class_id}"
        )

        print(
            f"{int(class_id)} | "
            f"{label_name:20s} | "
            f"{count:,}"
        )


# ============================================================
# PROCESS ONE FEATURE VERSION
# ============================================================

def process_feature_version(feature_count):

    print("\n")
    print("=" * 70)
    print(f"PROCESSING TOP {feature_count} FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------------

    train_file = (
        SCALED_DIR /
        f"train_top_{feature_count}_scaled.csv"
    )

    test_file = (
        SCALED_DIR /
        f"test_top_{feature_count}_scaled.csv"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not train_file.exists():

        print(
            f"\nWARNING: Training file not found:"
            f"\n{train_file}"
        )

        return

    if not test_file.exists():

        print(
            f"\nWARNING: Testing file not found:"
            f"\n{test_file}"
        )

        return

    # ========================================================
    # TRAINING DATA
    # ========================================================

    print("\nLoading training data:")
    print(train_file)

    train_df = pd.read_csv(train_file)

    print(
        f"Training shape: {train_df.shape}"
    )

    if "Label" not in train_df.columns:

        raise ValueError(
            "Label column not found in training data."
        )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=["Label"]
    ).to_numpy(
        dtype=np.float32
    )

    y_train = train_df["Label"].to_numpy(
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    unique_labels = set(
        np.unique(y_train)
    )

    unknown_labels = (
        unique_labels -
        set(LABEL_NAMES.keys())
    )

    if unknown_labels:

        raise ValueError(
            f"Unknown numeric labels found: "
            f"{unknown_labels}"
        )

    # ========================================================
    # CREATE TRAINING SEQUENCES
    # ========================================================

    print("\n" + "-" * 70)
    print("CREATING TRAINING SEQUENCES")
    print("-" * 70)

    print(
        f"Original rows : {len(X_train):,}"
    )

    print(
        f"Features      : {X_train.shape[1]}"
    )

    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )

    print(
        f"Stride        : {STRIDE}"
    )

    X_train_seq, y_train_seq = create_sequences(
        X_train,
        y_train,
        SEQUENCE_LENGTH,
        STRIDE
    )

    print("\nSequence shape:")
    print(X_train_seq.shape)

    print("\nLabel shape:")
    print(y_train_seq.shape)

    print_class_distribution(
        y_train_seq
    )

    # --------------------------------------------------------
    # Save training sequences
    # --------------------------------------------------------

    train_X_file = (
        SEQUENCE_DIR /
        f"train_top_{feature_count}_X.npy"
    )

    train_y_file = (
        SEQUENCE_DIR /
        f"train_top_{feature_count}_y.npy"
    )

    np.save(
        train_X_file,
        X_train_seq
    )

    np.save(
        train_y_file,
        y_train_seq
    )

    print("\nSaved:")
    print(train_X_file)
    print(train_y_file)

    # Free memory
    del train_df
    del X_train
    del y_train
    del X_train_seq
    del y_train_seq

    # ========================================================
    # TESTING DATA
    # ========================================================

    print("\n" + "-" * 70)
    print("LOADING TESTING DATA")
    print("-" * 70)

    print(test_file)

    test_df = pd.read_csv(test_file)

    print(
        f"Testing shape: {test_df.shape}"
    )

    if "Label" not in test_df.columns:

        raise ValueError(
            "Label column not found in testing data."
        )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X_test = test_df.drop(
        columns=["Label"]
    ).to_numpy(
        dtype=np.float32
    )

    y_test = test_df["Label"].to_numpy(
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    unique_labels = set(
        np.unique(y_test)
    )

    unknown_labels = (
        unique_labels -
        set(LABEL_NAMES.keys())
    )

    if unknown_labels:

        raise ValueError(
            f"Unknown numeric labels found: "
            f"{unknown_labels}"
        )

    # ========================================================
    # CREATE TESTING SEQUENCES
    # ========================================================

    print("\n" + "-" * 70)
    print("CREATING TESTING SEQUENCES")
    print("-" * 70)

    print(
        f"Original rows : {len(X_test):,}"
    )

    print(
        f"Features      : {X_test.shape[1]}"
    )

    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )

    print(
        f"Stride        : {STRIDE}"
    )

    X_test_seq, y_test_seq = create_sequences(
        X_test,
        y_test,
        SEQUENCE_LENGTH,
        STRIDE
    )

    print("\nSequence shape:")
    print(X_test_seq.shape)

    print("\nLabel shape:")
    print(y_test_seq.shape)

    print_class_distribution(
        y_test_seq
    )

    # --------------------------------------------------------
    # Save testing sequences
    # --------------------------------------------------------

    test_X_file = (
        SEQUENCE_DIR /
        f"test_top_{feature_count}_X.npy"
    )

    test_y_file = (
        SEQUENCE_DIR /
        f"test_top_{feature_count}_y.npy"
    )

    np.save(
        test_X_file,
        X_test_seq
    )

    np.save(
        test_y_file,
        y_test_seq
    )

    print("\nSaved:")
    print(test_X_file)
    print(test_y_file)

    print("\n" + "=" * 70)
    print(
        f"TOP {feature_count} SEQUENCE CREATION COMPLETE"
    )
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CICIDS2017 CNN-GRU SEQUENCE CONSTRUCTION")
    print("=" * 70)

    print(
        f"\nFeature configurations: "
        f"{FEATURE_COUNTS}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Stride: "
        f"{STRIDE}"
    )

    for feature_count in FEATURE_COUNTS:

        process_feature_version(
            feature_count
        )

    print("\n")
    print("=" * 70)
    print("ALL SEQUENCE CREATION COMPLETE")
    print("=" * 70)

    print("\nOutput directory:")
    print(SEQUENCE_DIR)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()