from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCALED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scaled"
)

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences_class"
)

SEQUENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_COUNTS = [10, 20, 30, 40]

SEQUENCE_LENGTH = 10


# ============================================================
# LABEL NAMES
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
# CREATE CLASS-PURE SEQUENCES
# ============================================================

def create_class_sequences(
    X,
    y,
    sequence_length=10
):

    sequences = []
    labels = []

    unique_classes = np.unique(y)

    for class_id in unique_classes:

        # ----------------------------------------------------
        # Get rows belonging to this class
        # ----------------------------------------------------

        class_indices = np.where(
            y == class_id
        )[0]

        class_X = X[class_indices]

        # ----------------------------------------------------
        # Number of complete sequences
        # ----------------------------------------------------

        n_sequences = (
            len(class_X)
            // sequence_length
        )

        usable_rows = (
            n_sequences
            * sequence_length
        )

        if n_sequences == 0:
            continue

        class_X = class_X[
            :usable_rows
        ]

        # ----------------------------------------------------
        # Reshape into sequences
        # ----------------------------------------------------

        class_sequences = class_X.reshape(
            n_sequences,
            sequence_length,
            X.shape[1]
        )

        class_labels = np.full(
            n_sequences,
            class_id,
            dtype=np.int64
        )

        sequences.append(
            class_sequences
        )

        labels.append(
            class_labels
        )

    # --------------------------------------------------------
    # Combine classes
    # --------------------------------------------------------

    X_sequences = np.concatenate(
        sequences,
        axis=0
    )

    y_sequences = np.concatenate(
        labels,
        axis=0
    )

    # --------------------------------------------------------
    # Shuffle sequences
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed=42
    )

    indices = rng.permutation(
        len(X_sequences)
    )

    X_sequences = X_sequences[
        indices
    ]

    y_sequences = y_sequences[
        indices
    ]

    return (
        X_sequences,
        y_sequences
    )


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(y):

    print(
        "\nSequence class distribution:"
    )

    counts = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    for class_id, count in counts.items():

        name = LABEL_NAMES.get(
            int(class_id),
            "UNKNOWN"
        )

        print(
            f"{int(class_id)} | "
            f"{name:20s} | "
            f"{count:,}"
        )


# ============================================================
# PROCESS ONE FEATURE VERSION
# ============================================================

def process_feature_version(
    feature_count
):

    print("\n")
    print("=" * 70)
    print(
        f"PROCESSING TOP {feature_count} FEATURES"
    )
    print("=" * 70)

    train_file = (
        SCALED_DIR
        / f"train_top_{feature_count}_scaled.csv"
    )

    test_file = (
        SCALED_DIR
        / f"test_top_{feature_count}_scaled.csv"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not train_file.exists():

        print(
            "\nWARNING: Training file not found:"
        )

        print(train_file)

        return

    if not test_file.exists():

        print(
            "\nWARNING: Testing file not found:"
        )

        print(test_file)

        return

    # ========================================================
    # TRAINING
    # ========================================================

    print("\nLoading training data:")
    print(train_file)

    train_df = pd.read_csv(
        train_file
    )

    print(
        f"Training shape: "
        f"{train_df.shape}"
    )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=["Label"]
    ).to_numpy(
        dtype=np.float32
    )

    y_train = train_df[
        "Label"
    ].to_numpy(
        dtype=np.int64
    )

    # ========================================================
    # CREATE TRAINING SEQUENCES
    # ========================================================

    print("\n" + "-" * 70)
    print("CREATING CLASS-PURE TRAINING SEQUENCES")
    print("-" * 70)

    X_train_seq, y_train_seq = (
        create_class_sequences(
            X_train,
            y_train,
            SEQUENCE_LENGTH
        )
    )

    print("\nSequence shape:")
    print(
        X_train_seq.shape
    )

    print("\nLabel shape:")
    print(
        y_train_seq.shape
    )

    print_distribution(
        y_train_seq
    )

    # --------------------------------------------------------
    # Save training
    # --------------------------------------------------------

    train_X_file = (
        SEQUENCE_DIR
        / f"train_top_{feature_count}_X.npy"
    )

    train_y_file = (
        SEQUENCE_DIR
        / f"train_top_{feature_count}_y.npy"
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
    # TESTING
    # ========================================================

    print("\nLoading testing data:")
    print(test_file)

    test_df = pd.read_csv(
        test_file
    )

    print(
        f"Testing shape: "
        f"{test_df.shape}"
    )

    X_test = test_df.drop(
        columns=["Label"]
    ).to_numpy(
        dtype=np.float32
    )

    y_test = test_df[
        "Label"
    ].to_numpy(
        dtype=np.int64
    )

    # ========================================================
    # CREATE TEST SEQUENCES
    # ========================================================

    print("\n" + "-" * 70)
    print("CREATING CLASS-PURE TESTING SEQUENCES")
    print("-" * 70)

    X_test_seq, y_test_seq = (
        create_class_sequences(
            X_test,
            y_test,
            SEQUENCE_LENGTH
        )
    )

    print("\nSequence shape:")
    print(
        X_test_seq.shape
    )

    print("\nLabel shape:")
    print(
        y_test_seq.shape
    )

    print_distribution(
        y_test_seq
    )

    # --------------------------------------------------------
    # Save testing
    # --------------------------------------------------------

    test_X_file = (
        SEQUENCE_DIR
        / f"test_top_{feature_count}_X.npy"
    )

    test_y_file = (
        SEQUENCE_DIR
        / f"test_top_{feature_count}_y.npy"
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
        f"TOP {feature_count} "
        "CLASS SEQUENCE CREATION COMPLETE"
    )
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CICIDS2017 CLASS-PURE "
        "CNN-GRU SEQUENCE CONSTRUCTION"
    )
    print("=" * 70)

    print(
        f"\nFeature configurations: "
        f"{FEATURE_COUNTS}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    for feature_count in FEATURE_COUNTS:

        process_feature_version(
            feature_count
        )

    print("\n")
    print("=" * 70)
    print(
        "CLASS-PURE SEQUENCE CREATION COMPLETE"
    )
    print("=" * 70)

    print("\nOutput directory:")
    print(SEQUENCE_DIR)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()