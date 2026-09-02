from pathlib import Path
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = PROJECT_ROOT / "data" / "processed" / "sequences_class"
MODEL_DIR = PROJECT_ROOT / "models" / "loao"
RESULT_DIR = PROJECT_ROOT / "results" / "loao"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_COUNT = 40
SEQUENCE_LENGTH = 10

# Attack classes
ATTACK_CLASSES = {
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
    6: "SSH-Patator",
}

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
# RANDOM SEED
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD SEQUENCES
# ============================================================

def load_sequences():

    print("=" * 70)
    print("LOADING CLASS-PURE SEQUENCES")
    print("=" * 70)

    train_x_path = (
        SEQUENCE_DIR / f"train_top_{FEATURE_COUNT}_X.npy"
    )

    train_y_path = (
        SEQUENCE_DIR / f"train_top_{FEATURE_COUNT}_y.npy"
    )

    test_x_path = (
        SEQUENCE_DIR / f"test_top_{FEATURE_COUNT}_X.npy"
    )

    test_y_path = (
        SEQUENCE_DIR / f"test_top_{FEATURE_COUNT}_y.npy"
    )

    print("\nTraining X:")
    print(train_x_path)

    X_train = np.load(train_x_path)

    print("\nTraining y:")
    print(train_y_path)

    y_train = np.load(train_y_path)

    print("\nTesting X:")
    print(test_x_path)

    X_test = np.load(test_x_path)

    print("\nTesting y:")
    print(test_y_path)

    y_test = np.load(test_y_path)

    print("\nShapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    return X_train, y_train, X_test, y_test


# ============================================================
# BUILD CNN-GRU
# ============================================================

def build_model(input_shape, num_classes):

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling1D(
            pool_size=2
        ),

        layers.Dropout(0.2),

        layers.GRU(
            64,
            return_sequences=False
        ),

        layers.Dropout(0.2),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dropout(0.2),

        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# CREATE LABEL MAPPING
# ============================================================

def create_label_mapping(unseen_class):

    """
    Training classes exclude the unseen attack.

    Example:
    unseen_class = 6

    Original:
    0 = BENIGN
    1 = DoS Hulk
    ...
    6 = SSH-Patator

    Training classes:
    0,1,2,3,4,5

    These are remapped to:
    0,1,2,3,4,5

    The unseen attack remains completely outside
    the training label space.
    """

    train_classes = [
        c for c in LABEL_NAMES.keys()
        if c != unseen_class
    ]

    label_to_new = {
        original: new
        for new, original in enumerate(train_classes)
    }

    new_to_original = {
        new: original
        for original, new in label_to_new.items()
    }

    return label_to_new, new_to_original


# ============================================================
# PREPARE LOAO DATA
# ============================================================

def prepare_loao_data(
    X_train,
    y_train,
    X_test,
    y_test,
    unseen_class
):

    print("\n" + "-" * 70)
    print(
        f"UNSEEN ATTACK: "
        f"{LABEL_NAMES[unseen_class]}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # TRAINING DATA
    # Remove unseen attack completely
    # --------------------------------------------------------

    train_mask = y_train != unseen_class

    X_train_loao = X_train[train_mask]
    y_train_loao_original = y_train[train_mask]

    # --------------------------------------------------------
    # TESTING DATA
    #
    # We evaluate the model specifically on:
    # 1. BENIGN
    # 2. unseen attack
    #
    # This measures whether the model can distinguish
    # an unseen attack from normal traffic.
    # --------------------------------------------------------

    test_mask = np.isin(
        y_test,
        [0, unseen_class]
    )

    X_test_loao = X_test[test_mask]
    y_test_original = y_test[test_mask]

    # --------------------------------------------------------
    # CREATE TRAINING LABEL MAPPING
    # --------------------------------------------------------

    label_to_new, new_to_original = create_label_mapping(
        unseen_class
    )

    y_train_loao = np.array([
        label_to_new[int(label)]
        for label in y_train_loao_original
    ])

    # --------------------------------------------------------
    # Binary evaluation labels
    #
    # BENIGN = 0
    # UNSEEN ATTACK = 1
    # --------------------------------------------------------

    y_test_binary = np.where(
        y_test_original == unseen_class,
        1,
        0
    )

    print("\nTraining classes:")

    for original_class in sorted(label_to_new.keys()):
        print(
            f"{original_class} | "
            f"{LABEL_NAMES[original_class]:20s} | "
            f"{np.sum(y_train_loao_original == original_class):,}"
        )

    print("\nTraining samples:", len(y_train_loao))

    print("\nZero-day test set:")

    print(
        f"BENIGN               : "
        f"{np.sum(y_test_binary == 0):,}"
    )

    print(
        f"{LABEL_NAMES[unseen_class]:20s}: "
        f"{np.sum(y_test_binary == 1):,}"
    )

    return (
        X_train_loao,
        y_train_loao,
        X_test_loao,
        y_test_binary,
        label_to_new,
        new_to_original
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    unseen_class
):

    num_classes = len(
        np.unique(y_train)
    )

    print("\n" + "=" * 70)
    print("TRAINING CNN-GRU")
    print("=" * 70)

    print(
        "Unseen attack:",
        LABEL_NAMES[unseen_class]
    )

    print(
        "Number of training classes:",
        num_classes
    )

    print(
        "Training shape:",
        X_train.shape
    )

    model = build_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        ),
        num_classes=num_classes
    )

    model.summary()

    model_path = (
        MODEL_DIR /
        f"cnn_gru_loao_unseen_{unseen_class}.keras"
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=256,
        callbacks=[
            early_stop,
            checkpoint
        ],
        verbose=1
    )

    print("\nBest model saved:")
    print(model_path)

    return model


# ============================================================
# EVALUATE ZERO-DAY
# ============================================================

def evaluate_zero_day(
    model,
    X_test,
    y_test_binary,
    unseen_class
):

    print("\n" + "=" * 70)
    print("ZERO-DAY EVALUATION")
    print("=" * 70)

    print(
        "Unseen attack:",
        LABEL_NAMES[unseen_class]
    )

    probabilities = model.predict(
        X_test,
        batch_size=256,
        verbose=1
    )

    predicted_classes = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The trained model does not have an unseen-attack output.
    #
    # Therefore we use confidence-based rejection.
    #
    # If the model's maximum probability is low,
    # classify the sample as UNKNOWN / ZERO-DAY.
    # --------------------------------------------------------

    max_probability = np.max(
        probabilities,
        axis=1
    )

    threshold = 0.70

    predicted_binary = np.where(
        max_probability < threshold,
        1,
        0
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test_binary,
        predicted_binary
    )

    precision = precision_score(
        y_test_binary,
        predicted_binary,
        zero_division=0
    )

    recall = recall_score(
        y_test_binary,
        predicted_binary,
        zero_division=0
    )

    f1 = f1_score(
        y_test_binary,
        predicted_binary,
        zero_division=0
    )

    print("\n" + "-" * 70)
    print("ZERO-DAY METRICS")
    print("-" * 70)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Precision         : {precision:.4f}"
    )

    print(
        f"Recall            : {recall:.4f}"
    )

    print(
        f"F1 Score          : {f1:.4f}"
    )

    print(
        f"Detection Rate    : {recall:.4f}"
    )

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_test_binary,
            predicted_binary,
            target_names=[
                "BENIGN",
                "UNSEEN ATTACK"
            ],
            digits=4,
            zero_division=0
        )
    )

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    cm = confusion_matrix(
        y_test_binary,
        predicted_binary
    )

    print(cm)

    return {
        "unseen_class": unseen_class,
        "unseen_attack": LABEL_NAMES[unseen_class],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold
    }


# ============================================================
# MAIN LOAO EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("CICIDS2017 LOAO ZERO-DAY EXPERIMENT")
    print("=" * 70)

    print(
        f"\nFeature count: {FEATURE_COUNT}"
    )

    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_sequences()

    results = []

    # --------------------------------------------------------
    # RUN EACH UNSEEN ATTACK
    # --------------------------------------------------------

    for unseen_class in ATTACK_CLASSES.keys():

        print("\n\n")
        print("#" * 70)
        print(
            f"LOAO EXPERIMENT "
            f"{unseen_class}/6"
        )
        print(
            f"Unseen attack: "
            f"{LABEL_NAMES[unseen_class]}"
        )
        print("#" * 70)

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        (
            X_train_loao,
            y_train_loao,
            X_test_loao,
            y_test_binary,
            label_to_new,
            new_to_original
        ) = prepare_loao_data(
            X_train,
            y_train,
            X_test,
            y_test,
            unseen_class
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = train_model(
            X_train_loao,
            y_train_loao,
            unseen_class
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        result = evaluate_zero_day(
            model,
            X_test_loao,
            y_test_binary,
            unseen_class
        )

        results.append(result)

        # ----------------------------------------------------
        # Save individual report
        # ----------------------------------------------------

        report_file = (
            RESULT_DIR /
            f"loao_unseen_{unseen_class}.txt"
        )

        with open(
            report_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "CICIDS2017 LOAO ZERO-DAY "
                "EVALUATION\n"
            )

            f.write("=" * 70 + "\n\n")

            f.write(
                f"Feature count: "
                f"{FEATURE_COUNT}\n"
            )

            f.write(
                f"Sequence length: "
                f"{SEQUENCE_LENGTH}\n"
            )

            f.write(
                f"Unseen attack: "
                f"{LABEL_NAMES[unseen_class]}\n\n"
            )

            f.write(
                "METRICS\n"
            )

            f.write("-" * 70 + "\n")

            f.write(
                f"Accuracy          : "
                f"{result['accuracy']:.4f}\n"
            )

            f.write(
                f"Precision         : "
                f"{result['precision']:.4f}\n"
            )

            f.write(
                f"Recall            : "
                f"{result['recall']:.4f}\n"
            )

            f.write(
                f"F1 Score          : "
                f"{result['f1']:.4f}\n"
            )

            f.write(
                f"Threshold         : "
                f"{result['threshold']:.2f}\n"
            )

        print(
            "\nReport saved:"
        )

        print(report_file)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    summary_file = (
        RESULT_DIR /
        "loao_summary.csv"
    )

    results_df.to_csv(
        summary_file,
        index=False
    )

    print("\n\n")
    print("=" * 70)
    print("LOAO EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nSUMMARY")

    print(
        results_df[
            [
                "unseen_attack",
                "accuracy",
                "precision",
                "recall",
                "f1"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(summary_file)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()