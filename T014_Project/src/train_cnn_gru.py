from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    GRU,
    Dropout,
    Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences_class"
)

WEIGHT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "class_weights"
    / "class_weights.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_COUNT = 40
SEQUENCE_LENGTH = 10
NUM_CLASSES = 7

BATCH_SIZE = 256
EPOCHS = 30
RANDOM_SEED = 42

tf.keras.utils.set_random_seed(RANDOM_SEED)


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
    6: "SSH-Patator"
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("CNN-GRU TRAINING")
    print("=" * 70)

    train_x_path = (
        SEQUENCE_DIR
        / f"train_top_{FEATURE_COUNT}_X.npy"
    )

    train_y_path = (
        SEQUENCE_DIR
        / f"train_top_{FEATURE_COUNT}_y.npy"
    )

    test_x_path = (
        SEQUENCE_DIR
        / f"test_top_{FEATURE_COUNT}_X.npy"
    )

    test_y_path = (
        SEQUENCE_DIR
        / f"test_top_{FEATURE_COUNT}_y.npy"
    )

    print("\nLoading training data:")

    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nLoading testing data:")

    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)

    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    return X_train, y_train, X_test, y_test


# ============================================================
# LOAD CLASS WEIGHTS
# ============================================================

def load_class_weights():

    print("\n" + "=" * 70)
    print("LOADING CLASS WEIGHTS")
    print("=" * 70)

    weights_df = pd.read_csv(WEIGHT_FILE)

    print("\nCSV columns:")
    print(list(weights_df.columns))

    print("\nClass weights:")

    class_weights = {}

    # --------------------------------------------------------
    # Detect columns automatically
    # --------------------------------------------------------

    columns_lower = {
        str(col).strip().lower(): col
        for col in weights_df.columns
    }

    # Find class column
    class_column = None

    for possible in [
        "class_id",
        "class",
        "label",
        "classid"
    ]:
        if possible in columns_lower:
            class_column = columns_lower[possible]
            break

    # Find weight column
    weight_column = None

    for possible in [
        "weight",
        "class_weight",
        "weights"
    ]:
        if possible in columns_lower:
            weight_column = columns_lower[possible]
            break

    if class_column is None:
        raise ValueError(
            "Could not find class ID column in class_weights.csv.\n"
            f"Available columns: {list(weights_df.columns)}"
        )

    if weight_column is None:
        raise ValueError(
            "Could not find weight column in class_weights.csv.\n"
            f"Available columns: {list(weights_df.columns)}"
        )

    # --------------------------------------------------------
    # Read weights
    # --------------------------------------------------------

    for _, row in weights_df.iterrows():

        class_id = int(row[class_column])
        weight = float(row[weight_column])

        class_weights[class_id] = weight

        print(
            f"{class_id} | "
            f"{LABEL_NAMES[class_id]:20s} | "
            f"{weight:.6f}"
        )

    # --------------------------------------------------------
    # Verify all 7 classes exist
    # --------------------------------------------------------

    expected_classes = set(range(NUM_CLASSES))
    actual_classes = set(class_weights.keys())

    missing_classes = expected_classes - actual_classes

    if missing_classes:
        raise ValueError(
            f"Missing class weights for classes: "
            f"{sorted(missing_classes)}"
        )

    return class_weights


# ============================================================
# BUILD CNN-GRU MODEL
# ============================================================

def build_model():

    print("\n" + "=" * 70)
    print("BUILDING CNN-GRU MODEL")
    print("=" * 70)

    model = Sequential([

        Input(
            shape=(
                SEQUENCE_LENGTH,
                FEATURE_COUNT
            )
        ),

        # ----------------------------------------------------
        # CNN FEATURE EXTRACTION
        # ----------------------------------------------------

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        # ----------------------------------------------------
        # SECOND CNN LAYER
        # ----------------------------------------------------

        Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        # ----------------------------------------------------
        # GRU SEQUENCE LEARNING
        # ----------------------------------------------------

        GRU(
            64,
            return_sequences=False
        ),

        Dropout(0.3),

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            NUM_CLASSES,
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

    model.summary()

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    class_weights
):

    print("\n" + "=" * 70)
    print("TRAINING CNN-GRU")
    print("=" * 70)

    best_model_path = (
        MODEL_DIR
        / f"cnn_gru_top{FEATURE_COUNT}_best.keras"
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(

        X_train,
        y_train,

        validation_split=0.15,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1,

        shuffle=True
    )

    return history, best_model_path


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print("\n" + "-" * 70)
    print("OVERALL METRICS")
    print("-" * 70)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1          : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1       : {weighted_f1:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=list(LABEL_NAMES.keys()),
        target_names=list(LABEL_NAMES.values()),
        digits=4,
        zero_division=0
    )

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(LABEL_NAMES.keys())
    )

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(cm)

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report_file = (
        RESULT_DIR
        / f"cnn_gru_top{FEATURE_COUNT}_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CNN-GRU EVALUATION REPORT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Feature count: {FEATURE_COUNT}\n"
        )

        f.write(
            f"Sequence length: {SEQUENCE_LENGTH}\n\n"
        )

        f.write(
            "OVERALL METRICS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Accuracy          : {accuracy:.4f}\n"
        )

        f.write(
            f"Balanced Accuracy : {balanced_accuracy:.4f}\n"
        )

        f.write(
            f"Macro F1          : {macro_f1:.4f}\n"
        )

        f.write(
            f"Weighted F1       : {weighted_f1:.4f}\n\n"
        )

        f.write(
            "CLASSIFICATION REPORT\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(report)

        f.write(
            "\n\nCONFUSION MATRIX\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            np.array2string(cm)
        )

    print(
        f"\nReport saved to:\n{report_file}"
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "y_pred": y_pred,
        "confusion_matrix": cm
    }


# ============================================================
# MAIN
# ============================================================

def main():

    X_train, y_train, X_test, y_test = load_data()

    # --------------------------------------------------------
    # Verify shapes
    # --------------------------------------------------------

    expected_train_shape = (
        None,
        SEQUENCE_LENGTH,
        FEATURE_COUNT
    )

    expected_test_shape = (
        None,
        SEQUENCE_LENGTH,
        FEATURE_COUNT
    )

    if X_train.shape[1:] != expected_train_shape[1:]:

        raise ValueError(
            f"Unexpected training shape: "
            f"{X_train.shape}"
        )

    if X_test.shape[1:] != expected_test_shape[1:]:

        raise ValueError(
            f"Unexpected testing shape: "
            f"{X_test.shape}"
        )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = load_class_weights()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history, best_model_path = train_model(
        model,
        X_train,
        y_train,
        class_weights
    )

    # --------------------------------------------------------
    # Evaluate best model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    best_model = tf.keras.models.load_model(
        best_model_path
    )

    results = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        MODEL_DIR
        / f"cnn_gru_top{FEATURE_COUNT}_final.keras"
    )

    model.save(final_model_path)

    print(
        f"\nFinal model saved to:\n"
        f"{final_model_path}"
    )

    print("\n" + "=" * 70)
    print("CNN-GRU TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()