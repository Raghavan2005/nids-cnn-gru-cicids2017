from pathlib import Path
import numpy as np
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "cnn_gru_top30_final.keras"
)

X_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
    / "test_top_30_X.npy"
)

Y_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
    / "test_top_30_y.npy"
)


# ============================================================
# CLASS LABELS
# ============================================================

CLASS_NAMES = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
    6: "SSH-Patator",
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("CICIDS2017 CNN-GRU INTRUSION DETECTION DEMO")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("LOADING CNN-GRU MODEL")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_FILE)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"\nModel file not found:\n{MODEL_FILE}"
        )

    model = tf.keras.models.load_model(MODEL_FILE)

    print("\nModel loaded successfully.")
    print("\nModel input shape:")
    print(model.input_shape)

    return model


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print("\n" + "=" * 70)
    print("LOADING TEST SEQUENCES")
    print("=" * 70)

    print("\nX:")
    print(X_TEST_FILE)

    print("\nY:")
    print(Y_TEST_FILE)

    if not X_TEST_FILE.exists():
        raise FileNotFoundError(
            f"\nTest X file not found:\n{X_TEST_FILE}"
        )

    if not Y_TEST_FILE.exists():
        raise FileNotFoundError(
            f"\nTest y file not found:\n{Y_TEST_FILE}"
        )

    X_test = np.load(X_TEST_FILE)
    y_test = np.load(Y_TEST_FILE)

    print("\nX_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_test, y_test


# ============================================================
# VERIFY PIPELINE
# ============================================================

def verify_pipeline(model, X_test, y_test):

    print("\n" + "=" * 70)
    print("PIPELINE VERIFICATION")
    print("=" * 70)

    if len(X_test.shape) != 3:
        raise ValueError(
            f"Expected 3D input, received {X_test.shape}"
        )

    sequence_length = X_test.shape[1]
    feature_count = X_test.shape[2]

    print(
        f"\nEach input = {sequence_length} flows × "
        f"{feature_count} selected features"
    )

    print("\nClasses:")

    for number, name in CLASS_NAMES.items():
        print(f"{number} -> {name}")

    model_features = model.input_shape[-1]

    if model_features != feature_count:
        raise ValueError(
            "\nMODEL/DATA FEATURE MISMATCH\n"
            f"Model expects : {model_features}\n"
            f"Data contains : {feature_count}"
        )

    if model.input_shape[1] != sequence_length:
        raise ValueError(
            "\nMODEL/DATA SEQUENCE LENGTH MISMATCH\n"
            f"Model expects : {model.input_shape[1]}\n"
            f"Data contains : {sequence_length}"
        )

    print("\n[PASS] Model and test sequence dimensions match.")


# ============================================================
# PREDICT ONE SAMPLE
# ============================================================

def predict_sample(model, X_test, y_test, sample_number):

    if sample_number < 0 or sample_number >= len(X_test):

        print(
            f"\nInvalid sample number: {sample_number}"
        )

        print(
            f"Valid range: 0 - {len(X_test) - 1}"
        )

        return

    X_sample = X_test[
        sample_number:sample_number + 1
    ]

    actual_label = int(
        y_test[sample_number]
    )

    probabilities = model.predict(
        X_sample,
        verbose=0
    )[0]

    predicted_label = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_label]
    )

    actual_name = CLASS_NAMES.get(
        actual_label,
        f"Unknown ({actual_label})"
    )

    predicted_name = CLASS_NAMES.get(
        predicted_label,
        f"Unknown ({predicted_label})"
    )

    print("\n")
    print("=" * 70)
    print("INTRUSION DETECTION RESULT")
    print("=" * 70)

    print(f"\nTest sample    : {sample_number}")
    print(f"Flow sequence  : {X_test.shape[1]} flows")
    print(f"Features       : {X_test.shape[2]}")

    print("\nPrediction probabilities:")
    print("-" * 45)

    for class_id in range(len(probabilities)):

        class_name = CLASS_NAMES.get(
            class_id,
            f"Class {class_id}"
        )

        print(
            f"{class_name:<20} "
            f"{probabilities[class_id]:.4f}"
        )

    print("\n" + "-" * 70)

    print(
        f"Actual class    : {actual_name}"
    )

    print(
        f"Predicted class : {predicted_name}"
    )

    print(
        f"Confidence      : {confidence * 100:.2f}%"
    )

    print("-" * 70)

    # --------------------------------------------------------
    # DETECTION STATUS
    # --------------------------------------------------------

    if predicted_label == 0:

        print("STATUS: NORMAL TRAFFIC")
        print("No intrusion detected.")

    else:

        print("STATUS: INTRUSION DETECTED")
        print(
            f"Attack type: {predicted_name}"
        )

    # --------------------------------------------------------
    # GROUND-TRUTH VERIFICATION
    # --------------------------------------------------------

    if predicted_label == actual_label:

        print("Prediction: CORRECT")

    else:

        print("Prediction: INCORRECT")

        print(
            f"Expected: {actual_name}"
        )

    print("=" * 70)


# ============================================================
# FIND REPRESENTATIVE SAMPLES
# ============================================================

def find_samples(y_test):

    samples = {}

    for class_id in CLASS_NAMES:

        indices = np.where(
            y_test == class_id
        )[0]

        if len(indices) > 0:

            samples[class_id] = int(
                indices[0]
            )

    return samples


# ============================================================
# SHOW RECOMMENDED DEMO SAMPLES
# ============================================================

def show_recommended_samples(y_test):

    print("\n" + "=" * 70)
    print("RECOMMENDED DEMO SAMPLES")
    print("=" * 70)

    samples = find_samples(y_test)

    for class_id, class_name in CLASS_NAMES.items():

        if class_id in samples:

            print(
                f"{class_name:<20} "
                f"sample {samples[class_id]}"
            )

        else:

            print(
                f"{class_name:<20} "
                "NOT AVAILABLE"
            )

    print("\nUse these sample numbers for the presentation.")


# ============================================================
# AUTOMATIC DEMO
# ============================================================

def automatic_demo(model, X_test, y_test):

    print("\n" + "=" * 70)
    print("AUTOMATIC DEMONSTRATION")
    print("=" * 70)

    samples = find_samples(y_test)

    # --------------------------------------------------------
    # Normal traffic
    # --------------------------------------------------------

    if 0 in samples:

        print("\n[1] BENIGN / NORMAL TRAFFIC")

        predict_sample(
            model,
            X_test,
            y_test,
            samples[0]
        )

    # --------------------------------------------------------
    # Attack classes available in test set
    # --------------------------------------------------------

    for class_id in [1, 2, 3, 4, 5, 6]:

        if class_id not in samples:
            continue

        print(
            f"\n[{class_id + 1}] "
            f"{CLASS_NAMES[class_id]} ATTACK"
        )

        predict_sample(
            model,
            X_test,
            y_test,
            samples[class_id]
        )


# ============================================================
# INTERACTIVE DEMO
# ============================================================

def interactive_demo(model, X_test, y_test):

    print("\n" + "=" * 70)
    print("INTERACTIVE DEMONSTRATION")
    print("=" * 70)

    print(
        f"\nAvailable test samples: {len(X_test)}"
    )

    print(
        "\nEnter a sample number."
        "\nPress ENTER to show recommended samples."
        "\nType A for automatic demonstration."
        "\nType Q to quit."
    )

    while True:

        try:

            choice = input(
                "\nSample number: "
            ).strip()

        except KeyboardInterrupt:

            print("\n\nDemo stopped.")

            break

        except EOFError:

            print("\n\nDemo stopped.")

            break

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if choice.upper() == "Q":

            print("\nDemo finished.")

            break

        # ----------------------------------------------------
        # AUTOMATIC DEMO
        # ----------------------------------------------------

        if choice.upper() == "A":

            automatic_demo(
                model,
                X_test,
                y_test
            )

            continue

        # ----------------------------------------------------
        # SHOW RECOMMENDED
        # ----------------------------------------------------

        if choice == "":

            show_recommended_samples(
                y_test
            )

            continue

        # ----------------------------------------------------
        # SAMPLE NUMBER
        # ----------------------------------------------------

        try:

            sample_number = int(choice)

        except ValueError:

            print(
                "\nPlease enter a valid sample number, "
                "A, or Q."
            )

            continue

        predict_sample(
            model,
            X_test,
            y_test,
            sample_number
        )


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    X_test, y_test = load_test_data()

    verify_pipeline(
        model,
        X_test,
        y_test
    )

    show_recommended_samples(
        y_test
    )

    interactive_demo(
        model,
        X_test,
        y_test
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()