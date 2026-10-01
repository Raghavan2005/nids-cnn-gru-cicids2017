# ============================================================
# CICIDS2017 CNN-GRU INTERACTIVE PRESENTATION DEMO
# ============================================================
#
# Classes:
#   0 - BENIGN
#   1 - DoS Hulk
#   2 - DDoS
#   3 - PortScan
#   4 - DoS GoldenEye
#   5 - FTP-Patator
#   6 - SSH-Patator
#
# Model:
#   CNN-GRU
#   Top-40 ANOVA selected features
#   Sequence length = 10
#
# ============================================================

import os
import numpy as np
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cnn_gru_top40_final.keras"
)

X_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sequences_class",
    "test_top_40_X.npy"
)

Y_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sequences_class",
    "test_top_40_y.npy"
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
    6: "SSH-Patator"
}


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 75)
print("          CICIDS2017 CNN-GRU INTRUSION DETECTION")
print("                    PRESENTATION DEMO")
print("=" * 75)

print("\nChecking project files...")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model not found:")
    print(MODEL_PATH)
    raise SystemExit

if not os.path.exists(X_PATH):
    print("\nERROR: Test sequence file not found:")
    print(X_PATH)
    raise SystemExit

if not os.path.exists(Y_PATH):
    print("\nERROR: Test label file not found:")
    print(Y_PATH)
    raise SystemExit

print("All required files found.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading CNN-GRU model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print(
    "Model input shape :",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading Top-40 test sequences...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("              AVAILABLE SAMPLE NUMBERS")
print("=" * 75)

for class_id in range(7):

    indices = np.where(y == class_id)[0]

    print(
        f"\nClass {class_id}: {CLASS_NAMES[class_id]}"
    )

    print(
        f"Total samples: {len(indices)}"
    )

    if len(indices) > 0:

        # Show first 20 sample numbers
        display_indices = indices[:20]

        print(
            "Sample numbers:",
            display_indices.tolist()
        )

    else:
        print(
            "NO SAMPLES AVAILABLE"
        )


print("\n" + "=" * 75)


# ============================================================
# HELPER FUNCTION
# ============================================================

def predict_sample(sample_number):

    # --------------------------------------------------------
    # Check sample number
    # --------------------------------------------------------

    if sample_number < 0 or sample_number >= len(X):

        print("\nERROR: Invalid sample number.")

        print(
            f"Valid range: 0 - {len(X) - 1}"
        )

        return


    # --------------------------------------------------------
    # Extract sample
    # --------------------------------------------------------

    sample = X[
        sample_number:sample_number + 1
    ]


    # --------------------------------------------------------
    # Actual class
    # --------------------------------------------------------

    actual_id = int(
        y[sample_number]
    )

    actual_class = CLASS_NAMES.get(
        actual_id,
        f"Unknown ({actual_id})"
    )


    # --------------------------------------------------------
    # CNN-GRU prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        sample,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_id = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES.get(
        predicted_id,
        f"Unknown ({predicted_id})"
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = (
        float(probabilities[predicted_id])
        * 100.0
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n")
    print("=" * 75)
    print("                    PREDICTION RESULT")
    print("=" * 75)

    print(
        f"Sample number : {sample_number}"
    )

    print(
        f"Actual class  : {actual_class}"
    )

    print(
        f"Predicted     : {predicted_class}"
    )

    print(
        f"Confidence    : {confidence:.2f}%"
    )

    print("-" * 75)


    # --------------------------------------------------------
    # Correct / incorrect
    # --------------------------------------------------------

    if predicted_id == actual_id:

        print(
            "RESULT        : CORRECT"
        )

    else:

        print(
            "RESULT        : INCORRECT"
        )


    # ========================================================
    # ALL CLASS PROBABILITIES
    # ========================================================

    print("\nClass probabilities:")

    ranked_classes = np.argsort(
        probabilities
    )[::-1]

    for class_id in ranked_classes:

        class_name = CLASS_NAMES.get(
            int(class_id),
            f"Unknown ({class_id})"
        )

        probability = (
            float(probabilities[class_id])
            * 100.0
        )

        print(
            f"  {class_name:<20} "
            f"{probability:8.4f}%"
        )


    print("=" * 75)


# ============================================================
# INTERACTIVE MENU
# ============================================================

while True:

    print("\n")
    print("=" * 75)
    print("                    DEMO MENU")
    print("=" * 75)

    print("1. Enter sample number")
    print("2. Show sample numbers for all classes")
    print("3. Quit")

    print("=" * 75)

    choice = input(
        "Select option: "
    ).strip()


    # ========================================================
    # OPTION 1
    # ========================================================

    if choice == "1":

        value = input(
            "\nEnter sample number: "
        ).strip()

        try:

            sample_number = int(value)

        except ValueError:

            print(
                "\nPlease enter a valid integer."
            )

            continue


        predict_sample(
            sample_number
        )


    # ========================================================
    # OPTION 2
    # ========================================================

    elif choice == "2":

        print("\n")
        print("=" * 75)
        print("             SAMPLE NUMBERS BY CLASS")
        print("=" * 75)

        for class_id in range(7):

            indices = np.where(
                y == class_id
            )[0]

            print(
                f"\n{class_id} - "
                f"{CLASS_NAMES[class_id]}"
            )

            print(
                f"Total: {len(indices)}"
            )

            if len(indices) > 0:

                print(
                    "First 30:",
                    indices[:30].tolist()
                )

            else:

                print(
                    "No samples found."
                )

        print("=" * 75)


    # ========================================================
    # OPTION 3
    # ========================================================

    elif choice == "3":

        print(
            "\nPresentation demo finished."
        )

        break


    # ========================================================
    # INVALID OPTION
    # ========================================================

    else:

        print(
            "\nInvalid option. "
            "Please select 1, 2, or 3."
        )