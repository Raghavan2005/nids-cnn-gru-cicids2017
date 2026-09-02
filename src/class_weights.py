from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scaled"
    / "train_top_30_scaled.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "class_weights"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CLASS WEIGHT CALCULATION")
print("=" * 70)

print("\nLoading:")
print(TRAIN_FILE)

df = pd.read_csv(TRAIN_FILE)

print("\nDataset shape:", df.shape)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

y = df["Label"]

print("\n" + "-" * 70)
print("CLASS DISTRIBUTION")
print("-" * 70)

print(y.value_counts())


# ============================================================
# ENCODE LABELS
# ============================================================

labels = sorted(y.unique())

label_to_id = {
    label: idx
    for idx, label in enumerate(labels)
}

id_to_label = {
    idx: label
    for label, idx in label_to_id.items()
}

y_encoded = y.map(label_to_id).values


# ============================================================
# COMPUTE BALANCED CLASS WEIGHTS
# ============================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(labels)),
    y=y_encoded
)

class_weight_dict = {
    int(class_id): float(weight)
    for class_id, weight in enumerate(class_weights)
}


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

for class_id, weight in class_weight_dict.items():

    print(
    f"{class_id:2d} | "
    f"{str(id_to_label[class_id]):20s} | "
    f"{weight:.6f}"
)


# ============================================================
# SAVE LABEL MAPPING
# ============================================================

mapping_df = pd.DataFrame({
    "Label_ID": list(id_to_label.keys()),
    "Label": list(id_to_label.values()),
    "Class_Weight": [
        class_weight_dict[i]
        for i in id_to_label.keys()
    ]
})

output_file = OUTPUT_DIR / "class_weights.csv"

mapping_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CLASS WEIGHT CALCULATION COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(output_file)