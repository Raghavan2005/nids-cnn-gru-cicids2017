"""
Synthetic stand-in for the real CICIDS2017 raw dataset.

NOT part of the original research pipeline described in
PROJECT_CONTEXT.md. The real, multi-GB CICIDS2017 CSVs are not
available in this environment, so this script generates small CSV
files that match the real dataset's column schema (78 feature
columns + Label) so the existing preprocessing/training/evaluation
scripts in src/ can be exercised end-to-end.

Metrics produced downstream from this data verify pipeline
MECHANICS only -- they do not reflect the paper's actual findings,
which depended on ~2.8M rows of real network traffic.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
ROWS_PER_CLASS = 1000

# ============================================================
# REAL CICIDS2017 COLUMN SCHEMA (from PROJECT_CONTEXT.md / the
# tracked ANOVA ranking file)
# ============================================================

TARGET_LABELS = [
    "BENIGN",
    "DoS Hulk",
    "DDoS",
    "PortScan",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
]

# The 61 "usable" features, in their real ANOVA-rank order.
USABLE_FEATURES = [
    "Bwd Packet Length Std", "Bwd Packet Length Mean", "Bwd Packet Length Max",
    "Packet Length Std", "Fwd IAT Std", "Max Packet Length",
    "Packet Length Variance", "Idle Min", "Idle Mean", "Average Packet Size",
    "Idle Max", "Packet Length Mean", "Flow IAT Max", "Fwd IAT Max",
    "Flow IAT Std", "PSH Flag Count", "Fwd IAT Total", "Flow Duration",
    "FIN Flag Count", "min_seg_size_forward", "Bwd IAT Std", "Flow IAT Mean",
    "Min Packet Length", "ACK Flag Count", "Bwd Packet Length Min",
    "Fwd IAT Mean", "Bwd IAT Max", "Flow IAT Min", "Idle Std",
    "Init_Win_bytes_forward", "Down/Up Ratio", "Destination Port",
    "URG Flag Count", "Bwd Packets/s", "Fwd PSH Flags",
    "Fwd Packet Length Min", "Bwd IAT Total", "Fwd Packets/s",
    "Flow Packets/s", "Fwd IAT Min", "Fwd Packet Length Mean",
    "Init_Win_bytes_backward", "Fwd Packet Length Max", "Fwd Packet Length Std",
    "Bwd IAT Mean", "Active Min", "Active Mean", "Active Max", "Bwd IAT Min",
    "Active Std", "Flow Bytes/s", "Total Length of Fwd Packets",
    "ECE Flag Count", "RST Flag Count", "Fwd URG Flags", "Bwd Header Length",
    "Fwd Header Length", "Total Fwd Packets", "Total Backward Packets",
    "act_data_pkt_fwd", "Total Length of Bwd Packets",
]
assert len(USABLE_FEATURES) == 61

# 8 constant columns that feature_cleaning.py drops.
CONSTANT_FEATURES = [
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

# 9 duplicate columns that feature_cleaning.py drops.
DUPLICATE_FEATURES = [
    "Subflow Fwd Packets", "Subflow Bwd Packets", "Fwd Header Length.1",
    "CWE Flag Count", "SYN Flag Count", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Subflow Fwd Bytes", "Subflow Bwd Bytes",
]

# First 18 usable features carry class-conditional signal so ANOVA
# selection and the classifier have real structure to find; the rest
# are shared-distribution noise.
INFORMATIVE_FEATURES = USABLE_FEATURES[:18]
NOISE_FEATURES = USABLE_FEATURES[18:]

BINARY_FLAG_FEATURES = {
    "FIN Flag Count", "ACK Flag Count", "URG Flag Count",
    "ECE Flag Count", "RST Flag Count", "Fwd PSH Flags", "Fwd URG Flags",
}


def generate_class_block(label, class_idx, n_classes, rng):
    n = ROWS_PER_CLASS
    data = {}

    # ------------------------------------------------------------
    # Informative features: class-conditional Gaussian clusters.
    # Each feature gets its own random ordering of classes so the
    # resulting ANOVA ranking isn't just a trivial linear pattern.
    # ------------------------------------------------------------
    for i, feat in enumerate(INFORMATIVE_FEATURES):
        base_mean = 50.0 + i * 12.0
        std = base_mean * 0.12 + 1.0
        order = rng.permutation(n_classes)
        position = np.where(order == class_idx)[0][0]
        shift = base_mean * 0.55 * position
        data[feat] = np.clip(rng.normal(base_mean + shift, std, size=n), 0, None)

    # ------------------------------------------------------------
    # Noise features: shared distribution across all classes.
    # ------------------------------------------------------------
    for feat in NOISE_FEATURES:
        if feat in BINARY_FLAG_FEATURES:
            data[feat] = rng.binomial(1, 0.3, size=n)
        elif feat == "Destination Port":
            data[feat] = rng.choice(
                [80, 443, 21, 22, 53, 3389, 8080, 8443], size=n
            )
        elif feat == "Down/Up Ratio":
            data[feat] = np.round(rng.uniform(0.1, 5.0, size=n), 2)
        elif feat.startswith("Init_Win_bytes"):
            data[feat] = rng.integers(0, 65536, size=n)
        else:
            data[feat] = np.clip(rng.normal(20.0, 8.0, size=n), 0, None)

    # ------------------------------------------------------------
    # Constant features: literally constant, matching their real role.
    # ------------------------------------------------------------
    for feat in CONSTANT_FEATURES:
        data[feat] = np.zeros(n)

    # ------------------------------------------------------------
    # Duplicate features: mirror an existing informative/noise column.
    # ------------------------------------------------------------
    data["Subflow Fwd Packets"] = data["Total Fwd Packets"].copy()
    data["Subflow Bwd Packets"] = data["Total Backward Packets"].copy()
    data["Fwd Header Length.1"] = data["Fwd Header Length"].copy()
    data["CWE Flag Count"] = np.zeros(n)
    data["SYN Flag Count"] = rng.binomial(1, 0.3, size=n)
    data["Avg Fwd Segment Size"] = data["Fwd Packet Length Mean"].copy()
    data["Avg Bwd Segment Size"] = data["Bwd Packet Length Mean"].copy()
    data["Subflow Fwd Bytes"] = data["Total Length of Fwd Packets"].copy()
    data["Subflow Bwd Bytes"] = data["Total Length of Bwd Packets"].copy()

    data["Label"] = label

    return pd.DataFrame(data)


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    n_classes = len(TARGET_LABELS)

    all_columns = (
        USABLE_FEATURES + CONSTANT_FEATURES + DUPLICATE_FEATURES + ["Label"]
    )
    assert len(all_columns) == 78 + 1

    blocks = {
        label: generate_class_block(label, idx, n_classes, rng)[all_columns]
        for idx, label in enumerate(TARGET_LABELS)
    }

    # A small slice of an out-of-scope label so preprocessing.py's
    # SELECTED_LABELS filter has something real to drop.
    noise_block = generate_class_block("Heartbleed", 0, n_classes, rng)
    noise_block = noise_block.iloc[:30].copy()
    noise_block["Label"] = "Heartbleed"
    noise_block = noise_block[all_columns]

    # Split into 3 raw files, mirroring CICIDS2017 shipping as
    # multiple day-based CSVs (exercises preprocessing.py's glob).
    file1 = pd.concat(
        [blocks["BENIGN"], blocks["DoS Hulk"], blocks["DDoS"], noise_block],
        ignore_index=True,
    )
    file2 = pd.concat(
        [blocks["PortScan"], blocks["DoS GoldenEye"]], ignore_index=True
    )
    file3 = pd.concat(
        [blocks["FTP-Patator"], blocks["SSH-Patator"]], ignore_index=True
    )

    files = {
        "synthetic_day1.csv": file1,
        "synthetic_day2.csv": file2,
        "synthetic_day3.csv": file3,
    }

    for name, df in files.items():
        out_path = RAW_DIR / name
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df):,} rows x {len(df.columns)} cols -> {out_path}")

    print("\nSynthetic raw data generation complete.")


if __name__ == "__main__":
    main()
