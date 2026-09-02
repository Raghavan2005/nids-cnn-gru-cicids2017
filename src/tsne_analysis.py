"""
CICIDS2017 t-SNE Feature-Space Analysis
----------------------------------------

Purpose:
    Visualize the Top-30 ANOVA feature space in 2D using t-SNE.

Input:
    data/processed/scaled/train_top_30_scaled.csv

Output:
    results/feature_analysis/tsne_top30.png
    results/feature_analysis/tsne_summary.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\PS26-PAPER1"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "scaled",
    "train_top_30_scaled.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "feature_analysis"
)

OUTPUT_IMAGE = os.path.join(
    OUTPUT_DIR,
    "tsne_top30.png"
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "tsne_summary.txt"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Number of samples PER CLASS
SAMPLES_PER_CLASS = 2000

# t-SNE parameters
PERPLEXITY = 30
LEARNING_RATE = "auto"
N_ITER = 1000


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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CICIDS2017 t-SNE FEATURE-SPACE ANALYSIS")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("\nDataset shape:")
    print(df.shape)

    # --------------------------------------------------------
    # IDENTIFY LABEL
    # --------------------------------------------------------

    if "Label" not in df.columns:
        raise ValueError(
            "Label column not found in the dataset."
        )

    feature_columns = [
        column for column in df.columns
        if column != "Label"
    ]

    print("\nNumber of features:", len(feature_columns))
    print("Number of samples :", len(df))

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ORIGINAL CLASS DISTRIBUTION")
    print("-" * 70)

    class_counts = df["Label"].value_counts().sort_index()

    for class_id, count in class_counts.items():

        class_name = LABEL_NAMES.get(
            int(class_id),
            str(class_id)
        )

        print(
            f"{int(class_id)} | "
            f"{class_name:20s} | "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # BALANCED SAMPLING
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CREATING BALANCED t-SNE SAMPLE")
    print("-" * 70)

    sampled_parts = []

    rng = np.random.RandomState(RANDOM_STATE)

    for class_id in sorted(df["Label"].unique()):

        class_df = df[
            df["Label"] == class_id
        ]

        sample_size = min(
            SAMPLES_PER_CLASS,
            len(class_df)
        )

        sampled_class = class_df.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

        sampled_parts.append(sampled_class)

        class_name = LABEL_NAMES.get(
            int(class_id),
            str(class_id)
        )

        print(
            f"{int(class_id)} | "
            f"{class_name:20s} | "
            f"{sample_size:,} samples"
        )

    sampled_df = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    # Shuffle the final sample
    sampled_df = sampled_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print("\nTotal t-SNE samples:")
    print(f"{len(sampled_df):,}")

    # --------------------------------------------------------
    # PREPARE X AND Y
    # --------------------------------------------------------

    X = sampled_df[feature_columns].values.astype(
        np.float32
    )

    y = sampled_df["Label"].values.astype(int)

    print("\nFeature matrix:")
    print(X.shape)

    print("\nLabel vector:")
    print(y.shape)

    # --------------------------------------------------------
    # RUN t-SNE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING t-SNE")
    print("=" * 70)

    print("\nParameters:")
    print("Perplexity   :", PERPLEXITY)
    print("Learning rate:", LEARNING_RATE)
    print("Iterations   :", N_ITER)
    print("Random state :", RANDOM_STATE)

    print("\nThis may take several minutes...")

    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        learning_rate=LEARNING_RATE,
        max_iter=N_ITER,
        random_state=RANDOM_STATE,
        init="pca",
        verbose=1
    )

    X_tsne = tsne.fit_transform(X)

    print("\nt-SNE completed.")

    print("t-SNE output shape:")
    print(X_tsne.shape)

    # --------------------------------------------------------
    # CREATE PLOT
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CREATING t-SNE VISUALIZATION")
    print("-" * 70)

    plt.figure(figsize=(12, 9))

    for class_id in sorted(np.unique(y)):

        mask = y == class_id

        class_name = LABEL_NAMES.get(
            int(class_id),
            str(class_id)
        )

        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=8,
            alpha=0.6,
            label=class_name
        )

    plt.title(
        "t-SNE Visualization of Top-30 ANOVA Features",
        fontsize=16
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    plt.legend(
        loc="best",
        markerscale=2
    )

    plt.grid(
        True,
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_IMAGE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved:")
    print(OUTPUT_IMAGE)

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "CICIDS2017 t-SNE FEATURE-SPACE ANALYSIS\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write(
            "Feature configuration: Top-30 ANOVA features\n"
        )

        f.write(
            f"Original samples: {len(df):,}\n"
        )

        f.write(
            f"t-SNE samples: {len(sampled_df):,}\n"
        )

        f.write(
            f"Samples per class: {SAMPLES_PER_CLASS:,}\n"
        )

        f.write(
            f"Perplexity: {PERPLEXITY}\n"
        )

        f.write(
            f"Iterations: {N_ITER}\n"
        )

        f.write(
            f"Random state: {RANDOM_STATE}\n\n"
        )

        f.write(
            "Sampled class distribution\n"
        )
        f.write("-" * 70 + "\n")

        sampled_counts = (
            sampled_df["Label"]
            .value_counts()
            .sort_index()
        )

        for class_id, count in sampled_counts.items():

            class_name = LABEL_NAMES.get(
                int(class_id),
                str(class_id)
            )

            f.write(
                f"{int(class_id)} | "
                f"{class_name:20s} | "
                f"{count:,}\n"
            )

        f.write("\n")
        f.write(
            "Interpretation:\n"
        )

        f.write(
            "The t-SNE plot provides a two-dimensional "
            "visualization of the Top-30 ANOVA feature space. "
            "Clusters that are well separated indicate that "
            "the corresponding classes have distinguishable "
            "feature representations, while overlapping regions "
            "indicate similarities between classes.\n"
        )

    print("\nSaved:")
    print(OUTPUT_SUMMARY)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("t-SNE ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")
    print("1.", OUTPUT_IMAGE)
    print("2.", OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()