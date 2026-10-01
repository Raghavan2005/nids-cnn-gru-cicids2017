"""
LOAO RESULT ANALYSIS
Paper 1 - CNN-GRU Zero-Day Generalization Analysis

Reads:
    results/loao/loao_summary.csv

Creates:
    results/loao/loao_metrics.png
    results/loao/loao_recall.png
    results/loao/loao_f1.png
    results/loao/loao_analysis.txt
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = str(Path(__file__).resolve().parent.parent)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "loao",
    "loao_summary.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "loao"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOAO RESULT ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"LOAO summary file not found:\n{INPUT_FILE}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    print("\nLoading:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("\nLOAO RESULTS")
    print("-" * 70)

    print(
        df[
            [
                "unseen_attack",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "threshold"
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # CONVERT TO PERCENTAGES
    # ========================================================

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    for metric in metrics:
        df[metric + "_pct"] = df[metric] * 100

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    mean_recall = df["recall"].mean()
    mean_f1 = df["f1"].mean()

    best_recall_row = df.loc[df["recall"].idxmax()]
    worst_recall_row = df.loc[df["recall"].idxmin()]

    print(
        f"\nMean LOAO Recall : {mean_recall * 100:.2f}%"
    )

    print(
        f"Mean LOAO F1     : {mean_f1 * 100:.2f}%"
    )

    print(
        f"\nBest unseen attack recall:"
        f" {best_recall_row['unseen_attack']} "
        f"({best_recall_row['recall'] * 100:.2f}%)"
    )

    print(
        f"Worst unseen attack recall:"
        f" {worst_recall_row['unseen_attack']} "
        f"({worst_recall_row['recall'] * 100:.2f}%)"
    )

    # ========================================================
    # GRAPH 1
    # ALL METRICS
    # ========================================================

    attacks = df["unseen_attack"]

    plt.figure(figsize=(12, 7))

    x = range(len(attacks))

    width = 0.2

    plt.bar(
        [i - 1.5 * width for i in x],
        df["accuracy_pct"],
        width=width,
        label="Accuracy"
    )

    plt.bar(
        [i - 0.5 * width for i in x],
        df["precision_pct"],
        width=width,
        label="Precision"
    )

    plt.bar(
        [i + 0.5 * width for i in x],
        df["recall_pct"],
        width=width,
        label="Recall"
    )

    plt.bar(
        [i + 1.5 * width for i in x],
        df["f1_pct"],
        width=width,
        label="F1"
    )

    plt.xticks(
        list(x),
        attacks,
        rotation=30,
        ha="right"
    )

    plt.ylabel("Score (%)")
    plt.xlabel("Unseen Attack")
    plt.title("LOAO Zero-Day Generalization Performance")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        "loao_metrics.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:\n{output_file}"
    )

    # ========================================================
    # GRAPH 2
    # RECALL
    # ========================================================

    plt.figure(figsize=(11, 6))

    plt.bar(
        attacks,
        df["recall_pct"]
    )

    plt.ylabel("Recall (%)")
    plt.xlabel("Unseen Attack")
    plt.title("LOAO Recall for Unseen Attack Families")
    plt.ylim(0, 100)

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        "loao_recall.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{output_file}"
    )

    # ========================================================
    # GRAPH 3
    # F1
    # ========================================================

    plt.figure(figsize=(11, 6))

    plt.bar(
        attacks,
        df["f1_pct"]
    )

    plt.ylabel("F1 Score (%)")
    plt.xlabel("Unseen Attack")
    plt.title("LOAO F1 Score for Unseen Attack Families")
    plt.ylim(0, 100)

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        "loao_f1.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{output_file}"
    )

    # ========================================================
    # WRITE TEXT ANALYSIS
    # ========================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        "loao_analysis.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 70 + "\n")
        f.write("LOAO ZERO-DAY GENERALIZATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "The Leave-One-Attack-Out (LOAO) experiment evaluates "
            "the ability of the CNN-GRU model to recognize an attack "
            "family that was excluded from training.\n\n"
        )

        f.write("RESULTS\n")
        f.write("-" * 70 + "\n\n")

        for _, row in df.iterrows():

            f.write(
                f"{row['unseen_attack']}\n"
            )

            f.write(
                f"  Accuracy  : {row['accuracy'] * 100:.2f}%\n"
            )

            f.write(
                f"  Precision : {row['precision'] * 100:.2f}%\n"
            )

            f.write(
                f"  Recall    : {row['recall'] * 100:.2f}%\n"
            )

            f.write(
                f"  F1        : {row['f1'] * 100:.2f}%\n"
            )

            f.write(
                f"  Threshold : {row['threshold']}\n\n"
            )

        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Mean Recall : {mean_recall * 100:.2f}%\n"
        )

        f.write(
            f"Mean F1     : {mean_f1 * 100:.2f}%\n\n"
        )

        f.write(
            "The results show a substantial reduction in recall "
            "when an attack family is completely excluded from "
            "training. Although the overall accuracy remains high, "
            "the low recall and F1 scores indicate that the model "
            "does not reliably identify unseen attack families.\n\n"
        )

        f.write(
            "This provides evidence that conventional CNN-GRU "
            "classification performance on known attack classes "
            "does not necessarily translate into strong zero-day "
            "generalization.\n\n"
        )

        f.write(
            "The finding motivates further analysis of the learned "
            "feature representations to determine whether attack "
            "families form distinct clusters and whether the model "
            "learns attack-family-dependent representations.\n"
        )

    print(
        f"Saved:\n{output_file}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("LOAO ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        "1. results\\loao\\loao_metrics.png"
    )

    print(
        "2. results\\loao\\loao_recall.png"
    )

    print(
        "3. results\\loao\\loao_f1.png"
    )

    print(
        "4. results\\loao\\loao_analysis.txt"
    )


if __name__ == "__main__":
    main()