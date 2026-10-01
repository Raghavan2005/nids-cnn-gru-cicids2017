"""
PAPER 1 FINAL FIGURE GENERATION
CICIDS2017 CNN-GRU

Uses already-generated results.
NO MODEL TRAINING.
NO DATA MODIFICATION.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
FINAL_DIR = RESULTS_DIR / "final"

FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

feature_file = FINAL_DIR / "cnn_gru_feature_comparison.csv"
loao_file = FINAL_DIR / "loao_results.csv"

print("=" * 70)
print("PAPER 1 FINAL FIGURE GENERATION")
print("=" * 70)

print("\nLoading feature comparison:")
print(feature_file)

feature_df = pd.read_csv(feature_file)

print("\nLoading LOAO results:")
print(loao_file)

loao_df = pd.read_csv(loao_file)


# ============================================================
# DISPLAY DATA
# ============================================================

print("\n" + "=" * 70)
print("FEATURE COMPARISON")
print("=" * 70)

print(feature_df.to_string(index=False))

print("\n" + "=" * 70)
print("LOAO RESULTS")
print("=" * 70)

print(loao_df.to_string(index=False))


# ============================================================
# 1. MACRO F1 COMPARISON
# ============================================================

print("\nCreating Macro-F1 figure...")

plt.figure(figsize=(8, 5))

plt.plot(
    feature_df["Features"],
    feature_df["Macro_F1"],
    marker="o",
    linewidth=2
)

plt.xlabel("Number of Selected Features")
plt.ylabel("Macro F1")
plt.title("CNN-GRU Performance Across Feature Configurations")

plt.xticks(feature_df["Features"])
plt.ylim(
    max(0.0, feature_df["Macro_F1"].min() - 0.01),
    1.001
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

output = FINAL_DIR / "feature_macro_f1.png"
plt.savefig(output, dpi=300)
plt.close()

print("Saved:")
print(output)


# ============================================================
# 2. ACCURACY COMPARISON
# ============================================================

print("\nCreating Accuracy figure...")

plt.figure(figsize=(8, 5))

plt.plot(
    feature_df["Features"],
    feature_df["Accuracy"],
    marker="o",
    linewidth=2
)

plt.xlabel("Number of Selected Features")
plt.ylabel("Accuracy")
plt.title("CNN-GRU Accuracy Across Feature Configurations")

plt.xticks(feature_df["Features"])
plt.ylim(
    max(0.0, feature_df["Accuracy"].min() - 0.01),
    1.001
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

output = FINAL_DIR / "feature_accuracy.png"
plt.savefig(output, dpi=300)
plt.close()

print("Saved:")
print(output)


# ============================================================
# 3. LOAO RECALL
# ============================================================

print("\nCreating LOAO Recall figure...")

plt.figure(figsize=(10, 5))

plt.bar(
    loao_df["unseen_attack"],
    loao_df["recall"]
)

plt.xlabel("Unseen Attack Class")
plt.ylabel("Recall")
plt.title("LOAO Generalization: Recall on Unseen Attacks")

plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1.0)

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

output = FINAL_DIR / "loao_recall.png"
plt.savefig(output, dpi=300)
plt.close()

print("Saved:")
print(output)


# ============================================================
# 4. LOAO F1
# ============================================================

print("\nCreating LOAO F1 figure...")

plt.figure(figsize=(10, 5))

plt.bar(
    loao_df["unseen_attack"],
    loao_df["f1"]
)

plt.xlabel("Unseen Attack Class")
plt.ylabel("F1 Score")
plt.title("LOAO Generalization: F1 Score on Unseen Attacks")

plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1.0)

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

output = FINAL_DIR / "loao_f1.png"
plt.savefig(output, dpi=300)
plt.close()

print("Saved:")
print(output)


# ============================================================
# 5. COMBINED NUMERICAL SUMMARY
# ============================================================

print("\nCreating final numerical summary...")

summary_file = FINAL_DIR / "paper1_figure_summary.txt"

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("PAPER 1 FINAL FIGURE SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("CONVENTIONAL TEST PERFORMANCE\n")
    f.write("-" * 70 + "\n")

    for _, row in feature_df.iterrows():

        f.write(
            f"Top-{int(row['Features'])}: "
            f"Accuracy={row['Accuracy']:.4f}, "
            f"Balanced Accuracy={row['Balanced_Accuracy']:.4f}, "
            f"Macro F1={row['Macro_F1']:.4f}, "
            f"Weighted F1={row['Weighted_F1']:.4f}\n"
        )

    f.write("\n")
    f.write("LOAO UNSEEN-ATTACK GENERALIZATION\n")
    f.write("-" * 70 + "\n")

    for _, row in loao_df.iterrows():

        f.write(
            f"{row['unseen_attack']}: "
            f"Accuracy={row['accuracy']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}\n"
        )

    f.write("\n")
    f.write(
        f"Mean LOAO Recall: "
        f"{loao_df['recall'].mean():.4f}\n"
    )

    f.write(
        f"Mean LOAO F1: "
        f"{loao_df['f1'].mean():.4f}\n"
    )


print("Saved:")
print(summary_file)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL FIGURE GENERATION COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print("1.", FINAL_DIR / "feature_macro_f1.png")
print("2.", FINAL_DIR / "feature_accuracy.png")
print("3.", FINAL_DIR / "loao_recall.png")
print("4.", FINAL_DIR / "loao_f1.png")
print("5.", FINAL_DIR / "paper1_figure_summary.txt")

print("\nThese figures use your existing experimental results.")
print("No model training was performed.")